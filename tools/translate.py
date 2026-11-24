# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Sim-to-real image translation with a trained model (SDEdit, optionally ILVR).

Each input image x_sim is noised to sigma_start and denoised by the model,
which was trained on REAL images only:

    x = x_sim + sigma_start * eps  ->  EDM Heun sampler from sigma_start to 0

Detail finer than the scale ~sigma_start (grain, sensor noise, blur, fine
texture) is regenerated "the real way"; the coarser scene is kept. With
--ilvr-factor N, the low frequencies (below ~1/N of the image size) are forced
to follow the input at every step, which keeps the scene even at higher
sigma_start (Choi et al. 2021).

By default the translation is UNCONDITIONAL (NULL label): no class needed,
and classes that are weak or collapsed in the conditional model cannot take
over. Give --label-col to condition on the class of each input instead.

Outputs, per sigma_start value:
    outdir/sigma-<s>/<relative path>.png       (or _k<k>.png with --samples-per-image > 1)
    outdir/metadata.csv                        source, output, sigma_start, seed, label, ...
    outdir/compare.png                         input | sigma_1 | sigma_2 | ... for the first images

To choose sigma_start / ILVR visually and quantitatively, use tools/style_mosaic.py.

Examples:
    # sweep sigma_start on 32 images, unconditional
    python tools/translate.py --network snap.pkl --csv sim.csv --image-root /data/sim \\
        --sigma-start 0.05,0.1,0.2,0.4 --max-images 32 --outdir sweep

    # full set, chosen sigma, ILVR to lock the scene, 2 variants per image
    python tools/translate.py --network snap.pkl --input-dir /data/sim --sigma-start 0.2 \\
        --ilvr-factor 8 --samples-per-image 2 --outdir sim2real
"""

import os
import sys
import csv
import click
import torch
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_utils import distributed as dist # pylint: disable=wrong-import-position
from training.sampling import make_grid, save_image, to_uint8 # pylint: disable=wrong-import-position
from training import translation as tr # pylint: disable=wrong-import-position

def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]

def sigma_tag(s):
    return f'sigma-{s:g}'

def out_rel(rel, k, samples_per_image):
    base = os.path.splitext(rel.replace('\\', '/'))[0]
    return f'{base}_k{k}.png' if samples_per_image > 1 else f'{base}.png'

@click.command(context_settings=dict(max_content_width=120))
@click.option('--network', 'network_pkl', help='Network snapshot trained on REAL images', metavar='PKL', required=True)
@click.option('--csv', 'csv_path', help='CSV listing the simulated images', metavar='PATH', default=None)
@click.option('--image-root', help='Root of the relative paths in --csv', metavar='DIR', default=None)
@click.option('--input-dir', help='Alternative to --csv: translate every image under DIR', metavar='DIR', default=None)
@click.option('--path-col', help='CSV column with relative paths', default='path', show_default=True)
@click.option('--label-col', help='CSV column with class labels (omit = unconditional)', default=None)
@click.option('--outdir', help='Output directory', metavar='DIR', required=True)
@click.option('--sigma-start', help='Starting noise level(s); a list runs a sweep', metavar='LIST', default='0.1', show_default=True)
@click.option('--steps', 'num_steps', help='Heun steps from sigma_start to 0', type=click.IntRange(min=2), default=16, show_default=True)
@click.option('--guidance', help='CFG weight (only with --label-col)', type=float, default=1.0, show_default=True)
@click.option('--ilvr-factor', help='ILVR low-pass factor N (0 = off); must divide the resolution', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--ilvr-sigma-min', help='Stop ILVR below this noise level', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--samples-per-image', help='Stochastic variants per input', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--seed', help='Base seed (seed of image i, variant k = seed + i * K + k)', type=int, default=0, show_default=True)
@click.option('--max-images', help='Only the first N inputs (for sweeps)', type=click.IntRange(min=1), default=None)
@click.option('--compare', 'num_compare', help='Inputs shown in compare.png (0 = none)', type=click.IntRange(min=0), default=8, show_default=True)
@click.option('--batch', 'batch_size', help='Batch size', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--ema-halflife', help='Pick a secondary EMA by half-life', metavar='MIMG', type=float, default=None)
@click.option('--device', help='Device [default: cuda if available]', type=click.Choice(['cuda', 'cpu']), default=None)
def main(network_pkl, csv_path, image_root, input_dir, path_col, label_col, outdir, sigma_start, num_steps, guidance,
         ilvr_factor, ilvr_sigma_min, samples_per_image, seed, max_images, num_compare, batch_size, ema_halflife, device):
    """Translate simulated images toward the real domain (SDEdit / ILVR)."""
    from generate import load_network
    if (input_dir is None) == (csv_path is None):
        raise click.ClickException('Give exactly one of --input-dir or --csv (with --image-root).')
    if csv_path is not None and image_root is None:
        raise click.ClickException('--csv requires --image-root.')
    if input_dir is not None and label_col is not None:
        raise click.ClickException('--label-col requires --csv.')
    sigmas = parse_float_list(sigma_start)
    device_type = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dist.init(device_type)
    device = torch.device('cuda', torch.cuda.current_device()) if device_type == 'cuda' else torch.device('cpu')

    net, label_json, _noise_config = load_network(network_pkl, device, ema_halflife)
    res, C = net.img_resolution, net.img_channels
    try:
        root, items = tr.list_inputs(csv_path, image_root, input_dir, path_col, label_col)
    except IOError as e:
        raise click.ClickException(str(e))
    if max_images:
        items = items[:max_images]

    # Conditioning: unconditional (NULL label) unless a label column is given.
    conditional = label_col is not None
    label_to_idx = None
    if conditional:
        try:
            label_to_idx = tr.check_labels(net, label_json, [l for _, l in items])
        except ValueError as e:
            raise click.ClickException(str(e))
    mode = f'conditional (guidance {guidance})' if conditional else 'unconditional'
    print(f'{len(items)} input(s) at {res}x{res}, {mode}, sigma_start {sigmas}, {num_steps} steps'
          + (f', ILVR factor {ilvr_factor}' if ilvr_factor > 1 else '') + f', {samples_per_image} variant(s) each')

    # Jobs: (input index, variant k). Seed of (i, k) = seed + i * K + k, identical for every sigma.
    jobs = [(i, k) for i in range(len(items)) for k in range(samples_per_image)]
    records = []
    compare = {} # input index -> {'in': img, sigma: img}
    os.makedirs(outdir, exist_ok=True)
    for s in sigmas:
        for start in tqdm.trange(0, len(jobs), batch_size, desc=sigma_tag(s), unit='batch'):
            batch = jobs[start:start + batch_size]
            try:
                x_sim = tr.load_inputs(root, [items[i][0] for i, _ in batch], res, C)
            except IOError as e:
                raise click.ClickException(str(e))
            seeds = [seed + i * samples_per_image + k for i, k in batch]
            cond = tr.make_cond(net, [items[i][1] for i, _ in batch], label_to_idx, device)
            out = tr.translate(net, x_sim, seeds, s, device, cond=cond, num_steps=num_steps,
                               guidance=(guidance if conditional else 1.0), ilvr_factor=ilvr_factor,
                               ilvr_sigma_min=ilvr_sigma_min, batch_size=len(batch))
            for j, ((i, k), sd, img) in enumerate(zip(batch, seeds, out)):
                rel = items[i][0]
                orel = os.path.join(sigma_tag(s), out_rel(rel, k, samples_per_image))
                os.makedirs(os.path.dirname(os.path.join(outdir, orel)), exist_ok=True)
                save_image(img.permute(1, 2, 0).numpy(), os.path.join(outdir, orel))
                records.append(dict(source=rel, output=orel.replace('\\', '/'), sigma_start=s, sample=k, seed=sd,
                                    label=items[i][1] if conditional else '', guidance=guidance if conditional else '',
                                    ilvr_factor=ilvr_factor, steps=num_steps))
                if i < num_compare and k == 0:
                    entry = compare.setdefault(i, {'in': x_sim[j]})
                    entry[s] = img

    with open(os.path.join(outdir, 'metadata.csv'), 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    # compare.png: one row per input: input | sigma_1 | sigma_2 | ...
    if compare:
        rows = sorted(compare)
        tiles = [t for i in rows for t in [compare[i]['in']] + [compare[i][s] for s in sigmas]]
        save_image(make_grid(torch.stack(tiles), len(rows), 1 + len(sigmas)), os.path.join(outdir, 'compare.png'))
        print('compare.png columns: input | ' + ' | '.join(sigma_tag(s) for s in sigmas))
    print(f'Wrote {len(records)} image(s) to {outdir}')

if __name__ == '__main__':
    main()
