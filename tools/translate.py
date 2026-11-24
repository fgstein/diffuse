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
import glob
import click
import numpy as np
import torch
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib # pylint: disable=wrong-import-position
from torch_utils import distributed as dist # pylint: disable=wrong-import-position
from training.dataset import load_image # pylint: disable=wrong-import-position
from training.sampling import StackedRandomGenerator, edm_sampler, to_uint8, make_grid, save_image # pylint: disable=wrong-import-position

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]

def list_inputs(csv_path, image_root, input_dir, path_col, label_col):
    """Return (root, [(relative path, label or '')])."""
    if input_dir is not None:
        files = sorted(f for f in glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
                       if f.lower().endswith(IMAGE_EXTS))
        if not files:
            raise click.ClickException(f'No images found under {input_dir}')
        return input_dir, [(os.path.relpath(f, input_dir), '') for f in files]
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for c in [path_col, label_col]:
        if c is not None and c not in df.columns:
            raise click.ClickException(f'Column "{c}" not in {csv_path}; available: {list(df.columns)}')
    labels = df[label_col].tolist() if label_col else [''] * len(df)
    return image_root, list(zip(df[path_col].tolist(), labels))

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
    root, items = list_inputs(csv_path, image_root, input_dir, path_col, label_col)
    if max_images:
        items = items[:max_images]

    # Conditioning: unconditional unless a label column is given.
    label_to_idx = dict(label_json['label_to_idx']) if label_json else {}
    conditional = label_col is not None
    if conditional:
        if 'class' not in net.cond_stack.names:
            raise click.ClickException('This network is unconditional; drop --label-col.')
        unknown = sorted({l for _, l in items if l and l not in label_to_idx})
        if unknown:
            raise click.ClickException(f'Labels not in the snapshot label map: {unknown[:10]}')
    null_idx = net.cond_stack.encoders['class'].null_index if 'class' in net.cond_stack.names else None
    mode = f'conditional (guidance {guidance})' if conditional else 'unconditional'
    print(f'{len(items)} input(s) at {res}x{res}, {mode}, sigma_start {sigmas}, {num_steps} steps'
          + (f', ILVR factor {ilvr_factor}' if ilvr_factor > 1 else '') + f', {samples_per_image} variant(s) each')

    # Jobs: (input index, variant k).
    jobs = [(i, k) for i in range(len(items)) for k in range(samples_per_image)]
    records = []
    compare = {} # input index -> {'in': img, sigma: img}
    os.makedirs(outdir, exist_ok=True)
    for s in sigmas:
        for start in tqdm.trange(0, len(jobs), batch_size, desc=sigma_tag(s), unit='batch'):
            batch = jobs[start:start + batch_size]
            imgs = []
            for i, _k in batch:
                rel = items[i][0]
                try:
                    with open(os.path.join(root, rel), 'rb') as f:
                        imgs.append(load_image(f, resolution=res, channels=C, name=rel))
                except Exception as e: # pylint: disable=broad-except
                    raise click.ClickException(f'Cannot load "{rel}": {e}')
            x_sim = torch.as_tensor(np.stack(imgs)).to(device).to(torch.float32) / 127.5 - 1
            seeds = [seed + i * samples_per_image + k for i, k in batch]
            rnd = StackedRandomGenerator(device, seeds)
            eps = rnd.randn(list(x_sim.shape), device=device)
            cond = None
            if null_idx is not None:
                idx = [label_to_idx[items[i][1]] if (conditional and items[i][1]) else null_idx for i, _ in batch]
                cond = {'class': torch.as_tensor(idx, dtype=torch.int64, device=device)}
            with torch.no_grad():
                out = edm_sampler(net, eps, cond, randn_like=rnd.randn_like, num_steps=num_steps,
                                  guidance=(guidance if conditional else 1.0), x_init=x_sim, sigma_start=s,
                                  ilvr_ref=(x_sim if ilvr_factor > 1 else None), ilvr_factor=ilvr_factor,
                                  ilvr_sigma_min=ilvr_sigma_min)
            out = to_uint8(out).cpu()
            for (i, k), sd, img in zip(batch, seeds, out):
                rel = items[i][0]
                orel = os.path.join(sigma_tag(s), out_rel(rel, k, samples_per_image))
                os.makedirs(os.path.dirname(os.path.join(outdir, orel)), exist_ok=True)
                save_image(img.permute(1, 2, 0).numpy(), os.path.join(outdir, orel))
                records.append(dict(source=rel, output=orel.replace('\\', '/'), sigma_start=s, sample=k, seed=sd,
                                    label=items[i][1] if conditional else '', guidance=guidance if conditional else '',
                                    ilvr_factor=ilvr_factor, steps=num_steps))
                if i < num_compare and k == 0:
                    entry = compare.setdefault(i, {'in': to_uint8(x_sim[batch.index((i, k))]).cpu()})
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
