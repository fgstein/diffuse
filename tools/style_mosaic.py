# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Mosaics for tuning the style level of sim-to-real translation.

Translates a small selection of simulated images at several sigma_start values
(and optionally several ILVR factors), then writes annotated mosaics that make
the realism / fidelity trade-off visible:

  full_ilvr<N>.png    rows = images (x variants); columns = input | sigma_1 ... | real reference(s).
                      Column headers show grain / sharpness / high-frequency energy of the
                      column (relative to the real images if --real is given). The zoom
                      region is outlined on the input.
  zoom_ilvr<N>.png    same layout, zoomed crops (nearest-neighbour) of a detailed region:
                      the place to judge grain, noise and blur at full resolution.
  diff_ilvr<N>.png    |output - input| x gain: what each setting changed. Changes should be
                      fine-grained (texture, edges); visible shapes mean scene drift.
  grid2d_img<i>.png   (several ILVR factors) one image: rows = ILVR factor, columns = sigma.
  grid2d_img<i>_zoom.png  same, zoomed.
  metrics.png         grain / sharpness / HF energy / scene drift vs sigma_start, one curve per
                      ILVR factor, with the input and real levels: pick sigma_start where the
                      realism curves reach the real level while the drift stays small.
  metrics.csv, settings.json

N = 0 means plain SDEdit (no ILVR). The same seed is used for an image/variant in
every column, so differences between columns come from the settings only.

Example:
    python tools/style_mosaic.py --network snap.pkl --csv sim.csv --image-root /data/sim \\
        --real /data/real-holdout.csv@/data/images --images 8 \\
        --sigma-start 0.02,0.05,0.1,0.2,0.4,0.8 --ilvr-factors 0,8 --outdir mosaic
"""

import os
import sys
import csv
import json
import click
import numpy as np
import torch
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_utils import distributed as dist # pylint: disable=wrong-import-position
from training import translation as tr # pylint: disable=wrong-import-position
from training.sampling import lowpass # pylint: disable=wrong-import-position

#----------------------------------------------------------------------------
# Drawing helpers.

def _font(size):
    try:
        return PIL.ImageFont.load_default(size=size)
    except TypeError: # Pillow < 10.1
        return PIL.ImageFont.load_default()

def to_rgb(img_chw):
    """uint8 tensor/array [C, H, W] -> uint8 HWC RGB numpy."""
    a = np.asarray(img_chw).transpose(1, 2, 0)
    return np.repeat(a, 3, axis=2) if a.shape[2] == 1 else a

def upscale(a, factor):
    return a if factor == 1 else np.kron(a, np.ones((factor, factor, 1), dtype=np.uint8))

def zoomed_crop(img_chw, box, zoom):
    x, y, s = box
    return upscale(to_rgb(img_chw[:, y:y + s, x:x + s]), zoom)

def auto_crop_box(img_chw, size):
    """Square crop of `size` with the most edge energy (where blur / grain differences show)."""
    lum = tr.luminance(torch.as_tensor(img_chw)[None])[0]
    gx = lum[:, 1:] - lum[:, :-1]
    gy = lum[1:, :] - lum[:-1, :]
    e = torch.zeros_like(lum)
    e[:, 1:] += gx.abs()
    e[1:, :] += gy.abs()
    H, W = lum.shape
    if size >= min(H, W):
        return 0, 0, min(H, W)
    score = torch.nn.functional.avg_pool2d(e[None, None], size, stride=max(1, size // 4))[0, 0]
    iy, ix = divmod(int(score.argmax()), score.shape[1])
    s = max(1, size // 4)
    return min(ix * s, W - size), min(iy * s, H - size), size

def outline(a, box, scale, color=(255, 40, 40)):
    x, y, s = (v * scale for v in box)
    a = a.copy()
    t = max(1, scale)
    a[y:y + t, x:x + s] = color
    a[y + s - t:y + s, x:x + s] = color
    a[y:y + s, x:x + t] = color
    a[y:y + s, x + s - t:x + s] = color
    return a

def compose(rows, col_headers, row_labels, title, gap=4, label_w=150):
    """rows: list of lists of HWC uint8 arrays (None = empty). Returns a PIL image."""
    th, tw0 = next(t for r in rows for t in r if t is not None).shape[:2]
    tw = max(tw0, 132) # column pitch wide enough for the header text
    ncol = max(len(r) for r in rows)
    f_title, f_head, f_small = _font(16), _font(13), _font(11)
    head_h = 46
    title_h = 26
    W = label_w + ncol * (tw + gap) + gap
    H = title_h + head_h + len(rows) * (th + gap) + gap
    img = PIL.Image.new('RGB', (W, H), (255, 255, 255))
    d = PIL.ImageDraw.Draw(img)
    d.text((8, 5), title, fill=(0, 0, 0), font=f_title)
    for c, (h1, h2) in enumerate(col_headers):
        x = label_w + gap + c * (tw + gap)
        d.text((x + 2, title_h + 2), h1, fill=(0, 0, 0), font=f_head)
        for li, line in enumerate(h2.split('\n')):
            d.text((x + 2, title_h + 19 + 12 * li), line, fill=(90, 90, 90), font=f_small)
    for r, row in enumerate(rows):
        y = title_h + head_h + gap + r * (th + gap)
        for li, line in enumerate(row_labels[r].split('\n')):
            d.text((6, y + 2 + 13 * li), line[:24], fill=(0, 0, 0), font=f_small)
        for c, t in enumerate(row):
            if t is not None:
                img.paste(PIL.Image.fromarray(t), (label_w + gap + c * (tw + gap) + (tw - t.shape[1]) // 2, y))
    return img

def _fmt_stats(st, ref):
    if ref is None:
        return f"grain {st['grain']:.2f}  sharp {st['sharpness']:.1f}\nHF {st['hf_energy']:.3g}"
    return (f"grain x{st['grain'] / max(ref['grain'], 1e-8):.2f}  sharp x{st['sharpness'] / max(ref['sharpness'], 1e-8):.2f}\n"
            f"HF x{st['hf_energy'] / max(ref['hf_energy'], 1e-12):.2f}")

def drift(out_u8, in_u8, factor):
    """Scene drift: mean |LP(out) - LP(in)| in 0-255 units (LP = ILVR low-pass)."""
    a = lowpass(out_u8.to(torch.float64), factor)
    b = lowpass(in_u8.to(torch.float64), factor)
    return float((a - b).abs().mean())

def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]

def parse_int_list(s):
    return [int(x) for x in s.split(',') if x.strip()]

#----------------------------------------------------------------------------

@click.command(context_settings=dict(max_content_width=120))
@click.option('--network', 'network_pkl', help='Network snapshot trained on REAL images', metavar='PKL', required=True)
@click.option('--csv', 'csv_path', help='CSV listing the simulated images', metavar='PATH', default=None)
@click.option('--image-root', help='Root of the relative paths in --csv', metavar='DIR', default=None)
@click.option('--input-dir', help='Alternative to --csv: simulated images under DIR', metavar='DIR', default=None)
@click.option('--path-col', help='CSV column with relative paths', default='path', show_default=True)
@click.option('--label-col', help='CSV column with class labels (omit = unconditional)', default=None)
@click.option('--images', 'num_images', help='Number of simulated images in the mosaic', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--select', help='Which images: evenly spread over the list, the first ones, or random', type=click.Choice(['spread', 'first', 'random']), default='spread', show_default=True)
@click.option('--sigma-start', help='sigma_start values (columns)', metavar='LIST', default='0.02,0.05,0.1,0.2,0.4,0.8', show_default=True)
@click.option('--ilvr-factors', help='ILVR factors (0 = plain SDEdit); one mosaic per factor', metavar='LIST', default='0', show_default=True)
@click.option('--variants', help='Stochastic variants per image (extra rows)', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--steps', 'num_steps', help='Heun steps from sigma_start to 0', type=click.IntRange(min=2), default=16, show_default=True)
@click.option('--guidance', help='CFG weight (only with --label-col)', type=float, default=1.0, show_default=True)
@click.option('--ilvr-sigma-min', help='Stop ILVR below this noise level', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--real', 'real_path', help='Real reference images: dir, prepared zip/dir, or CSV@ROOT (held-out!)', default=None)
@click.option('--real-cols', help='Real reference columns in the mosaics', type=click.IntRange(min=0), default=1, show_default=True)
@click.option('--real-max', help='Max real images used for the reference statistics', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--crop', help="Zoom region: 'auto' (most edges), 'center', or X,Y (top-left, in pixels)", default='auto', show_default=True)
@click.option('--crop-size', help='Zoom region size in pixels [default: resolution/4]', type=click.IntRange(min=4), default=None)
@click.option('--crop-zoom', help='Zoom factor of the crops (raised so that zoom tiles are >= 128 px)', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--diff-gain', help='Amplification of the difference maps', type=float, default=4.0, show_default=True)
@click.option('--tile-scale', help='Upscale full tiles by this factor [default: auto, tiles >= 128 px]', type=click.IntRange(min=0), default=0)
@click.option('--grid2d-images', help='Images with a sigma x ILVR grid (when several factors)', type=click.IntRange(min=0), default=2, show_default=True)
@click.option('--max-rows', help='Rows per mosaic page', type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--seed', help='Base seed', type=int, default=0, show_default=True)
@click.option('--batch', 'batch_size', help='Batch size', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--ema-halflife', help='Pick a secondary EMA by half-life', metavar='MIMG', type=float, default=None)
@click.option('--device', help='Device [default: cuda if available]', type=click.Choice(['cuda', 'cpu']), default=None)
@click.option('--outdir', help='Output directory', metavar='DIR', required=True)
def main(network_pkl, csv_path, image_root, input_dir, path_col, label_col, num_images, select, sigma_start, ilvr_factors,
         variants, num_steps, guidance, ilvr_sigma_min, real_path, real_cols, real_max, crop, crop_size, crop_zoom,
         diff_gain, tile_scale, grid2d_images, max_rows, seed, batch_size, ema_halflife, device, outdir):
    """Annotated mosaics (full / zoom / difference / metrics) across sigma_start and ILVR factors."""
    from generate import load_network
    if (input_dir is None) == (csv_path is None):
        raise click.ClickException('Give exactly one of --input-dir or --csv (with --image-root).')
    if csv_path is not None and image_root is None:
        raise click.ClickException('--csv requires --image-root.')
    if input_dir is not None and label_col is not None:
        raise click.ClickException('--label-col requires --csv.')
    sigmas = parse_float_list(sigma_start)
    factors = parse_int_list(ilvr_factors)
    device_type = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dist.init(device_type)
    device = torch.device('cuda', torch.cuda.current_device()) if device_type == 'cuda' else torch.device('cpu')
    os.makedirs(outdir, exist_ok=True)

    net, label_json, _ = load_network(network_pkl, device, ema_halflife)
    res, C = net.img_resolution, net.img_channels
    for f in factors:
        if f > 1 and res % f:
            raise click.ClickException(f'ILVR factor {f} must divide the resolution {res}.')
    try:
        root, items = tr.list_inputs(csv_path, image_root, input_dir, path_col, label_col)
    except IOError as e:
        raise click.ClickException(str(e))

    # Select images.
    n = min(num_images, len(items))
    if select == 'first':
        sel = list(range(n))
    elif select == 'random':
        sel = sorted(np.random.RandomState(seed).permutation(len(items))[:n].tolist())
    else:
        sel = sorted(set(np.linspace(0, len(items) - 1, n).round().astype(int).tolist()))
    items = [items[i] for i in sel]
    x_in = tr.load_inputs(root, [rel for rel, _ in items], res, C)

    conditional = label_col is not None
    label_to_idx = None
    if conditional:
        try:
            label_to_idx = tr.check_labels(net, label_json, [l for _, l in items])
        except ValueError as e:
            raise click.ClickException(str(e))

    # Rows = (image, variant); identical seeds in every column.
    jobs = [(i, k) for i in range(len(items)) for k in range(variants)]
    x_jobs = x_in[[i for i, _ in jobs]]
    seeds = [seed + i * variants + k for i, k in jobs]
    cond = tr.make_cond(net, [items[i][1] for i, _ in jobs], label_to_idx, device)

    # Real reference.
    real = None
    if real_path is not None:
        try:
            real = tr.load_image_set(real_path, res, C, real_max, path_col, seed=seed)
        except IOError as e:
            raise click.ClickException(f'--real: {e}')

    # Translate.
    total = len(factors) * len(sigmas)
    print(f'{len(items)} image(s) x {variants} variant(s), {len(sigmas)} sigma_start x {len(factors)} ILVR setting(s) '
          f'at {res}x{res}, {"conditional" if conditional else "unconditional"}')
    outs = {}
    for f in factors:
        for s in sigmas:
            print(f'  [{len(outs) + 1}/{total}] ILVR {f}, sigma_start {s:g}', flush=True)
            outs[(f, s)] = tr.translate(net, x_jobs, seeds, s, device, cond=cond, num_steps=num_steps,
                                        guidance=(guidance if conditional else 1.0), ilvr_factor=f,
                                        ilvr_sigma_min=ilvr_sigma_min, batch_size=batch_size)

    # Statistics.
    drift_factor = max(2, res // 28)
    while res % drift_factor:
        drift_factor -= 1
    ref = tr.image_stats(real) if real is not None else None
    st_in = tr.image_stats(x_jobs)
    stats = {key: dict(tr.image_stats(o), drift=drift(o, x_jobs, drift_factor)) for key, o in outs.items()}
    rows_csv = [dict(setting='input', ilvr_factor='', sigma_start=0, drift=0.0, **st_in)]
    if ref is not None:
        rows_csv.append(dict(setting='real', ilvr_factor='', sigma_start='', drift='', **ref))
    for (f, s), st in stats.items():
        rows_csv.append(dict(setting='translated', ilvr_factor=f, sigma_start=s, **st))
    for r in rows_csv:
        if ref is not None:
            for k in ('grain', 'sharpness', 'hf_energy'):
                r[f'{k}_vs_real'] = r[k] / max(ref[k], 1e-12)
    with open(os.path.join(outdir, 'metrics.csv'), 'wt', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[-1].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    # Tiles.
    scale = tile_scale or max(1, int(np.ceil(128 / res)))
    cs = crop_size or max(4, res // 4)
    cs = min(cs, res)
    if crop == 'auto':
        boxes = [auto_crop_box(x_jobs[j], cs) for j in range(len(jobs))]
    elif crop == 'center':
        boxes = [((res - cs) // 2, (res - cs) // 2, cs)] * len(jobs)
    else:
        try:
            cx, cy = parse_int_list(crop)
        except ValueError:
            raise click.ClickException("--crop must be 'auto', 'center' or X,Y")
        boxes = [(min(max(cx, 0), res - cs), min(max(cy, 0), res - cs), cs)] * len(jobs)
    zoom_scale = max(crop_zoom, int(np.ceil(128 / cs))) # zoom tiles at least 128 px
    rnd = np.random.RandomState(seed + 1)
    real_pick = [rnd.choice(real.shape[0], size=real_cols, replace=real.shape[0] < real_cols).tolist() for _ in jobs] if (real is not None and real_cols) else None

    def row_label(j):
        i, k = jobs[j]
        name = os.path.splitext(os.path.basename(items[i][0]))[0]
        lab = f'\n[{items[i][1]}]' if items[i][1] else ''
        var = f'\nvariant {k}' if variants > 1 else ''
        return f'{name}{lab}{var}'

    def save_pages(make_row, headers, title, stem):
        page_rows = [make_row(j) for j in range(len(jobs))]
        labels = [row_label(j) for j in range(len(jobs))]
        npages = (len(page_rows) - 1) // max_rows + 1
        written = []
        for p in range(npages):
            sl = slice(p * max_rows, (p + 1) * max_rows)
            name = f'{stem}.png' if npages == 1 else f'{stem}_p{p + 1}.png'
            t = title if npages == 1 else f'{title}  (page {p + 1}/{npages})'
            compose(page_rows[sl], headers, labels[sl], t).save(os.path.join(outdir, name))
            written.append(name)
        return written

    vs = 'x real' if ref is not None else 'absolute'
    written = []
    for f in factors:
        tag = 'SDEdit' if f <= 1 else f'ILVR N={f}'
        headers = [('input', _fmt_stats(st_in, ref))] + [(f'sigma {s:g}', _fmt_stats(stats[(f, s)], ref)) for s in sigmas]
        if real_pick:
            headers += [('real', _fmt_stats(ref, ref))] * real_cols
        def full_row(j, f=f):
            r = [outline(upscale(to_rgb(x_jobs[j]), scale), boxes[j], scale)]
            r += [upscale(to_rgb(outs[(f, s)][j]), scale) for s in sigmas]
            if real_pick:
                r += [upscale(to_rgb(real[q]), scale) for q in real_pick[j]]
            return r
        def zoom_row(j, f=f):
            r = [zoomed_crop(x_jobs[j], boxes[j], zoom_scale)] + [zoomed_crop(outs[(f, s)][j], boxes[j], zoom_scale) for s in sigmas]
            if real_pick:
                r += [zoomed_crop(real[q], auto_crop_box(real[q], cs), zoom_scale) for q in real_pick[j]]
            return r
        def diff_row(j, f=f):
            r = [upscale(to_rgb(x_jobs[j]), scale)]
            for s in sigmas:
                dmap = (outs[(f, s)][j].to(torch.float32) - x_jobs[j].to(torch.float32)).abs().mean(0, keepdim=True)
                dmap = (dmap * diff_gain).clamp(0, 255).to(torch.uint8)
                r.append(upscale(to_rgb(dmap), scale))
            return r
        dheaders = [('input', '')] + [(f'sigma {s:g}', f"drift {stats[(f, s)]['drift']:.2f}") for s in sigmas]
        written += save_pages(full_row, headers, f'{tag}: full images ({vs})', f'full_ilvr{f}')
        written += save_pages(zoom_row, headers, f'{tag}: zoom x{zoom_scale} on {cs}x{cs} px ({vs})', f'zoom_ilvr{f}')
        written += save_pages(diff_row, dheaders, f'{tag}: |output - input| x {diff_gain:g}  (drift = low-pass change, 0-255)', f'diff_ilvr{f}')

    # sigma x ILVR grid for a few images.
    if len(factors) > 1:
        for j in range(min(grid2d_images, len(items))):
            jj = jobs.index((j, 0))
            heads = [('input', '')] + [(f'sigma {s:g}', '') for s in sigmas]
            labels = [('SDEdit' if f <= 1 else f'ILVR N={f}') for f in factors]
            full = [[upscale(to_rgb(x_jobs[jj]), scale)] + [upscale(to_rgb(outs[(f, s)][jj]), scale) for s in sigmas] for f in factors]
            zoom = [[zoomed_crop(x_jobs[jj], boxes[jj], zoom_scale)] + [zoomed_crop(outs[(f, s)][jj], boxes[jj], zoom_scale) for s in sigmas] for f in factors]
            name = os.path.basename(items[j][0])
            compose(full, heads, labels, f'{name}: sigma_start x ILVR factor').save(os.path.join(outdir, f'grid2d_img{j}.png'))
            compose(zoom, heads, labels, f'{name}: sigma_start x ILVR factor (zoom x{zoom_scale})').save(os.path.join(outdir, f'grid2d_img{j}_zoom.png'))
            written += [f'grid2d_img{j}.png', f'grid2d_img{j}_zoom.png']

    # Metric curves.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [('grain', 'grain (noise std)'), ('sharpness', 'sharpness (gradient)'), ('hf_energy', 'HF energy (f > 0.25)'), ('drift', f'scene drift (low-pass /{drift_factor}, 0-255)')]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    for ax, (k, title) in zip(axes, keys):
        norm = (lambda v, k=k: v / max(ref[k], 1e-12)) if (ref is not None and k != 'drift') else (lambda v: v)
        for f in factors:
            ax.plot(sigmas, [norm(stats[(f, s)][k]) for s in sigmas], 'o-', label=('SDEdit' if f <= 1 else f'ILVR N={f}'))
        if k != 'drift':
            ax.axhline(norm(st_in[k]), color='tab:gray', ls=':', label='input (sim)')
            if ref is not None:
                ax.axhline(1.0, color='k', ls='--', label='real')
        ax.set_xscale('log'); ax.set_xlabel('sigma_start')
        ax.set_ylabel(title + (' / real' if (ref is not None and k != 'drift') else ''))
        if k == 'hf_energy' and ref is None:
            ax.set_yscale('log')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('Pick sigma_start where grain / sharpness / HF reach the real level while scene drift stays small')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'metrics.png'), dpi=110)
    written.append('metrics.png')

    with open(os.path.join(outdir, 'settings.json'), 'wt') as fh:
        json.dump(dict(network=network_pkl, inputs=[rel for rel, _ in items], conditional=conditional, guidance=guidance,
                       sigma_start=sigmas, ilvr_factors=factors, variants=variants, steps=num_steps, seed=seed,
                       real=real_path, crop_boxes=boxes, crop_zoom=zoom_scale, drift_lowpass_factor=drift_factor), fh, indent=2)

    # Console summary.
    print(f'\n{"setting":>14s} {"sigma":>7s} {"grain":>8s} {"sharp":>8s} {"HF":>9s} {"drift":>7s}   ({vs})')
    def fmt(st, k):
        return st[k] / max(ref[k], 1e-12) if ref is not None else st[k]
    print(f'{"input":>14s} {"":>7s} {fmt(st_in, "grain"):8.3f} {fmt(st_in, "sharpness"):8.3f} {fmt(st_in, "hf_energy"):9.3g} {0:7.2f}')
    for (f, s), st in stats.items():
        name = 'SDEdit' if f <= 1 else f'ILVR N={f}'
        print(f'{name:>14s} {s:7g} {fmt(st, "grain"):8.3f} {fmt(st, "sharpness"):8.3f} {fmt(st, "hf_energy"):9.3g} {st["drift"]:7.2f}')
    print(f'\nWrote {len(written)} file(s) to {outdir}: ' + ', '.join(written[:6]) + (' ...' if len(written) > 6 else ''))

if __name__ == '__main__':
    main()
