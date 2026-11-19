# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modified: class names from the snapshot label map, classifier-free guidance
# (with optional guidance interval), per-class generation, grids, metadata.csv,
# CPU support, runs without torchrun.

"""Generate images with a trained class-conditional EDM model."""

import os
import re
import csv
import click
import tqdm
import pickle
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from training.sampling import StackedRandomGenerator, edm_sampler, to_uint8, make_grid, save_image

#----------------------------------------------------------------------------

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def parse_interval(s):
    if s is None: return None
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 2 or parts[0] > parts[1]:
        raise click.BadParameter('expected SMIN,SMAX with SMIN <= SMAX')
    return tuple(parts)

def safe_dirname(label):
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label)).strip('._')
    return s or 'label'

#----------------------------------------------------------------------------

def load_network(network_pkl, device, ema_halflife=None):
    """Load a snapshot. Returns (net, label_map_json or None, noise_config dict)."""
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    net = data['ema']
    if ema_halflife is not None:
        emas = data.get('emas', {})
        key = str(float(ema_halflife * 1000))
        if key not in emas:
            raise click.ClickException(f'No EMA with half-life {ema_halflife} Mimg in snapshot; available (kimg): {sorted(emas)}')
        net = emas[key]
    net = net.eval().requires_grad_(False).to(device)
    return net, data.get('label_map'), dict(data.get('noise_config') or {})

def build_jobs(seeds, seeds_given, idx_to_label, counts, classes, class_idx, per_class, class_dist):
    """Return a list of (seed, class_idx or None)."""
    K = len(idx_to_label)
    if K == 0:
        if classes or class_idx is not None or per_class:
            raise click.ClickException('This network is unconditional; --class/--class-idx/--per-class are not applicable.')
        return [(s, None) for s in seeds]
    selected = None
    if classes:
        label_to_idx = {name: i for i, name in enumerate(idx_to_label)}
        selected = []
        for name in classes.split(','):
            if name not in label_to_idx:
                raise click.ClickException(f'Unknown class "{name}". Known classes: {idx_to_label}')
            selected.append(label_to_idx[name])
    if class_idx is not None:
        if class_idx >= K:
            raise click.ClickException(f'--class-idx {class_idx} out of range (0..{K - 1})')
        selected = (selected or []) + [class_idx]
    if per_class:
        if seeds_given:
            if len(seeds) < per_class:
                raise click.ClickException(f'--per-class {per_class} needs at least that many --seeds')
            seeds = seeds[:per_class]
        else:
            seeds = list(range(per_class))
        return [(s, c) for c in (selected if selected is not None else range(K)) for s in seeds]
    if selected is not None:
        return [(s, c) for c in selected for s in seeds]
    # Random classes, deterministic per seed.
    if class_dist == 'data' and counts:
        p = np.array([counts.get(name, 0) for name in idx_to_label], dtype=np.float64)
        p = p / p.sum() if p.sum() > 0 else None
    else:
        p = None
    return [(s, int(np.random.RandomState(s % (1 << 31)).choice(K, p=p))) for s in seeds]

#----------------------------------------------------------------------------

@click.command(context_settings=dict(max_content_width=120))
@click.option('--network', 'network_pkl',  help='Network snapshot (network-snapshot-*.pkl)', metavar='PATH',        type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default=None)
@click.option('--class', 'classes',        help='Class name(s), comma-separated', metavar='NAME[,NAME...]',          type=str, default=None)
@click.option('--class-idx',               help='Class index', metavar='INT',                                       type=click.IntRange(min=0), default=None)
@click.option('--per-class',               help='Generate N images for every (selected) class', metavar='N',        type=click.IntRange(min=1), default=None)
@click.option('--class-dist',              help='Random class distribution', metavar='uniform|data',                type=click.Choice(['uniform', 'data']), default='uniform', show_default=True)
@click.option('--guidance',                help='CFG weight (1 = no guidance, 0 = unconditional)', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--guidance-interval',       help='Apply CFG only for sigma in [SMIN,SMAX]', metavar='SMIN,SMAX',     type=str, default=None)
@click.option('--ema-halflife',            help='Pick a secondary EMA by half-life', metavar='MIMG',                type=float, default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--grid',                    help='Also write grid.png (rows = classes)', metavar='BOOL',             type=bool, default=False, show_default=True)
@click.option('--device',                  help='Device [default: cuda if available]', metavar='cuda|cpu',          type=click.Choice(['cuda', 'cpu']), default=None)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=2), default=32, show_default=True)
@click.option('--sigma-min', '--sigma_min', 'sigma_min', help='Lowest noise level', metavar='FLOAT',               type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma-max', '--sigma_max', 'sigma_max', help='Highest noise level  [default: snapshot hint or 80]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S-churn', '--S_churn', 'S_churn', help='Stochasticity strength', metavar='FLOAT',                 type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S-min', '--S_min', 'S_min', help='Stoch. min noise level', metavar='FLOAT',                       type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S-max', '--S_max', 'S_max', help='Stoch. max noise level', metavar='FLOAT',                       type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S-noise', '--S_noise', 'S_noise', help='Stoch. noise inflation', metavar='FLOAT',                 type=float, default=1, show_default=True)

def main(network_pkl, outdir, seeds, classes, class_idx, per_class, class_dist, guidance, guidance_interval,
         ema_halflife, max_batch_size, grid, device, **sampler_kwargs):
    """Generate images with a trained class-conditional model.

    Examples:

    \b
    # 8 images per class with CFG weight 2, plus a grid
    python generate.py --network runs/00000-xxx/network-snapshot-001000.pkl \\
        --outdir out --per-class 8 --guidance 2 --grid 1

    \b
    # 64 images of two named classes
    python generate.py --network snap.pkl --outdir out --class cat_a,cat_b --seeds 0-31
    """
    device_type = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device_type == 'cuda' and not torch.cuda.is_available():
        raise click.ClickException('--device cuda requested but CUDA is not available.')
    dist.init(device_type)
    device = torch.device('cuda', torch.cuda.current_device()) if device_type == 'cuda' else torch.device('cpu')
    guidance_interval = parse_interval(guidance_interval)

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    net, label_json, noise_config = load_network(network_pkl, device, ema_halflife)
    idx_to_label = list(label_json['idx_to_label']) if (label_json and 'class' in net.cond_stack.names) else []
    counts = (label_json or {}).get('counts', {})
    if sampler_kwargs['sigma_max'] is None:
        sampler_kwargs['sigma_max'] = float(noise_config.get('sigma_max', 80.0))
    dist.print0(f'Classes: {len(idx_to_label)}  resolution: {net.img_resolution}  sigma_max: {sampler_kwargs["sigma_max"]:.1f}  guidance: {guidance}')

    # Jobs.
    seeds_given = seeds is not None
    if seeds is None:
        seeds = list(range(64))
    jobs = build_jobs(seeds, seeds_given, idx_to_label, counts, classes, class_idx, per_class, class_dist)
    num_batches = ((len(jobs) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.arange(len(jobs)), num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Generate.
    dist.print0(f'Generating {len(jobs)} images to "{outdir}"...')
    records = []
    grid_images = {}
    for batch_idx in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        if len(batch_idx) == 0:
            continue
        batch_jobs = [jobs[i] for i in batch_idx]
        batch_seeds = [s for s, _ in batch_jobs]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_jobs), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        cond = None
        if idx_to_label:
            cond = {'class': torch.as_tensor([c for _, c in batch_jobs], dtype=torch.int64, device=device)}
        with torch.no_grad():
            images = edm_sampler(net, latents, cond, randn_like=rnd.randn_like, guidance=guidance,
                                 guidance_interval=guidance_interval, **sampler_kwargs)
        images = to_uint8(images).cpu()
        for (seed, c), img in zip(batch_jobs, images):
            label = idx_to_label[c] if c is not None else ''
            sub = safe_dirname(label) if c is not None else 'uncond'
            os.makedirs(os.path.join(outdir, sub), exist_ok=True)
            rel = f'{sub}/{seed:06d}.png'
            save_image(img.permute(1, 2, 0).numpy(), os.path.join(outdir, rel))
            records.append(dict(file=rel, label=label, label_idx=('' if c is None else c), seed=seed, guidance=guidance))
            if grid:
                grid_images.setdefault(c, []).append(img)

    # Metadata (one file per rank when running multi-GPU).
    dist.barrier()
    suffix = '' if dist.get_world_size() == 1 else f'-rank{dist.get_rank()}'
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f'metadata{suffix}.csv'), 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'label', 'label_idx', 'seed', 'guidance'])
        w.writeheader()
        w.writerows(records)

    # Grid: rows = classes, columns = images of that class (padded with white).
    if grid and grid_images:
        keys = sorted(grid_images, key=lambda k: -1 if k is None else k)
        cols = max(len(v) for v in grid_images.values())
        blank = torch.full_like(next(iter(grid_images.values()))[0], 255)
        tiles = [ (grid_images[k] + [blank] * (cols - len(grid_images[k]))) for k in keys ]
        tiles = torch.stack([t for row in tiles for t in row])
        save_image(make_grid(tiles, len(keys), cols), os.path.join(outdir, f'grid{suffix}.png'))
        dist.print0('Grid rows: ' + ', '.join('uncond' if k is None else idx_to_label[k] for k in keys))

    dist.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
