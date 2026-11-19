# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Memorization check: pixel-space nearest neighbours of generated samples.

For every generated image (a directory written by generate.py), find the
nearest training image by L2 distance in pixel space (both resized to
--resolution, default 64), save side-by-side pairs (generated | nearest real)
for the closest matches, and report the fraction of samples whose RMS distance
is below --threshold (in [0, 1] pixel units).

No pretrained features are used. CPU-friendly for moderate sizes (chunked
matrix products); use --max-train to subsample very large training sets.

Example:
    python tools/memorization_check.py --gen out/ --data /data/prepared-64.zip --outdir memcheck
"""

import os
import sys
import json
import glob
import click
import numpy as np
import torch
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset import CSVImageDataset, load_image # pylint: disable=wrong-import-position

@click.command()
@click.option('--gen', 'gen_dir', help='Directory of generated PNGs (searched recursively)', required=True)
@click.option('--data', help='Prepared training dataset (zip/dir)', default=None)
@click.option('--csv', help='Training CSV (alternative to --data)', default=None)
@click.option('--image-root', help='Image root for --csv', default=None)
@click.option('--path-col', default='path', show_default=True)
@click.option('--label-col', default='label', show_default=True)
@click.option('--resolution', help='Comparison resolution', type=int, default=64, show_default=True)
@click.option('--threshold', help='RMS distance threshold in [0,1] units', type=float, default=0.05, show_default=True)
@click.option('--max-train', help='Subsample the training set to at most N images (0 = all)', type=int, default=0, show_default=True)
@click.option('--num-pairs', help='Number of closest pairs to save', type=int, default=32, show_default=True)
@click.option('--outdir', help='Output directory', required=True)
def main(gen_dir, data, csv, image_root, path_col, label_col, resolution, threshold, max_train, num_pairs, outdir):
    """Find pixel-space nearest training neighbours of generated samples."""
    os.makedirs(outdir, exist_ok=True)
    ds = CSVImageDataset(data=data, csv=csv, image_root=image_root, path_col=path_col, label_col=label_col,
                         split='all', resolution=resolution, max_size=(max_train or None))
    print(f'Loading {len(ds)} training images...')
    train = torch.stack([torch.as_tensor(ds[i][0]) for i in range(len(ds))]).float().div(255).flatten(1)
    files = sorted(glob.glob(os.path.join(gen_dir, '**', '*.png'), recursive=True))
    files = [f for f in files if os.path.basename(f) != 'grid.png']
    if not files:
        raise click.ClickException(f'No PNGs found in {gen_dir}')
    gen = []
    for f in files:
        with open(f, 'rb') as fh:
            gen.append(torch.as_tensor(load_image(fh, resolution=resolution, channels=ds.num_channels)))
    gen = torch.stack(gen).float().div(255).flatten(1)

    D = gen.shape[1]
    t2 = train.square().sum(1)
    best_d, best_i = [], []
    for chunk in gen.split(256):
        d2 = chunk.square().sum(1, keepdim=True) - 2 * chunk @ train.T + t2[None]
        d, i = d2.clamp_min(0).min(dim=1)
        best_d.append((d / D).sqrt()); best_i.append(i)
    rms = torch.cat(best_d).numpy()
    idx = torch.cat(best_i).numpy()

    order = np.argsort(rms)
    C, H = ds.num_channels, resolution
    for rank, g in enumerate(order[:num_pairs]):
        a = (gen[g].view(C, H, H).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        b = (train[idx[g]].view(C, H, H).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        pair = np.concatenate([a, np.full([H, 4, C], 255, np.uint8), b], axis=1)
        PIL.Image.fromarray(pair[:, :, 0] if C == 1 else pair).save(os.path.join(outdir, f'pair-{rank:03d}-rms{rms[g]:.4f}.png'))
    frac = float((rms < threshold).mean())
    summary = dict(num_generated=len(files), num_train=len(ds), resolution=resolution, threshold=threshold,
                   fraction_below_threshold=frac, rms_min=float(rms.min()), rms_median=float(np.median(rms)))
    with open(os.path.join(outdir, 'summary.json'), 'wt') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(outdir, 'nearest.csv'), 'wt') as f:
        f.write('file,nearest_train_path,rms\n')
        for g in range(len(files)):
            f.write(f'{os.path.relpath(files[g], gen_dir)},{ds.paths[idx[g]]},{rms[g]:.6f}\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
