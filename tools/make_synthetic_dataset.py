# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Generate a tiny synthetic dataset in the exact user format (for testing).

Writes 224x224 RGB PNGs in nested sub-folders (to exercise relative paths) and
a CSV with relative paths and string labels.

Example:
    python tools/make_synthetic_dataset.py --out /tmp/synth --n-per-class 30 --classes 3
"""

import os
import sys
import click
import numpy as np
import PIL.Image
import PIL.ImageDraw

CLASS_NAMES = ['disk', 'square', 'stripes', 'triangle', 'ring', 'cross', 'checker', 'gradient']

def _rand_color(rnd, lo=30, hi=230):
    return tuple(int(c) for c in rnd.randint(lo, hi, size=3))

def draw_sample(name, rnd, size=224):
    bg = tuple(int(c) for c in rnd.randint(200, 256, size=3))
    img = PIL.Image.new('RGB', (size, size), bg)
    d = PIL.ImageDraw.Draw(img)
    fg = _rand_color(rnd, 0, 160)
    r = rnd.randint(size // 6, size // 3)
    cx, cy = rnd.randint(r, size - r, size=2)
    if name == 'disk':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif name == 'square':
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif name == 'stripes':
        period = rnd.randint(16, 40)
        w = period // 2
        vertical = rnd.rand() < 0.5
        off = rnd.randint(period)
        for k in range(-period + off, size + period, period):
            box = [k, 0, k + w, size] if vertical else [0, k, size, k + w]
            d.rectangle(box, fill=fg)
    elif name == 'triangle':
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
    elif name == 'ring':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=max(4, r // 4))
    elif name == 'cross':
        t = max(6, r // 3)
        d.rectangle([cx - r, cy - t, cx + r, cy + t], fill=fg)
        d.rectangle([cx - t, cy - r, cx + t, cy + r], fill=fg)
    elif name == 'checker':
        cell = rnd.randint(20, 45)
        for y in range(0, size, cell):
            for x in range(0, size, cell):
                if ((x // cell) + (y // cell)) % 2 == 0:
                    d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=fg)
    elif name == 'gradient':
        a = np.linspace(0, 1, size)[None, :, None]
        c0, c1 = np.array(_rand_color(rnd)), np.array(_rand_color(rnd))
        arr = np.broadcast_to(c0 * (1 - a) + c1 * a, (size, size, 3))
        img = PIL.Image.fromarray(arr.astype(np.uint8))
    else:
        raise ValueError(name)
    arr = np.asarray(img).astype(np.float32) + rnd.randn(size, size, 3) * 6.0 # light noise
    return PIL.Image.fromarray(arr.clip(0, 255).astype(np.uint8))

@click.command()
@click.option('--out', help='Output directory', required=True, type=str)
@click.option('--n-per-class', help='Images per class', default=30, show_default=True, type=click.IntRange(min=1))
@click.option('--classes', help=f'Number of classes (max {len(CLASS_NAMES)})', default=3, show_default=True, type=click.IntRange(min=1, max=len(CLASS_NAMES)))
@click.option('--size', help='Image size', default=224, show_default=True, type=int)
@click.option('--path-col', help='Name of the path column', default='path', show_default=True)
@click.option('--label-col', help='Name of the label column', default='label', show_default=True)
@click.option('--seed', help='Random seed', default=0, show_default=True, type=int)
def main(out, n_per_class, classes, size, path_col, label_col, seed):
    """Generate a small synthetic class-conditional dataset (PNGs + data.csv)."""
    import pandas as pd
    rnd = np.random.RandomState(seed)
    rows = []
    for ci, name in enumerate(CLASS_NAMES[:classes]):
        for i in range(n_per_class):
            rel = f'images/{name}/part{i % 3}/img_{ci:02d}_{i:05d}.png'
            os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
            draw_sample(name, rnd, size).save(os.path.join(out, rel))
            rows.append({path_col: rel, label_col: name})
    order = rnd.permutation(len(rows)) # shuffle rows so classes are interleaved
    pd.DataFrame([rows[i] for i in order]).to_csv(os.path.join(out, 'data.csv'), index=False)
    print(f'Wrote {len(rows)} images ({classes} classes) and {os.path.join(out, "data.csv")}')

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
