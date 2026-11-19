# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Plot training diagnostics from <run_dir>/stats.jsonl.

Writes <run_dir>/stats.png with four panels:
  1. training loss and validation loss vs kimg,
  2. training loss per log-sigma bin vs kimg,
  3. validation loss per fixed sigma (latest and earliest evaluation),
  4. learning rate and sec/kimg.

Example:
    python tools/plot_stats.py runs/00000-xxx
"""

import os
import sys
import json
import click
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_stats(run_dir):
    rows = []
    with open(os.path.join(run_dir, 'stats.jsonl'), 'rt') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and 'Progress/kimg' in r:
            xs.append(r['Progress/kimg']['mean'])
            ys.append(r[key]['mean'])
    return np.array(xs), np.array(ys)

@click.command()
@click.argument('run_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--out', help='Output PNG [default: <run_dir>/stats.png]', type=str, default=None)
def main(run_dir, out):
    """Plot loss curves (overall, per sigma bin, validation per sigma) from stats.jsonl."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from training.loss import SIGMA_BIN_EDGES
    from training.training_loop import VAL_SIGMAS

    rows = load_stats(run_dir)
    if not rows:
        raise click.ClickException('stats.jsonl is empty')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    x, y = series(rows, 'Loss/loss')
    ax.plot(x, y, label='train loss (tick mean)')
    x, y = series(rows, 'Val/loss')
    if len(x):
        ax.plot(x, y, 'o-', label='val loss (EMA, fixed sigmas)')
    ax.set_xlabel('kimg'); ax.set_ylabel('EDM-weighted loss'); ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    cmap = plt.get_cmap('viridis')
    nb = len(SIGMA_BIN_EDGES) - 1
    for k in range(nb):
        x, y = series(rows, f'Loss/sigma_bin_{k}')
        if len(x):
            ax.plot(x, y, color=cmap(k / max(nb - 1, 1)), label=f'sigma {SIGMA_BIN_EDGES[k]:.3g}-{SIGMA_BIN_EDGES[k+1]:.3g}')
    ax.set_xlabel('kimg'); ax.set_ylabel('train loss per sigma bin'); ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    val_rows = [r for r in rows if 'Val/loss_sigma_00' in r]
    if val_rows:
        picks = [val_rows[0], val_rows[len(val_rows) // 2], val_rows[-1]] if len(val_rows) > 2 else val_rows
        for r in picks:
            v = [r.get(f'Val/loss_sigma_{i:02d}', {}).get('mean', np.nan) for i in range(len(VAL_SIGMAS))]
            ax.plot(VAL_SIGMAS, v, 'o-', label=f'kimg {r["Progress/kimg"]["mean"]:.0f}')
        ax.legend()
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('sigma'); ax.set_ylabel('val loss'); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    x, y = series(rows, 'Progress/lr')
    if len(x):
        ax.plot(x, y, color='tab:blue'); ax.set_ylabel('lr', color='tab:blue')
    ax2 = ax.twinx()
    x, y = series(rows, 'Timing/sec_per_kimg')
    ax2.plot(x, y, color='tab:red', alpha=0.6); ax2.set_ylabel('sec/kimg', color='tab:red')
    ax.set_xlabel('kimg'); ax.grid(alpha=0.3)

    fig.suptitle(os.path.basename(os.path.normpath(run_dir)))
    fig.tight_layout()
    out = out or os.path.join(run_dir, 'stats.png')
    fig.savefig(out, dpi=110)
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
