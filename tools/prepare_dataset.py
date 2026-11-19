# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Validate and pre-resize a CSV image dataset.

Reads CSV + image root, validates every file (exists, decodable, expected
size), writes the bad rows to `bad_rows.csv` (next to the output), resizes the
good images to the target resolution, and writes either

  * a .zip of uint8 PNGs + dataset.json        (if --out ends with .zip), or
  * a directory of PNGs + data.csv + dataset.json  (otherwise).

The training script accepts the result via `--data`. Resizing uses exactly the
same code as on-the-fly loading (training/dataset.py: load_image), so both
paths produce identical tensors.

Example:
    python tools/prepare_dataset.py --csv /data/train.csv --image-root /data/images \\
        --path-col path --label-col label --resolution 64 --out /data/prepared-64.zip
"""

import os
import sys
import io
import json
import zipfile
import click
import numpy as np
import PIL.Image
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset import load_image, PREPARED_FORMAT, build_label_map # pylint: disable=wrong-import-position

def _process(args):
    i, root, rel, resolution, channels, expected_size = args
    full = os.path.join(root, rel)
    if not os.path.isfile(full):
        return i, None, 'file not found'
    try:
        with open(full, 'rb') as f:
            with PIL.Image.open(f) as im:
                w, h = im.size
            if expected_size and (w, h) != (expected_size, expected_size):
                return i, None, f'unexpected size {w}x{h} (expected {expected_size}x{expected_size})'
            f.seek(0)
            arr = load_image(f, resolution=resolution, channels=channels, name=rel)
    except Exception as e: # pylint: disable=broad-except
        return i, None, f'decode error: {e}'
    buf = io.BytesIO()
    hwc = arr.transpose(1, 2, 0)
    PIL.Image.fromarray(hwc[:, :, 0] if channels == 1 else hwc, 'L' if channels == 1 else 'RGB').save(buf, format='png', compress_level=1)
    return i, buf.getvalue(), None

def _out_name(rel):
    return os.path.splitext(rel.replace('\\', '/'))[0] + '.png'

@click.command()
@click.option('--csv', 'csv_path', help='Input CSV', required=True, type=str)
@click.option('--image-root', help='Root directory of the relative image paths', required=True, type=str)
@click.option('--path-col', help='CSV column with relative paths', default='path', show_default=True)
@click.option('--label-col', help='CSV column with string labels', default='label', show_default=True)
@click.option('--split-col', help='Optional CSV column with train/val split values', default=None, type=str)
@click.option('--resolution', help='Target resolution', required=True, type=click.IntRange(min=1))
@click.option('--channels', help='Number of channels (3 = RGB, 1 = grayscale)', default=3, show_default=True, type=click.Choice(['1', '3']))
@click.option('--expected-size', help='Expected source size (0 = do not check)', default=224, show_default=True, type=click.IntRange(min=0))
@click.option('--out', help='Output .zip or directory', required=True, type=str)
@click.option('--workers', help='Worker threads', default=8, show_default=True, type=click.IntRange(min=1))
def main(csv_path, image_root, path_col, label_col, split_col, resolution, channels, expected_size, out, workers):
    """Validate, resize and package a CSV image dataset for training."""
    import pandas as pd
    from tqdm import tqdm
    channels = int(channels)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for c in [path_col, label_col, split_col]:
        if c is not None and c not in df.columns:
            raise click.ClickException(f'Column "{c}" not in CSV; available: {list(df.columns)}')
    rels = df[path_col].tolist()

    jobs = [(i, image_root, rel, resolution, channels, expected_size) for i, rel in enumerate(rels)]
    is_zip = out.lower().endswith('.zip')
    out_dir = os.path.dirname(os.path.abspath(out)) if is_zip else os.path.abspath(out)
    os.makedirs(out_dir, exist_ok=True)
    zf = zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) if is_zip else None

    bad, good_rows, seen = [], [], set()
    with ThreadPoolExecutor(workers) as pool:
        for i, png, err in tqdm(pool.map(_process, jobs, chunksize=16), total=len(jobs), desc='Processing'):
            if err is None and not rels[i]:
                err = 'empty path'
            name = _out_name(rels[i])
            if err is None and name in seen:
                err = f'duplicate output name {name}'
            if err is not None:
                bad.append(dict(df.iloc[i].to_dict(), error=err))
                continue
            seen.add(name)
            if zf is not None:
                zf.writestr(name, png)
            else:
                os.makedirs(os.path.join(out_dir, os.path.dirname(name)), exist_ok=True)
                with open(os.path.join(out_dir, name), 'wb') as f:
                    f.write(png)
            row = dict(path=name, label=str(df.iloc[i][label_col]))
            if split_col is not None:
                row['split'] = str(df.iloc[i][split_col])
            good_rows.append(row)

    label_map = build_label_map([r['label'] for r in good_rows])
    meta = dict(
        format=PREPARED_FORMAT, resolution=resolution, channels=channels,
        path_col=path_col, label_col=label_col, split_col=split_col,
        source_csv=os.path.abspath(csv_path), source_root=os.path.abspath(image_root),
        rows=good_rows,
        labels=[[r['path'], label_map[r['label']]] for r in good_rows], # EDM-compatible field
    )
    if zf is not None:
        zf.writestr('dataset.json', json.dumps(meta))
        zf.close()
    else:
        with open(os.path.join(out_dir, 'dataset.json'), 'wt') as f:
            json.dump(meta, f)
        out_df = pd.DataFrame({path_col: [r['path'] for r in good_rows], label_col: [r['label'] for r in good_rows]})
        if split_col is not None:
            out_df[split_col] = [r['split'] for r in good_rows]
        out_df.to_csv(os.path.join(out_dir, 'data.csv'), index=False)

    bad_path = os.path.join(out_dir, (os.path.splitext(os.path.basename(out))[0] + '-' if is_zip else '') + 'bad_rows.csv')
    if bad:
        pd.DataFrame(bad).to_csv(bad_path, index=False)
    counts = {k: 0 for k in label_map}
    for r in good_rows:
        counts[r['label']] += 1
    print(f'Good images: {len(good_rows)}   bad rows: {len(bad)}' + (f'  (see {bad_path})' if bad else ''))
    print('Class counts: ' + json.dumps(counts))
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
