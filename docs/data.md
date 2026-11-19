# Data

## Input format

The input is a **CSV file** that lists images by **relative path**, plus an **image root directory**:

```csv
path,label
sub/dir/img_0001.png,cat_a
sub/dir/img_0002.png,cat_b
```

| Aspect | Behaviour |
|---|---|
| Columns | `--path-col` (default `path`), `--label-col` (default `label`). Other columns are ignored. |
| Paths | Relative to `--image-root`. `\` is normalised to `/`. Every listed file is checked for existence at start-up, and missing files produce a clear error with examples. |
| Images | Must be **square**. The sources are expected to be 224×224. Grayscale, palette and RGBA images are converted to RGB (transparency is composited on black). `--channels 1` trains on grayscale. |
| Labels | Arbitrary strings, mapped to `0..K-1` in **sorted order**, so the mapping is deterministic. |
| Split | `--split-col` (values `train` / `val`; `valid` and `validation` are also accepted, other values are ignored with a warning). Without it, a **stratified, seeded** split of `--val-fraction` (default 2%, at least 1 image per class with ≥ 2 images) is used. |

### Label map

The label map is built from **all** rows (train + val), so both splits agree. It is saved:

- as `labels.json` in the run directory:
  ```json
  {"label_to_idx": {"cat_a": 0, "cat_b": 1}, "idx_to_label": ["cat_a", "cat_b"],
   "label_col": "label", "counts": {"cat_a": 5120, "cat_b": 812}}
  ```
- inside **every network snapshot** (key `label_map`), so a snapshot is self-contained for sampling;
- inside every training state, so `--resume` reuses it. If the CSV contains a label that the saved map does not know, training **stops with an error** instead of silently re-indexing.

`counts` are counts in the training split. At start-up the counts are printed, with a warning if the max/min ratio is above 10. See [class-balanced sampling](training.md#class-imbalance).

## Preparing the data

Training can read the raw CSV and resize on the fly, but **preparing once is recommended**: it validates the files and makes data loading much cheaper.

```bash
python tools/prepare_dataset.py --csv /data/train.csv --image-root /data/images \
    --path-col path --label-col label --resolution 64 --out /data/prepared-64.zip
```

What it does:

1. **Validates every row**: the file exists, decodes, and has the expected size (`--expected-size 224`; `0` disables the check). Bad rows are written with an `error` column to `<out>-bad_rows.csv` (zip output) or `<out>/bad_rows.csv` (directory output), and dropped.
2. **Resizes** to `--resolution` with an anti-aliased filter: PIL `BOX` for integer downscale factors (224→112, 224→32) and `LANCZOS` otherwise (224→64).
3. **Writes** either
   - a `.zip` of uint8 PNGs + `dataset.json` (when `--out` ends in `.zip`), or
   - a directory of PNGs + `data.csv` (same column names) + `dataset.json`.
4. Prints the good and bad counts and the class counts.

Add `--split-col split` to carry your own split into the prepared data. `--workers` (default 8) sets the number of decoding threads.

The prepared data is used with `--data`:

```bash
python train.py ... --data /data/prepared-64.zip
```

> On-the-fly loading and prepared data give **bit-identical tensors**, because both use the same `load_image()` function and PNG is lossless. This is covered by a test.

## On-the-fly resizing

```bash
python train.py ... --csv /data/train.csv --image-root /data/images --resolution 64
```

The images are decoded at 224 and resized in the DataLoader workers. This is fine with enough `--workers` and `--cache 1`. Without `--resolution`, the native size of the files is used.

## RAM cache

`--cache 1` (the default) keeps the decoded uint8 images in RAM after the first epoch.

| Resolution | Per image | 100k images |
|---|---|---|
| 64×64 | ~12 KB | ~1.2 GB |
| 224×224 | ~150 KB | ~15 GB |

Each DataLoader worker keeps its **own** cache, so multiply by `--workers` (default 4). The estimate is printed at start-up. At 224 px with a large dataset, use `--cache 0` and a prepared zip.

## Augmentation and flips

- `--xflip` is **off** by default. Plain horizontal flips change the image but keep the label, which is wrong if left/right matters in your data.
- `--augment p` enables EDM's **non-leaking** geometric augmentation (flips, integer translation, scaling, rotation, anisotropy). The network is told which augmentation was applied, and sampling asks for "no augmentation", so the augmentations do not leak into generated images. This is the **main lever for small datasets (< ~20k images)**. EDM used `0.12` on CIFAR-10.

## Synthetic test data

`tools/make_synthetic_dataset.py` writes a small dataset in exactly this format: 224×224 PNGs in nested folders, with up to 8 visually distinct classes (`disk`, `square`, `stripes`, `triangle`, `ring`, `cross`, `checker`, `gradient`).

```bash
python tools/make_synthetic_dataset.py --out /tmp/synth --n-per-class 30 --classes 3 [--path-col p --label-col y]
```
