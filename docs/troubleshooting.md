# Troubleshooting

## Installation

**`ModuleNotFoundError: No module named 'torch'` after `uv sync`**
uv removed torch because `uv sync` was run without the extra. Run `uv sync --extra cu126` (or `--extra cpu`) again.

**`torch.cuda.is_available()` is False on the GPU machine**
Check that you installed the `cu126` extra, not `cpu`: `uv run python -c "import torch; print(torch.__version__)"` should end in `+cu126`. Check the driver with `nvidia-smi`; the cu126 wheels need driver ≥ 560.

## Data

**`N image file(s) listed in the CSV were not found under ...`**
Paths must be relative to `--image-root`. Check one example by hand: `ls "<image-root>/<path>"`. Run `tools/prepare_dataset.py` to get a full `bad_rows.csv`.

**`CSV ... is missing column(s) ['path']`**
Pass `--path-col` / `--label-col` with your column names.

**`Image "..." is not square`**
Only square images are supported. Crop or pad them before training.

**`The data contains N label(s) that are not in the saved label map`** (on `--resume`)
The CSV gained labels after the run started. Resuming would change the meaning of the class indices. Start a new run, possibly with `--transfer` from the latest snapshot (the class embedding then needs to be retrained for the new classes). Alternatively, remove those rows.

**Validation split is empty**
Every class has fewer than 2 images, or `--val-fraction 0`. Validation is then skipped with a warning.

## Memory

**CUDA out of memory**
Work through these in order:

1. halve `--batch-gpu`; the effective `--batch` stays the same;
2. `--grad-ckpt 1`;
3. at 224 px: `--patch-size 2` (or 4), fewer attention resolutions, smaller `--model-channels`;
4. if it fails during sample grids: lower `--sample-cols` / `--sample-max-classes`, or pass `--sample-every 0` and sample offline.

**Host RAM keeps growing**
That is the per-worker image cache filling up during the first epoch. The total is printed at start-up. Use fewer `--workers`, `--cache 0`, or a prepared zip at the target resolution.

## Speed

### Training is slow / GPU utilisation is low

- Use a **prepared** dataset at the target resolution. Resizing 224→64 with LANCZOS on the fly costs CPU time.
- Increase `--workers` (4–8), keep `--cache 1` if RAM allows.
- Check `--tf32 1` and `--precision bf16`.
- `--grad-ckpt 1` costs ~30%. Use it only when memory requires it.
- Try `--compile 1` (the first ticks are slower while compiling).
- Validation and grids run inside ticks (see `maintenance` in the status line). Make `--sample-every` / `--val-every` less frequent if they dominate.

## Training dynamics

**Loss is NaN or explodes**
Use `--precision bf16` (not fp16), lower `--lr`, add `--grad-clip 1`. NaN/Inf gradients are zeroed automatically, but a NaN in the weights is not recoverable: resume from an earlier state.

**`--batch (B) must be divisible by --batch-gpu (b) x number of GPUs`**
Choose `--batch-gpu` as a divisor of `--batch`.

**Validation loss goes up while training loss goes down**
Overfitting. Try `--augment 0.12`, more `--dropout`, a smaller model, or stop earlier. Check `tools/memorization_check.py`.

## Samples

**Guidance has no effect / `--guidance 0` looks like the conditional model**
The model was trained with `--label-dropout 0`, so the NULL token is untrained. Retrain with `--label-dropout 0.1`.

**Rows don't match their class at guidance 1 but do at 2–4**
Conditioning is still weak. This is normal early in training. Watch it improve over the grids.

**Over-saturated, repetitive samples**
Guidance is too high. Lower it, or use `--guidance-interval`.

**Samples look like noise or are blurry early on**
Early EMA weights lag behind the network. Wait, or compare with a shorter EMA (`--ema-list`).

**224 px samples have wrong global structure**
Try the noise shift (`--noise-shift-base 64`) and check that `generate.py` uses the snapshot's `sigma_max` (printed at start-up).

## Resume

**`Missing tensor in source module: ...`**
The architecture options differ from the original run. Resume with the same options (copy them from `training_options.json`).

**Stats look duplicated around the resume point**
Ticks after the last saved state were lost with the interruption and are written again. When plotting, keep the last occurrence of each kimg.
