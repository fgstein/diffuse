# Smoke / overfit test results (CPU only, no GPU available)

Run on a 4-core CPU sandbox (no GPU, so no sec/kimg measurement for the 64 px preset).

Data: `tools/make_synthetic_dataset.py --n-per-class 30 --classes 3` (disk / square / stripes, 224x224),
prepared to 32 px with `tools/prepare_dataset.py` (87 train + 3 val images).

Model: SongUNet, `--model-channels 32 --channel-mult 1,2,2 --num-blocks 2 --attn-res 8`,
batch 32, lr 5e-4, EMA 10 kimg, fp32. Stopped with Ctrl-C (graceful stop) at 50.7 kimg
(~20 min, ~13 sec/kimg on CPU). This is far smaller/shorter than the 9.3 spec (~20M params, 0.5-1 Mimg),
which was not feasible on CPU.

| File | Content |
|---|---|
| `reals.png` | real images, one row per class |
| `training-grid-40kimg-g1-2-4.png` | training-time grid at 40 kimg, guidance 1 / 2 / 4 (left to right) |
| `generate-50kimg-g1.png`, `generate-50kimg-g2.png` | `generate.py --per-class 8 --grid 1` at guidance 1 and 2 |
| `stats.png` | `tools/plot_stats.py` output |
| `log.txt` | training log |

Result: every row shows the correct class shape (disks, squares, vertical and horizontal stripes)
at all guidance values. Sample colours are less diverse and backgrounds paler than the reals,
as expected after 50 kimg on 87 images. Validation loss plateaus at ~0.49 while train loss
keeps falling (0.08): expected overfitting on a tiny training set.
