# edm-cond: class-conditional pixel-space diffusion, trained from scratch

A fork of NVIDIA's [EDM](https://github.com/NVlabs/edm) (Karras et al. 2022,
*Elucidating the Design Space of Diffusion-Based Generative Models*) changed to
train **class-conditional** diffusion models **in pixel space**, **from
scratch**, on a dataset described by a **CSV file**.

- No autoencoder or latent space, and **no pretrained component** anywhere in the training or sampling path.
- Configurable resolution: presets for **64x64** and native **224x224** (Simple-Diffusion-style patching, attention only at low resolution, noise shift).
- Data comes from a CSV (relative image paths and string labels) plus an image root directory, or from a pre-resized zip or directory.
- **Classifier-free guidance (CFG)** with a learned NULL token, plus an optional guidance interval.
- A **modular conditioning** system, so you can add more conditions later.
- Sized for a **single RTX 3090 (24 GB)** with bf16 autocast, TF32, optional gradient checkpointing and gradient accumulation. It also runs on **CPU**, slowly, for testing.

License: CC BY-NC-SA 4.0 (inherited from EDM, see `LICENSE.txt`).

---

## 1. Install

Packaging uses [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Choose **one** torch flavour with an extra:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is not installed yet

uv sync --extra cu126                                  # RTX 3090: CUDA 12.6 wheels (NVIDIA driver >= 560)
# uv sync --extra cpu                                  # CPU-only machine (tests, debugging)
# optional extras: --extra tensorboard  (for --tensorboard=1),  --extra fid  (scipy, for fid.py)

uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

`uv sync` creates `.venv/`. Either prefix commands with `uv run` (e.g. `uv run python train.py ...`) or run `source .venv/bin/activate` once. The commands below assume the venv is active.

The `cu126` extra resolves to torch 2.14.x. PyTorch publishes 2.15 only for CUDA 13.2+, which needs a very recent driver. To use a newer CUDA build, change the `pytorch-cu126` index URL in `pyproject.toml` and run `uv lock`. Pass the same `--extra` to every later `uv sync`, otherwise uv removes the torch flavour you chose. A plain `uv run` keeps what is installed.

Run the CPU test suite (about 1–2 min; `pytest` is in the default `dev` group):

```bash
uv run pytest -q
```

## 2. Data format, checks and preprocessing

The input is a CSV file plus an image root directory:

```
path,label
sub/dir/img_0001.png,cat_a
sub/dir/img_0002.png,cat_b
```

- Paths are relative to `--image-root`. Choose the column names with `--path-col` and `--label-col`.
- Labels are arbitrary strings. They are mapped to `0..K-1` in **sorted** order and saved as `labels.json` in the run directory and inside every snapshot.
- Images should be square (224x224). Grayscale and RGBA images are converted to RGB. `--channels 1` trains on grayscale.
- Validation split: give `--split-col` (values `train` / `val`), or let the code build a deterministic, stratified split of `--val-fraction` (default 0.02, seeded by `--seed`).

**Recommended:** validate the data and pre-resize it once. Every file is checked: it must exist, decode, and have the expected size. Bad rows are written to `*bad_rows.csv` and dropped.

```bash
python tools/prepare_dataset.py --csv /data/train.csv --image-root /data/images \
    --path-col path --label-col label --resolution 64 --out /data/prepared-64.zip
```

If `--out` is a directory instead of a `.zip`, the script writes PNGs plus `data.csv` (same column names) plus `dataset.json`. Add `--split-col split` to keep your own split.

Resizing uses PIL `BOX` for integer downscale factors and `LANCZOS` otherwise. Training can also resize on the fly from the raw CSV (pass `--csv`/`--image-root` instead of `--data`). The two paths give **bit-identical** tensors (this is tested).

RAM cache (`--cache`, on by default): decoded uint8 images stay in memory. The cost is about 12 KB per image at 64 px and about 150 KB at 224 px, times the number of DataLoader workers (each worker keeps its own cache). The estimate is printed at startup. Use `--cache 0` for large 224 px datasets.

## 3. Throughput probe: size `batch_gpu` and `duration`

The presets are **untuned starting points**. Before a multi-day run, measure throughput:

```bash
python train.py --config configs/64px_3090.yaml --data /data/prepared-64.zip --outdir runs-probe \
    --tick 5 --duration 0.015 --snap 1000 --dump 1000 --sample-every 0 --val-every 0
```

Each tick prints a line like:

```
tick 2  kimg 10.2  time 1m 05s  sec/tick 30.1  sec/kimg 5.90  ...  gpumem 17.3  reserved 18.0
```

- **`batch_gpu`**: use the largest value whose `reserved` memory stays below about 22 GB. `--batch` is the effective batch. Gradient accumulation (`batch / batch_gpu` rounds) keeps the optimization identical. If you run out of memory, halve `--batch-gpu`. At 224 px, keep `--grad-ckpt 1`.
- **`duration`** (in Mimg) = available hours × 3.6 / `sec/kimg`.
  For example, 72 h at 5.9 sec/kimg gives 72 × 3.6 / 5.9 ≈ **44 Mimg**.
- Ignore the very first tick: it includes start-up and cuDNN autotuning.

## 4. Full 64 px training

```bash
# From the prepared zip (recommended)
python train.py --config configs/64px_3090.yaml --data /data/prepared-64.zip --outdir runs

# From the raw CSV (on-the-fly 224 -> 64 resize)
python train.py --config configs/64px_3090.yaml --csv /data/train.csv --image-root /data/images \
    --path-col path --label-col label --resolution 64 --outdir runs
```

Explicit flags always override the YAML, for example `--duration 44 --batch-gpu 128`. `python train.py --help` lists every option. The fully resolved configuration is written to `runs/<run>/training_options.json`.

Useful knobs:

| Flag | Default | Notes |
|---|---|---|
| `--label-dropout` | 0.1 | Fraction of samples trained with the NULL label (needed for CFG). |
| `--class-balanced 1 --balance-alpha 0.5` | off | Weighted sampling ∝ 1/count^α for imbalanced data. A warning is printed if max/min class count > 10. |
| `--augment p` | 0 | EDM's non-leaking geometric augmentation (the network is conditioned on the augmentation, and sampling uses "none"). **This is the main lever if the dataset is small (< ~20k images)**, e.g. `--augment 0.12`. |
| `--xflip` | **off** | Plain dataset flips. Off because horizontal flips may not preserve the label for your data. |
| `--lr-decay invsqrt --lr-ref-kimg K` | none | EDM2-style lr / sqrt(max(kimg/K, 1)). |
| `--ema-list 0.05,0.5,2` | – | Several EMAs (half-lives in Mimg) in one run. The first is the primary one, used for grids and validation. Pick another at sampling time with `--ema-halflife`. |
| `--grad-clip`, `--weight-decay` | 0 | Off by default (EDM settings). |
| `--precision fp32/fp16/bf16`, `--tf32` | bf16, on | bf16 needs no loss scaling. |
| `--compile 1` | off | `torch.compile` of the training forward pass. Snapshots still pickle the uncompiled module. |
| `--keep-snapshots N --keep-every KIMG` | 10, 0 | Disk retention. |
| `--uncond 1` | off | Unconditional training, for debugging. |

## 5. Resume after an interruption

Press **Ctrl-C** once, or send SIGTERM. The run finishes the current step and writes `network-snapshot-*.pkl` and `training-state-*.pt` before it exits. Press Ctrl-C a second time to abort immediately.

To resume, run the **same command** plus `--resume`:

```bash
python train.py --config configs/64px_3090.yaml --data /data/prepared-64.zip --outdir runs \
    --resume runs/00000-prepared-64-64px-cond12-ddpmpp-gpus1-batch256-bf16/training-state-001234.pt
```

Training continues **in the same run directory** and appends to `log.txt` and `stats.jsonl`. The state file restores the network, the optimizer, all EMAs, the image counter (kimg), the tick, the RNG states and the position in the data stream. The data stream is a seeded sequence of epochs, and resume skips to exactly `cur_nimg`. The saved label map is reused. If the CSV now contains labels the map does not know, training stops with an error rather than silently re-indexing.

## 6. Monitoring (no pretrained metric needed)

The run directory `runs/<run>/` contains:

- `log.txt`: one status line per tick: `loss`, `sec/kimg`, `gpumem`, and `val` on validation ticks.
- `stats.jsonl`: all scalars per tick:
  - `Loss/loss`: training loss.
  - `Loss/sigma_bin_0..7`: training loss in 8 fixed log-σ bins spanning [0.002, 80].
  - `Val/loss` and `Val/loss_sigma_00..15`: validation denoising loss of the primary EMA at **16 fixed σ values with fixed noise**. It is comparable across ticks **and across runs**, and it is the primary model-selection metric.
  - `Progress/lr`, timing, memory.
- `reals.png`: real images, one row per class (up to `--sample-max-classes`, default 16).
- `samples-<kimg>.png`: generated images, one row per class and fixed seeds per column. There is one block per guidance value in `--sample-guidance` (default `1,2,4`, left to right), 32 steps, primary EMA. The seeds are fixed, so grids are comparable over time.
- `labels.json`, `training_options.json`, `network-snapshot-*.pkl`, `training-state-*.pt`.

```bash
python tools/plot_stats.py runs/<run>          # writes runs/<run>/stats.png
```

What to look at:

1. **Validation loss**, overall and per σ. It should decrease and then plateau. If it rises while the training loss keeps falling, the model is overfitting. Turn on `--augment`, reduce the model size, or stop earlier.
2. **Per-σ curves**: the low-σ bins (fine detail) usually improve last. A flat high-σ bin is normal: global layout is hard to predict at very high noise.
3. **Sample grids**: is every row recognisably its class, with diversity across columns? Compare the guidance blocks: too much guidance gives saturated, low-diversity images.
4. **Memorization** (optional, pixel space, no pretrained features):

   ```bash
   python generate.py --network runs/<run>/network-snapshot-XXXXXX.pkl --outdir gen-check --per-class 32 --guidance 2
   python tools/memorization_check.py --gen gen-check --data /data/prepared-64.zip --resolution 64 --outdir memcheck
   ```

   This writes nearest-neighbour pairs (generated | closest training image), `nearest.csv`, and a `summary.json` that includes the fraction of samples below an RMS-distance threshold.

TensorBoard: add `--tensorboard 1`, then run `tensorboard --logdir runs`.

## 7. Sampling

```bash
SNAP=runs/<run>/network-snapshot-XXXXXX.pkl

# 16 images for every class, CFG weight 2, plus a grid (rows = classes)
python generate.py --network $SNAP --outdir out/g2 --per-class 16 --guidance 2 --grid 1

# Compare guidance strengths
for w in 1 2 4; do python generate.py --network $SNAP --outdir out/g$w --per-class 8 --guidance $w --grid 1; done

# Specific classes by name (or --class-idx 3)
python generate.py --network $SNAP --outdir out/ab --class cat_a,cat_b --seeds 0-63 --guidance 2

# Random classes, following the training class frequencies
python generate.py --network $SNAP --outdir out/rand --seeds 0-999 --class-dist data --guidance 2

# CFG only in a sigma interval (Kynkaanniemi et al. 2024)
python generate.py --network $SNAP --outdir out/gi --per-class 8 --guidance 3 --guidance-interval 0.3,5
```

- Output layout: `outdir/<label>/<seed>.png`, plus `outdir/metadata.csv` with columns `file,label,label_idx,seed,guidance`, plus `grid.png` with `--grid 1`.
- CFG: `D = D_uncond + w·(D_cond − D_uncond)`, computed in one forward pass (cond and null batches concatenated). `w = 1` is the plain conditional model and `w = 0` is unconditional.
- The sampler is EDM's Heun sampler: `--steps` (default 32), `--sigma-min`, `--sigma-max`, `--rho`, `--S-churn`/`--S-min`/`--S-max`/`--S-noise`. The default `--sigma-max` is the hint stored in the snapshot: 80, or 80·res/B when the noise shift is on.
- Determinism: the same snapshot, seed, class, guidance and batch composition give a bit-identical image. Changing the batch size can change float round-off, which at most flips one uint8 level.
- `--device cpu` works. Multi-GPU: `torchrun --standalone --nproc_per_node=N generate.py ...`.

## 8. Moving to 224 px (native)

```bash
python tools/prepare_dataset.py --csv /data/train.csv --image-root /data/images --resolution 224 --out /data/prepared-224.zip
# (no resize happens, but files are validated and packed)
python train.py --config configs/224px_3090.yaml --data /data/prepared-224.zip --outdir runs --cache 0
```

The 224 preset follows the *Simple Diffusion* recipe (Hoogeboom et al. 2023). **It is untuned.** Do a throughput probe first (section 3). The knobs:

| Knob | Preset | What it does |
|---|---|---|
| `--patch-size 2` | 2 | `pixel_unshuffle(2)` at the input (3×224×224 → 12×112×112) and `pixel_shuffle(2)` at the output. All internal resolutions are relative to 112. This is the main memory and compute saving at a small quality cost. |
| `--channel-mult 1,2,3,4` | 112→56→28→14 | Four levels. The internal resolution must be divisible by 2^(levels−1). Invalid topologies are rejected with a readable error. |
| `--attn-res 28,14` | 28, 14 | Self-attention only at low resolution (and in the bottleneck), using PyTorch SDPA (flash / memory-efficient kernels). |
| `--noise-shift-base 64` | 64 | Shifts the training noise distribution upward for large images: `P_mean_eff = P_mean + ln(res/B)` (= −1.2 + 1.25 at 224, i.e. log-SNR shifted by −2·ln(3.5)). The sampler's default σ_max becomes 80·res/B = 280 (cap: `--sigma-max-cap`). Both values are stored in the snapshot and logged. |
| `--grad-ckpt 1` | on | Recomputes UNet blocks in the backward pass: much less activation memory for about 30% more compute. |
| `--precision bf16` | bf16 | Autocast around the UNet only. Preconditioning, loss and sampler stay in fp32/fp64. |
| `--batch 128 --batch-gpu 32` | | Use the probe to find the largest `batch_gpu` that fits. |

## 9. Adding a new condition

Conditioning lives in `training/conditioning.py`:

- `ConditionEncoder`: maps one raw condition to a vector that is **added to the noise embedding** (and so modulates every residual block). It has `forward(value, drop_mask)` and `null_value(n, device)`.
- `ClassEmbedding`: `nn.Embedding(K+1, dim)`. Index `K` is the learned NULL token used for label dropout and CFG.
- `ScalarFourierEmbedding`: a sketch of a scalar condition (NaN = NULL). It is not yet wired to the CLI.
- `ConditionStack`: a dict of encoders. It sums their outputs and handles per-condition dropout plus a "drop all jointly" probability (`joint_dropout`).

Conditions travel as a dict `{"class": LongTensor[N], ...}` from the dataset (`CSVImageDataset.get_cond`) to the loss, the preconditioner and the UNet. The condition config is saved in every snapshot. To add a condition (for example a second categorical attribute or a scalar):

1. Write the encoder class, decorate it with `@persistence.persistent_class`, and register it in `ENCODER_TYPES`.
2. Emit the raw value under its name in `CSVImageDataset.get_cond` (read from a new CSV column).
3. Append an entry to the condition config built in `train.py` (`cond_config`). The planned CLI form is `--cond-cols name:type,...`.
4. In `generate.py`, pass the value in the `cond` dict. Missing keys mean NULL, so CFG over all conditions keeps working.

## 10. Known limitations

- There is **no pretrained quality metric**. Model selection relies on the validation denoising loss, sample grids and the memorization check. `fid.py` (EDM's FID with a pretrained Inception-v3 network) is kept as an optional tool, but it is **not meaningful for data far from ImageNet** and nothing depends on it.
- `--xflip` is off by default because it may not preserve the label. `--augment` uses EDM's non-leaking augmentation instead.
- The presets are starting points, not tuned hyper-parameters.
- The dataset requires square images.
- Each DataLoader worker keeps its own RAM cache.
- The new training loop does not support EDM's VP/VE preconditioning, the `ablation_sampler`, or `ImageFolderDataset`. Those classes remain in `training/` only for compatibility. EDM-style pickles still load (`net(x, sigma, class_labels)` with one-hot or index labels works).
- Multi-GPU (`torchrun`) code paths are kept but were not the target and are untested here.

## 11. Possible next steps

- **EDM2 architecture** (magnitude-preserving layers, post-hoc EMA; Karras et al. 2024). This is the likely upgrade if quality plateaus.
- A **domain classifier trained from scratch** on the internal data. It would give an "FID-like" metric in its own feature space and measure the class accuracy of generated samples.
- Multi-GPU tuning.

## Repository layout

```
train.py                  CLI (YAML presets + overrides) -> training/training_loop.py
generate.py               sampling: CFG, class names, grids, metadata.csv
training/dataset.py       CSVImageDataset (raw CSV or prepared zip/dir)
training/conditioning.py  modular condition encoders, NULL token, dropout
training/networks.py      SongUNet / DhariwalUNet (patching, SDPA, grad ckpt) + EDMPrecond (autocast)
training/loss.py          EDMLoss + noise shift + sigma bins
training/sampling.py      EDM Heun sampler with CFG / guidance interval, grids
training/training_loop.py loop: accumulation, EMAs, val loss, grids, snapshots, resume
tools/                    prepare_dataset, make_synthetic_dataset, plot_stats, memorization_check
configs/                  64px_3090.yaml, 224px_3090.yaml
tests/                    pytest, CPU only
```
