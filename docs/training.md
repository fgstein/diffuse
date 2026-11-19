# Training

## Presets

`--config` loads a YAML preset. **Explicit command-line flags always override it.** YAML keys are the option names with `_` instead of `-` (e.g. `batch_gpu`, `attn_res`). Lists are written as YAML lists.

| Preset | Network | Params | Batch | Notes |
|---|---|---|---|---|
| [`configs/64px_3090.yaml`](../configs/64px_3090.yaml) | SongUNet (DDPM++), 128 ch, mult `1,2,2,2`, 3 blocks, attention at 16, 8 | ~52 M | 256 (128 per step) | lr 2e-4, 2 Mimg ramp-up, EMA half-life 0.5 Mimg, bf16 |
| [`configs/224px_3090.yaml`](../configs/224px_3090.yaml) | SongUNet, patch 2, 128 ch, mult `1,2,3,4`, 2 blocks, attention at 28, 14 | ~93 M | 128 (32 per step) | noise shift base 64, grad checkpointing, lr 1e-4 |

> **The presets are untuned starting points.** Always run the throughput probe first.

The fully resolved configuration of a run is written to `<run>/training_options.json`.

## 1. Size the run (throughput probe)

Run 2–3 short ticks with the preset you intend to use:

```bash
python train.py --config configs/64px_3090.yaml --data /data/prepared-64.zip --outdir runs-probe \
    --tick 5 --duration 0.015 --snap 1000 --dump 1000 --sample-every 0 --val-every 0
```

Each tick prints a line like:

```
tick 2  kimg 10.2  time 1m 05s  sec/tick 30.1  sec/kimg 5.90  maintenance 0.0  loss 0.4123  lr 1.0e-06  cpumem 3.1  gpumem 17.3  reserved 18.0
```

Ignore tick 0, which includes start-up and cuDNN autotuning. Then:

- **`--batch-gpu`**: the largest value whose `reserved` memory stays under ~22 GB. `--batch` is the *effective* batch. When `batch > batch_gpu`, gradients are accumulated over `batch / batch_gpu` rounds, so the optimisation is identical. On out-of-memory errors, halve `--batch-gpu`.
- **`--duration`** (Mimg) = available hours × 3.6 / `sec/kimg`.

  | Budget | sec/kimg | duration |
  |---|---|---|
  | 72 h | 5.9 | 72 × 3.6 / 5.9 ≈ **44** |
  | 48 h | 12.0 | 48 × 3.6 / 12 ≈ **14** |

- If `sec/kimg` barely changes when you halve `--batch-gpu`, or the GPU is not saturated (`nvidia-smi` shows low utilisation), data loading is the bottleneck. See [troubleshooting](troubleshooting.md#training-is-slow--gpu-utilisation-is-low).

## 2. Train

```bash
# Prepared zip (recommended)
python train.py --config configs/64px_3090.yaml --data /data/prepared-64.zip --outdir runs --duration 44

# Raw CSV, resized on the fly
python train.py --config configs/64px_3090.yaml --csv /data/train.csv --image-root /data/images \
    --path-col path --label-col label --resolution 64 --outdir runs --duration 44
```

Each run gets a new directory `runs/<id>-<dataset>-<res>px-cond<K>-<arch>-gpus<N>-batch<B>-<precision>[-<desc>]`. Use `--desc` to tag it. For long runs, launch inside `tmux` or `screen`, or under `nohup`.

## 3. Stop and resume

**Stopping:** press **Ctrl-C** once (or send `SIGTERM`, e.g. from a job scheduler). The current step finishes, a network snapshot and a training state are written, and the process exits. A second Ctrl-C aborts immediately.

**Resuming:** run the **same command** plus `--resume`:

```bash
python train.py --config configs/64px_3090.yaml --data /data/prepared-64.zip --outdir runs --duration 44 \
    --resume runs/00003-prepared-64-64px-cond12-ddpmpp-gpus1-batch256-bf16/training-state-012345.pt
```

- Training continues **in the same run directory** and appends to `log.txt` and `stats.jsonl`.
- The resume restores the network, the optimizer, **all EMAs**, the image counter (kimg), the tick, the RNG states, and the **exact position in the data stream**: the stream is a seeded sequence of epochs, and resume jumps to `cur_nimg` in O(1).
- The saved label map is reused. New labels in the CSV are an error.
- You can change `--duration` (to extend a run) and monitoring options. Changing the architecture fails with a "missing tensor" error.

`--transfer net.pkl` is different: it only initialises the weights (network and EMAs) from a snapshot and starts a fresh optimizer, counter and EMA ramp-up. Weights are matched **by name**, and layer names contain the feature-map resolution (e.g. `enc.32x32_block0`), so the architecture and resolution must match. Mismatching layers are silently left at their random initialisation.

## Key options

The full list is in the [CLI reference](cli-reference.md#trainpy).

### Model

| Option | Default | Notes |
|---|---|---|
| `--arch ddpmpp\|ncsnpp\|adm` | `ddpmpp` | Preset for the UNet family. The explicit topology flags below override it. `adm` = DhariwalUNet. |
| `--model-channels`, `--channel-mult`, `--num-blocks` | preset | Width, per-level multipliers, residual blocks per level. |
| `--attn-res` | preset | Feature-map resolutions that get self-attention. They must exist in the UNet (checked). |
| `--num-heads` / `--head-channels` | preset | Attention heads (SongUNet default 1; ADM: 64 channels per head). |
| `--dropout` | 0.1 | Dropout inside residual blocks. |
| `--patch-size` | 1 | See [224 px](224px.md#patching). |

### Conditioning

| Option | Default | Notes |
|---|---|---|
| `--label-dropout` | 0.1 | Fraction of samples trained with the **NULL** label. **Must be > 0 for CFG to work** (otherwise the NULL token is never trained). |
| `--uncond 1` | off | Unconditional model, for debugging. |

### Optimisation

| Option | Default | Notes |
|---|---|---|
| `--lr` | 2e-4 | Adam. AdamW when `--weight-decay > 0`. |
| `--lr-rampup` | 2000 kimg | Linear warm-up. |
| `--lr-decay invsqrt --lr-ref-kimg K` | none | EDM2-style `lr / sqrt(max(kimg / K, 1))`. |
| `--grad-clip` | 0 (off) | Global-norm clipping. The norm is logged as `Loss/grad_norm`. NaN/Inf gradients are always zeroed (EDM behaviour). |
| `--ema` | 0.5 Mimg | EMA half-life, with EDM's ramp-up (half-life ≤ 5% of the images seen so far). |
| `--ema-list 0.05,0.5,2` | – | Several EMAs at once. The **first** is the primary one (grids, validation, `ema` key). Pick another with `generate.py --ema-halflife`. |

### Class imbalance

`--class-balanced 1` samples images with weight ∝ `1 / count(class)^α` (`--balance-alpha`, default 0.5). α = 0 is uniform over images, α = 1 is uniform over classes. 0.5 is a compromise. Rare classes are then repeated more often, so watch for memorization of them.

### Noise distribution

| Option | Default | Notes |
|---|---|---|
| `--P-mean`, `--P-std` | −1.2, 1.2 | `ln σ ~ N(P_mean_eff, P_std²)` during training (EDM). |
| `--noise-shift-base B` | 0 (off) | `P_mean_eff = P_mean + ln(res / B)`. See [224 px](224px.md#noise-shift). |

### Performance

| Option | Default | Notes |
|---|---|---|
| `--precision fp32\|fp16\|bf16` | bf16 | Autocast around the UNet only. bf16 needs no loss scaling. fp16 uses a static `--ls`. |
| `--tf32` | 1 | TF32 matmuls and convolutions on Ampere. |
| `--grad-ckpt` | 0 | Recompute blocks in the backward pass: less memory, ~30% slower. |
| `--compile` | 0 | `torch.compile` the training forward pass (snapshots still pickle the plain module). |
| `--workers` | 4 | DataLoader processes. |
| `--cache` | 1 | Per-worker RAM cache, see [data](data.md#ram-cache). |
| `--device cuda\|cpu` | cuda if available | CPU works for tests. |

### Monitoring and I/O

| Option | Default | Notes |
|---|---|---|
| `--tick` | 50 kimg | Status line period. |
| `--snap` / `--dump` | 20 / 100 ticks | Network snapshot / full training state period. |
| `--sample-every` / `--val-every` | 20 ticks | Sample grids / validation loss (0 = off). |
| `--sample-guidance` | `1,2,4` | One grid block per value. |
| `--keep-snapshots N` | 10 | Keep only the last N snapshots and states… |
| `--keep-every KIMG` | 0 | …plus those at multiples of KIMG. |
| `--tensorboard 1` | off | Needs the `tensorboard` extra. |

## Multi-GPU

EDM's `torchrun` path is kept (`torchrun --standalone --nproc_per_node=N train.py ...`, NCCL on CUDA). It was not the target of this fork and has not been tested.
