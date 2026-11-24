# Installation

Requirements: Linux, Python ≥ 3.10, and [uv](https://docs.astral.sh/uv/). For GPU training you also need an NVIDIA GPU (developed for a single **RTX 3090, 24 GB**) and an **NVIDIA driver ≥ 560**.

No network access is needed at training or sampling time: nothing is downloaded and there are no pretrained weights.

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Create the environment

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Torch comes in two flavours, chosen with an **extra**. Pick exactly one:

```bash
git clone <this repo> edm-cond && cd edm-cond

uv sync --extra cu126      # GPU machine (RTX 3090): torch 2.14.x + CUDA 12.6
# or
uv sync --extra cpu        # CPU-only machine: tests, debugging, tiny runs
```

Optional extras can be combined with either flavour:

| Extra | Adds | For |
|---|---|---|
| `tensorboard` | tensorboard | `train.py --tensorboard 1` |
| `fid` | scipy | the optional `fid.py` (uses a *pretrained* Inception network, see [limitations](../README.md#limitations)) |

Example: `uv sync --extra cu126 --extra tensorboard`.

> **Repeat the extra on every `uv sync`.** A later `uv sync` without `--extra cu126` removes torch. A plain `uv run ...` keeps whatever is installed.

## 3. Check CUDA

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
# expected: 2.14.1+cu126 True NVIDIA GeForce RTX 3090
```

You can prefix every command with `uv run`, or activate the environment once with `source .venv/bin/activate`. The rest of the documentation writes `python ...` and assumes the environment is active.

## 4. Run the test suite

```bash
uv run pytest -q        # CPU only, ~35 s, 46 tests
```

The tests build a tiny synthetic dataset and cover the dataset, the network, sampling with CFG, snapshot loading in a fresh process, and the CLI with its presets.

## 5. Optional: end-to-end smoke test (CPU, ~5 min)

```bash
python tools/make_synthetic_dataset.py --out /tmp/synth --n-per-class 30 --classes 3
python train.py --device cpu --outdir /tmp/runs --csv /tmp/synth/data.csv --image-root /tmp/synth \
  --resolution 32 --model-channels 32 --channel-mult 1,2,2 --num-blocks 1 --attn-res 8 \
  --batch 32 --duration 0.02 --tick 2 --snap 2 --sample-every 2 --val-every 2 --precision fp32 --workers 0
python generate.py --device cpu --network "$(ls /tmp/runs/*/network-snapshot-*.pkl | tail -1)" \
  --outdir /tmp/gen --per-class 4 --guidance 2 --steps 18 --grid 1
```

## Using a newer CUDA build

The `cu126` index gives torch 2.14.x. PyTorch publishes torch 2.15 only for CUDA 13.2+, which needs a very recent driver. If your driver supports it:

1. In `pyproject.toml`, change the URL of the `pytorch-cu126` index, e.g. to `https://download.pytorch.org/whl/cu132`.
2. Run `uv lock && uv sync --extra cu126`. The extra keeps its name; only the index changes.
