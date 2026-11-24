# Documentation

| # | Page | You will learn |
|---|---|---|
| 1 | [Installation](installation.md) | uv setup on the RTX 3090 or on CPU, checking CUDA, running the tests |
| 2 | [Data](data.md) | the CSV format, labels, the validation split, `prepare_dataset.py`, caching |
| 3 | [Training](training.md) | presets, sizing `batch_gpu` / `duration`, the key knobs, resume, stopping |
| 4 | [Monitoring](monitoring.md) | run directory contents, validation loss, per-σ curves, grids, memorization check |
| 5 | [Sampling](sampling.md) | `generate.py`: classes, CFG, guidance interval, grids, determinism |
| 6 | [Going to 224 px](224px.md) | patching, low-resolution attention, noise shift, gradient checkpointing |
| 6b | [Sim-to-real style transfer](style-transfer.md) | SDEdit / ILVR translation of simulated images, choosing σ₀, measuring the domain gap |
| 7 | [Conditioning & extensions](conditioning.md) | how conditions flow, NULL token, adding a new condition |
| 8 | [How it works](how-it-works.md) | EDM formulation, CFG maths, code map, snapshot and state file contents |
| 9 | [Troubleshooting](troubleshooting.md) | out of memory, slow data loading, NaN, resume errors, odd samples |
| 10 | [CLI reference](cli-reference.md) | every option of every script, generated from the code |
| – | [Smoke-test results](smoke-test/README.md) | what the CPU end-to-end and overfit tests produced |

**The 5-minute path:** [Installation](installation.md) → [prepare the data](data.md#preparing-the-data) → [throughput probe](training.md#1-size-the-run-throughput-probe) → [train](training.md#2-train) → [watch the grids](monitoring.md).
