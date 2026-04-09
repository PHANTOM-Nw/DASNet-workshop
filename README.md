# DASNet-workshop

Mask R-CNN for DAS (Distributed Acoustic Sensing) event detection on HDF5 waveform data.

See `CLAUDE.md` for architecture, data conventions, and gotchas.

## Training on the Czech DAS dataset

### One-time setup

Log in to WandB (once per machine — API key from https://wandb.ai/authorize):

```bash
uv run wandb login
```

`wandb` is already in `pyproject.toml` — no extra `pip install` needed.

### Launch full training

**Must use `--batch-size 1`**: DASNet's `roi_heads.expand_features`
pads each sample in the batch by a per-sample scale, so samples in
the same batch end up with different spatial sizes and `torch.cat`
crashes. Use `--accumulation-steps` to simulate a larger effective
batch.

**cmd.exe:**
```bat
uv run python train_EC.py ^
  --data-path e:/dasdata/DAS-dataset_dasnet/data ^
  --anno-path e:/dasdata/DAS-dataset_dasnet/annotations/train.json ^
  --val-data-path e:/dasdata/DAS-dataset_dasnet/data ^
  --val-anno-path e:/dasdata/DAS-dataset_dasnet/annotations/val.json ^
  --output-dir ./output/czech_run1 ^
  --batch-size 1 ^
  --accumulation-steps 2 ^
  --epochs 80 ^
  --workers 4 ^
  --wandb --wandb-project DASNet --wandb-name czech_run1
```

**PowerShell** (note the backtick `` ` `` continuation — no trailing space allowed):
```powershell
uv run python train_EC.py `
  --data-path e:/dasdata/DAS-dataset_dasnet/data `
  --anno-path e:/dasdata/DAS-dataset_dasnet/annotations/train.json `
  --val-data-path e:/dasdata/DAS-dataset_dasnet/data `
  --val-anno-path e:/dasdata/DAS-dataset_dasnet/annotations/val.json `
  --output-dir ./output/czech_run1 `
  --batch-size 1 `
  --accumulation-steps 2 `
  --epochs 80 `
  --workers 4 `
  --wandb --wandb-project DASNet --wandb-name czech_run1
```

Checkpoints land in `output/czech_run1/` (`checkpoint.pth`, `model_best.pth`,
per-epoch `model_N.pth`, plus `figures/`).

### Smoke test (1 epoch, no wandb)

Use before any config change to verify the pipeline is healthy:

```bash
uv run python train_EC.py ^
  --data-path e:/dasdata/DAS-dataset_dasnet/data ^
  --anno-path e:/dasdata/DAS-dataset_dasnet/annotations/train.json ^
  --val-data-path e:/dasdata/DAS-dataset_dasnet/data ^
  --val-anno-path e:/dasdata/DAS-dataset_dasnet/annotations/val.json ^
  --output-dir ./output/czech_smoke ^
  --batch-size 1 --epochs 1 --workers 0
```

Expected:
- `[train_EC] auto num_classes = 1 + 9 categories = 10`
- 12/12 training steps complete, `loss_mask > 0` on at least one step
- No `ZeroDivisionError`, no NaN/Inf
- Checkpoints + `figures/epoch00_*.png` written

### Key CLI flags (added in train_EC.py)

| Flag | Default | Purpose |
|---|---|---|
| `--resize-scale` | 0.5 | Spatial resize ratio before model input |
| `--data-key` | `data` | HDF5 dataset name in each input .h5 |
| `--no-strain-rate` | (off) | Treat input as raw strain (DASNet will diff it) |
| `--synthetic-noise` | off | Enable noise overlay augmentation |
| `--noise-csv` | None | Path to noise manifest (required if `--synthetic-noise`) |

## Data preparation

Czech raw → DASNet-ready conversion (idempotent, h5 are skipped on re-run):

```bash
uv run python scripts/czech_das_to_dasnet.py ^
  --src-root e:/dasdata/DAS-dataset/data ^
  --out-root e:/dasdata/DAS-dataset_dasnet/data ^
  --coco-dir e:/dasdata/DAS-dataset_dasnet/annotations ^
  --workers 16
```
