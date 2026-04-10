# DASNet-workshop

Mask R-CNN for DAS (Distributed Acoustic Sensing) event detection on HDF5 waveform data.

See `CLAUDE.md` for architecture, data conventions, and gotchas.

## Data preparation

### 9-class COCO (original)

Czech raw → DASNet-ready conversion (idempotent, h5 are skipped on re-run):

```powershell
uv run python scripts/czech_das_to_dasnet.py --src-root e:/dasdata/DAS-dataset/data --out-root e:/dasdata/DAS-dataset_dasnet/data --coco-dir e:/dasdata/DAS-dataset_dasnet/annotations --workers 16
```

### Binary COCO (event-only)

With only 12 training samples across 9 classes (~1.3 per class), fine-grained
classification is not learnable. Collapse all event types into a single `event`
class. Outputs to a **separate** `annotations_bin/` directory (h5 are reused):

```powershell
uv run python scripts/czech_das_to_dasnet_objection.py --src-root e:/dasdata/DAS-dataset/data --out-root e:/dasdata/DAS-dataset_dasnet/data --coco-dir e:/dasdata/DAS-dataset_dasnet/annotations_bin --workers 16
```

Verify:
```powershell
uv run python -c "import json; j=json.load(open('e:/dasdata/DAS-dataset_dasnet/annotations_bin/train.json')); print('cats:', j['categories']); print('sample cat_id:', j['annotations'][0]['category_id'])"
```
Expected: `cats: [{'id': 1, 'name': 'event'}]` and `sample cat_id: 1`.

## Training on the Czech DAS dataset

### One-time setup

Log in to WandB (once per machine — API key from https://wandb.ai/authorize):

```powershell
uv run wandb login
```

`wandb` is already in `pyproject.toml` — no extra install needed.

### Launch binary training (recommended)

**Must use `--batch-size 1`**: DASNet's `roi_heads.expand_features`
pads each sample in the batch by a per-sample scale, so samples in
the same batch end up with different spatial sizes and `torch.cat`
crashes. Use `--accumulation-steps` to simulate a larger effective batch.

`--clip-value 1.0` prevents `inf` in `loss_rpn_box_reg` caused by the
extreme anchor aspect ratios (5–333) when regression targets are large.

Startup should log `[train_EC] auto num_classes = 1 + 1 categories = 2`.

```powershell
uv run python train_EC.py `
  --data-path e:/dasdata/DAS-dataset_dasnet/data `
  --anno-path e:/dasdata/DAS-dataset_dasnet/annotations_bin/train.json `
  --val-data-path e:/dasdata/DAS-dataset_dasnet/data `
  --val-anno-path e:/dasdata/DAS-dataset_dasnet/annotations_bin/val.json `
  --output-dir ./output/czech_run_bin_v3 `
  --batch-size 1 `
  --accumulation-steps 2 `
  --epochs 60 `
  --workers 4 `
  --clip-value 1.0 `
  --wandb --wandb-project DASNet --wandb-name czech_run_bin_v3
```

Checkpoints land in the output dir (`checkpoint.pth`, `model_best.pth`,
per-epoch `model_N.pth`, plus `figures/`).

### Key CLI flags (added in train_EC.py)

| Flag | Default | Purpose |
|---|---|---|
| `--resize-scale` | 0.5 | Spatial resize ratio before model input |
| `--data-key` | `data` | HDF5 dataset name in each input .h5 |
| `--no-strain-rate` | (off) | Treat input as raw strain (DASNet will diff it) |
| `--synthetic-noise` | off | Enable noise overlay augmentation |
| `--noise-csv` | None | Path to noise manifest (required if `--synthetic-noise`) |
| `--clip-value` | None | Gradient clipping value (recommended: 1.0) |
| `--accumulation-steps` | 1 | Gradient accumulation steps (simulates larger batch) |

## Inference

Run a trained checkpoint, dump per-sample bbox/mask IoU and score distributions,
save 3-panel GT vs Pred visualizations:

```powershell
uv run python scripts/predict_best_czech.py `
  --checkpoint ./output/czech_run_bin_v3/model_best.pth `
  --data-path e:/dasdata/DAS-dataset_dasnet/data `
  --anno-path e:/dasdata/DAS-dataset_dasnet/annotations_bin/val.json `
  --out-dir ./output/czech_run_bin_v3/inference `
  --num-classes 2 `
  --score-thresh 0.1 `
  --max-samples 4
```

## Conversion diagnostics

Side-by-side visualization of raw → decimated → DASTrainDataset:

```powershell
uv run python scripts/visualize_conversion.py `
  --src-h5 e:/dasdata/DAS-dataset/data/car/auto_2023-04-17T124152+0100.h5 `
  --src-json e:/dasdata/DAS-dataset/data/car/auto_2023-04-17T124152+0100.json `
  --conv-h5 e:/dasdata/DAS-dataset_dasnet/data/car/auto_2023-04-17T124152+0100.h5 `
  --coco e:/dasdata/DAS-dataset_dasnet/annotations_bin/train.json `
  --out ./output/conversion_check/car_auto.png
```

## Anchor analysis

Analyze the model-space bbox H/W distribution and suggest anchor ratios:

```powershell
uv run python scripts/analyze_bbox_shapes.py --coco e:/dasdata/DAS-dataset_dasnet/annotations_bin/train.json
```
