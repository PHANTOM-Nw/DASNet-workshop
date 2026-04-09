#!/usr/bin/env python3
"""One-shot diagnostic: run DASTrainDataset[0] -> model.forward -> print loss_mask path.

Expected output (verifies two hypotheses):
  target.attention_masks.sum() == 0  -> confirms missing attention_mask field
  losses['loss_mask'] == 0 (or nan)  -> gated to zero by attention_masks
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dasnet.data.das import DASTrainDataset
from dasnet.model.dasnet import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--anno-path", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = DASTrainDataset(
        ann_path=args.anno_path,
        root_dir=args.data_path,
        resize_scale=0.25,
        synthetic_noise=False,
        enable_stack_event=False,
        enable_vflip=False,
        allowed_bins={},
        min_keep_bins={},
    )
    image, target = ds[0]
    print(f"image: shape={tuple(image.shape)} dtype={image.dtype}")
    print(f"target.boxes: shape={tuple(target['boxes'].shape)}")
    print(f"target.labels: {target['labels'].tolist()}")
    print(f"target.masks: shape={tuple(target['masks'].shape)} sum={int(target['masks'].sum())}")
    print(f"target.attention_masks: shape={tuple(target['attention_masks'].shape)} sum={int(target['attention_masks'].sum())}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(
        model_name="maskrcnn_resnet50_selectable_fpn",
        num_classes=11,
        pretrained=False,
        pretrained_backbone=True,
    ).to(device).train()

    image = image.to(device)
    target = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()}

    losses = model([image], [target])
    print("\nlosses dict:")
    for k, v in losses.items():
        v_item = float(v.detach().cpu()) if torch.is_tensor(v) and v.numel() == 1 else v
        print(f"  {k}: {v_item}")

    total = sum(v for v in losses.values() if torch.is_tensor(v))
    total.backward()
    print(f"\ntotal loss: {float(total):.6f}")
    print("backward() ok")


if __name__ == "__main__":
    main()
