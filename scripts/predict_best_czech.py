#!/usr/bin/env python3
"""Load model_best.pth, run inference on the val set, print score distribution,
and save visualizations with a low (0.05) score threshold so we can actually see
the model's predictions — not just confident ones."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent))

from dasnet.data.das import DASTrainDataset
from dasnet.model.dasnet import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--anno-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = DASTrainDataset(
        ann_path=args.anno_path,
        root_dir=args.data_path,
        resize_scale=0.5,
        synthetic_noise=False,
        enable_stack_event=False,
        enable_vflip=False,
        allowed_bins={},
        min_keep_bins={},
    )
    print(f"val samples: {len(ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name="maskrcnn_resnet50_selectable_fpn",
        num_classes=args.num_classes,
        pretrained=False,
        pretrained_backbone=False,
        trainable_backbone_layers=3,
        target_layer="P4",
        box_roi_pool_size=7,
        mask_roi_pool_size=(42, 42),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        box_positive_fraction=0.25,
        max_size=6000,
        min_size=1422,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded. missing={len(missing)} unexpected={len(unexpected)}")
    # Disable internal filter so we see the raw score distribution
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.detections_per_img = 200
    print(f"roi_heads.score_thresh overridden to {model.roi_heads.score_thresh}")
    model.eval()

    dt = 0.01
    dx = 5.1 * 2

    n = min(len(ds), args.max_samples)
    for i in range(n):
        image, target = ds[i]
        with torch.no_grad():
            outputs = model([image.to(device)])
        pred = outputs[0]

        boxes = pred["boxes"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()

        print(f"\n=== sample {i} ===")
        print(f"  gt boxes: {target['boxes'].shape[0]}")
        print(f"  pred count: {len(scores)}")
        if len(scores) > 0:
            print(f"  score stats: min={scores.min():.3f} max={scores.max():.3f} "
                  f"mean={scores.mean():.3f} p50={np.median(scores):.3f}")
            print(f"  scores > 0.5: {(scores > 0.5).sum()}")
            print(f"  scores > 0.2: {(scores > 0.2).sum()}")
            print(f"  scores > 0.05: {(scores > 0.05).sum()}")

        img = image.detach().cpu().permute(1, 2, 0).numpy()
        H, W, _ = img.shape
        extent = (0, W * dt, 0, H * dx / 1e3)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        base = np.mean(img, axis=2)
        vstd = np.std(base) * 2 + 1e-12
        for ax, title in zip(axes, ["GT", f"Pred (>{args.score_thresh})"]):
            ax.imshow(base - np.mean(base), cmap="seismic",
                      vmin=-vstd, vmax=vstd, extent=extent,
                      aspect="auto", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Distance (km)")

        gt_boxes = target["boxes"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()
        for b, lbl in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = b
            rect = patches.Rectangle(
                (x1 * dt, y1 * dx / 1e3),
                (x2 - x1) * dt, (y2 - y1) * dx / 1e3,
                linewidth=1.5, edgecolor="lime", facecolor="none")
            axes[0].add_patch(rect)
            axes[0].text(x1 * dt, y2 * dx / 1e3, f"GT:{int(lbl)}",
                         color="lime", fontsize=8, weight="bold")

        for b, s, lbl in zip(boxes, scores, labels):
            if s < args.score_thresh:
                continue
            x1, y1, x2, y2 = b
            rect = patches.Rectangle(
                (x1 * dt, y1 * dx / 1e3),
                (x2 - x1) * dt, (y2 - y1) * dx / 1e3,
                linewidth=1.2, edgecolor="red", facecolor="none")
            axes[1].add_patch(rect)
            axes[1].text(x1 * dt, y2 * dx / 1e3,
                         f"P:{int(lbl)}({s:.2f})",
                         color="red", fontsize=7, weight="bold")

        out_path = os.path.join(args.out_dir, f"sample_{i:02d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
