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

        gt_boxes_t = target["boxes"].detach().cpu()

        print(f"\n=== sample {i} ===")
        print(f"  gt boxes: {gt_boxes_t.shape[0]}")
        print(f"  pred count: {len(scores)}")
        if len(scores) > 0:
            print(f"  score stats: min={scores.min():.3f} max={scores.max():.3f} "
                  f"mean={scores.mean():.3f} p50={np.median(scores):.3f}")
            print(f"  scores > 0.5: {(scores > 0.5).sum()}")
            print(f"  scores > 0.2: {(scores > 0.2).sum()}")
            print(f"  scores > 0.05: {(scores > 0.05).sum()}")

        # IoU analysis
        if len(scores) > 0 and gt_boxes_t.shape[0] > 0:
            from torchvision.ops import box_iou
            pred_boxes_t = torch.tensor(boxes)
            iou_matrix = box_iou(pred_boxes_t, gt_boxes_t).numpy()  # (N_pred, N_gt)
            # For each GT, find best matching pred
            best_iou_per_gt = iou_matrix.max(axis=0)  # (N_gt,)
            best_pred_per_gt = iou_matrix.argmax(axis=0)
            print(f"  --- IoU per GT box (best matching pred) ---")
            for g in range(len(best_iou_per_gt)):
                p = best_pred_per_gt[g]
                print(f"    GT {g}: best IoU={best_iou_per_gt[g]:.4f} "
                      f"(pred #{p}, score={scores[p]:.3f})")
            print(f"  IoU summary: min={best_iou_per_gt.min():.4f} "
                  f"max={best_iou_per_gt.max():.4f} "
                  f"mean={best_iou_per_gt.mean():.4f}")
            print(f"  GT with bbox IoU > 0.3: {(best_iou_per_gt > 0.3).sum()}/{len(best_iou_per_gt)}")
            print(f"  GT with bbox IoU > 0.5: {(best_iou_per_gt > 0.5).sum()}/{len(best_iou_per_gt)}")

            # Mask IoU: compare pred mask (sigmoid prob > 0.5) vs GT mask (binary)
            gt_masks_t = target.get("masks", None)
            pred_masks_t = pred.get("masks", None)
            if gt_masks_t is not None and pred_masks_t is not None:
                gt_masks_np = gt_masks_t.detach().cpu().numpy()   # (N_gt, H, W)
                pm_np = pred_masks_t.detach().cpu().numpy()       # (N_pred, 1, H, W)
                print(f"  --- Mask IoU per GT (best bbox-matched pred) ---")
                mask_ious = []
                for g in range(len(best_iou_per_gt)):
                    p = best_pred_per_gt[g]
                    gt_m = (gt_masks_np[g] > 0.5).astype(np.float32)
                    pr_m = (np.squeeze(pm_np[p]) > 0.5).astype(np.float32)
                    # Resize pred mask to GT mask shape if different
                    if pr_m.shape != gt_m.shape:
                        from PIL import Image as PILImage
                        pr_m = np.array(PILImage.fromarray(pr_m).resize(
                            (gt_m.shape[1], gt_m.shape[0]),
                            PILImage.NEAREST
                        )).astype(np.float32)
                    inter = (gt_m * pr_m).sum()
                    union = ((gt_m + pr_m) > 0).sum()
                    miou = float(inter / union) if union > 0 else 0.0
                    mask_ious.append(miou)
                    print(f"    GT {g}: mask IoU={miou:.4f} (pred #{p})")
                mask_ious = np.array(mask_ious)
                print(f"  Mask IoU summary: min={mask_ious.min():.4f} "
                      f"max={mask_ious.max():.4f} mean={mask_ious.mean():.4f}")
                print(f"  GT with mask IoU > 0.3: {(mask_ious > 0.3).sum()}/{len(mask_ious)}")
                print(f"  GT with mask IoU > 0.5: {(mask_ious > 0.5).sum()}/{len(mask_ious)}")

        img = image.detach().cpu().permute(1, 2, 0).numpy()
        H, W, _ = img.shape
        extent = (0, W * dt, 0, H * dx / 1e3)

        # 3 panels: GT (box+mask), Pred bbox, Pred mask
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        base = np.mean(img, axis=2)
        vstd = np.std(base) * 2 + 1e-12
        for ax, title in zip(axes, ["GT (box+mask)", f"Pred bbox (>{args.score_thresh})", "Pred mask"]):
            ax.imshow(base - np.mean(base), cmap="seismic",
                      vmin=-vstd, vmax=vstd, extent=extent,
                      aspect="auto", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Distance (km)")

        # --- GT panel: boxes + masks ---
        gt_boxes = target["boxes"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()
        gt_masks = target.get("masks", None)
        if gt_masks is not None:
            gt_masks_np = gt_masks.detach().cpu().numpy()
            combined_gt = np.zeros((H, W), dtype=np.float32)
            for m in gt_masks_np:
                combined_gt = np.maximum(combined_gt, m.astype(np.float32))
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[..., 1] = 1.0  # green
            rgba[..., 3] = combined_gt * 0.4
            axes[0].imshow(rgba, extent=extent, aspect="auto", origin="lower")
        for b, lbl in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = b
            rect = patches.Rectangle(
                (x1 * dt, y1 * dx / 1e3),
                (x2 - x1) * dt, (y2 - y1) * dx / 1e3,
                linewidth=1.5, edgecolor="lime", facecolor="none")
            axes[0].add_patch(rect)
            axes[0].text(x1 * dt, y2 * dx / 1e3, f"GT:{int(lbl)}",
                         color="lime", fontsize=8, weight="bold")

        # --- Pred bbox panel ---
        pred_masks = pred.get("masks", None)
        for j, (b, s, lbl) in enumerate(zip(boxes, scores, labels)):
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

        # --- Pred mask panel: top-K by score ---
        if pred_masks is not None:
            pmasks = pred_masks.detach().cpu().numpy()
            top_k = min(10, len(scores))
            top_idx = np.argsort(scores)[::-1][:top_k]
            combined_pred = np.zeros((H, W), dtype=np.float32)
            for j in top_idx:
                if scores[j] < args.score_thresh:
                    continue
                m = np.squeeze(pmasks[j])
                m = np.clip(m, 0, 1)
                combined_pred = np.maximum(combined_pred, m * scores[j])
            if combined_pred.max() > 0:
                combined_pred /= combined_pred.max()
            rgba = plt.get_cmap("hot")(combined_pred)
            rgba[..., 3] = np.where(combined_pred > 0.1, 0.5, 0.0)
            axes[2].imshow(rgba, extent=extent, aspect="auto", origin="lower")
            # also draw top-K boxes on mask panel
            for j in top_idx:
                if scores[j] < args.score_thresh:
                    continue
                x1, y1, x2, y2 = boxes[j]
                rect = patches.Rectangle(
                    (x1 * dt, y1 * dx / 1e3),
                    (x2 - x1) * dt, (y2 - y1) * dx / 1e3,
                    linewidth=0.8, edgecolor="yellow", facecolor="none")
                axes[2].add_patch(rect)
                axes[2].text(x1 * dt, y2 * dx / 1e3,
                             f"{scores[j]:.2f}",
                             color="yellow", fontsize=6)

        out_path = os.path.join(args.out_dir, f"sample_{i:02d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
