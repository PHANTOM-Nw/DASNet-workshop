#!/usr/bin/env python3
"""Side-by-side visualization of raw Czech DAS data vs DASNet-converted output.

Panel 1: Raw h5 (strain-rate waveform) with original JSON curve overlay
Panel 2: Decimated h5 (DASNet input) with COCO bbox + segmentation overlay
Panel 3: What DASTrainDataset actually produces (3-ch RGB + masks)

This script diagnoses whether the conversion pipeline preserves event locations.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent))


DECIMATE = 40  # must match czech_das_to_dasnet.py


def load_raw(src_h5: Path, src_json: Path):
    """Load raw Czech h5, decimate in-place to match converted resolution."""
    with h5py.File(src_h5, "r") as f:
        for key in ["Acquisition/Raw[0]/RawData", "data"]:
            if key in f:
                dset = f[key]
                nt_full, nch = dset.shape
                # Read every DECIMATE-th time sample → same resolution as converted
                data = dset[::DECIMATE, :].astype(np.float32)
                print(f"  raw: {nt_full}×{nch} → decimated to {data.shape[0]}×{nch}")
                break
        else:
            keys = list(f.keys())
            raise KeyError(f"Cannot find data in {src_h5}, keys={keys}")

    curves = json.loads(src_json.read_text())
    return data, curves


def load_converted(conv_h5: Path, coco_path: Path, file_name_prefix: str):
    """Load decimated h5 and matching COCO annotations."""
    with h5py.File(conv_h5, "r") as f:
        data = f["data"][:]
        dt_s = float(f["data"].attrs.get("dt_s", 0.002))

    coco = json.loads(coco_path.read_text())

    # Find image entry matching this file
    img_entry = None
    for img in coco["images"]:
        if file_name_prefix in img["file_name"]:
            img_entry = img
            break

    anns = []
    if img_entry:
        img_id = img_entry["id"]
        anns = [a for a in coco["annotations"] if a["image_id"] == img_id]

    return data, dt_s, img_entry, anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-h5", required=True, help="Original raw h5")
    ap.add_argument("--src-json", required=True, help="Original curve JSON")
    ap.add_argument("--conv-h5", required=True, help="Converted (decimated) h5")
    ap.add_argument("--coco", required=True, help="COCO JSON (train or val)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--file-prefix", default=None,
                    help="file_name prefix to match in COCO (auto-detected if omitted)")
    args = ap.parse_args()

    src_h5 = Path(args.src_h5)
    src_json = Path(args.src_json)
    conv_h5 = Path(args.conv_h5)
    coco_path = Path(args.coco)

    file_prefix = args.file_prefix or src_h5.stem

    # Load raw
    raw_data, curves = load_raw(src_h5, src_json)
    print(f"Raw: shape={raw_data.shape} dtype={raw_data.dtype}")
    print(f"JSON shape field: {curves.get('shape', 'N/A')}")
    curve_keys = [k for k in curves if k.startswith("curve")]
    print(f"Curves: {len(curve_keys)}")

    # Load converted
    conv_data, dt_s, img_entry, anns = load_converted(conv_h5, coco_path, file_prefix)
    print(f"Converted: shape={conv_data.shape} dtype={conv_data.dtype} dt_s={dt_s}")
    print(f"COCO image: {img_entry}")
    print(f"COCO annotations: {len(anns)}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Panel 1: Raw waveform + JSON curves
    ax = axes[0]
    # raw_data shape: (nt, nch) or (nch, nt)
    if raw_data.shape[0] > raw_data.shape[1]:
        # (nt, nch) — typical raw format
        plot_raw = raw_data.T.astype(np.float32)  # -> (nch, nt)
    else:
        plot_raw = raw_data.astype(np.float32)

    nch, nt = plot_raw.shape
    # Normalize for display
    vmax = np.percentile(np.abs(plot_raw), 99)
    ax.imshow(plot_raw, aspect="auto", origin="lower", cmap="seismic",
              vmin=-vmax, vmax=vmax)
    ax.set_title(f"Raw decimated ({nt}×{nch})")
    ax.set_xlabel("Time samples (decimated)")
    ax.set_ylabel("Channel")

    # Overlay JSON curves — convert y (time-window index) to decimated time
    # czech_das_to_dasnet uses: t_dec = y * ANN_Y_TO_T_DEC where ANN_Y_TO_T_DEC = 2048/40 = 51.2
    ANN_Y_TO_T_DEC = 2048.0 / DECIMATE
    for ck in curve_keys:
        cx = np.array(curves[ck]["x"])  # channel indices
        cy = np.array(curves[ck]["y"])  # time-window indices
        cy_dec = cy * ANN_Y_TO_T_DEC   # -> decimated time samples
        # We plot (x-axis=time, y-axis=channel)
        ax.plot(cy_dec, cx, "g-", linewidth=1.5, alpha=0.8)
    ax.set_xlim(0, nt)
    ax.set_ylim(0, nch)

    # Panel 2: Decimated waveform + COCO bbox/segmentation
    ax = axes[1]
    # conv_data shape: (nch, nt_dec)
    nch_c, nt_c = conv_data.shape
    vmax_c = np.percentile(np.abs(conv_data), 99) + 1e-9
    ax.imshow(conv_data, aspect="auto", origin="lower", cmap="seismic",
              vmin=-vmax_c, vmax=vmax_c)
    ax.set_title(f"Decimated ({nt_c}×{nch_c}) + COCO")
    ax.set_xlabel("Time samples (decimated)")
    ax.set_ylabel("Channel")

    for a in anns:
        # bbox: [x_time, y_channel, w_time, h_channel]
        x, y, w, h = a["bbox"]
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        # segmentation polygon
        if "segmentation" in a and a["segmentation"]:
            poly = np.array(a["segmentation"][0]).reshape(-1, 2)
            # poly columns: (time, channel)
            ax.plot(poly[:, 0], poly[:, 1], "g-", linewidth=1, alpha=0.7)

        # attention_mask rectangle
        if "attention_mask" in a and a["attention_mask"]:
            for am_rect in a["attention_mask"]:
                ax2 = patches.Rectangle((am_rect[0], am_rect[1]),
                                        am_rect[2], am_rect[3],
                                        linewidth=1, edgecolor="yellow",
                                        facecolor="none", linestyle="--")
                ax.add_patch(ax2)

    ax.set_xlim(0, nt_c)
    ax.set_ylim(0, nch_c)

    # Panel 3: What DASTrainDataset produces — mapped back to Panel 1/2 pixel coords
    ax = axes[2]
    try:
        from dasnet.data.das import DASTrainDataset
        import torch
        ds = DASTrainDataset(
            ann_path=str(coco_path),
            root_dir=str(conv_h5.parent.parent),
            resize_scale=0.5,
            synthetic_noise=False,
            enable_stack_event=False,
            enable_vflip=False,
            allowed_bins={},
            min_keep_bins={},
        )
        found_idx = None
        for idx in range(len(ds)):
            img_info = ds.coco.imgs[ds.ids[idx]]
            if file_prefix in img_info["file_name"]:
                found_idx = idx
                break

        if found_idx is not None:
            image, target = ds[found_idx]
            # image shape after das.py: (C=3, dim1=time, dim2=channel)
            # torchvision sees this as (C, H=time, W=channel)
            # Model box coords: x=channel, y=time
            ch0 = image[0].numpy()  # (time_r, channel_r)
            vstd = np.std(ch0) * 2 + 1e-12
            # To plot with same axes as Panel 1/2 (x=time, y=channel),
            # transpose so imshow rows=channel, cols=time
            ax.imshow(ch0.T, aspect="auto", origin="lower",
                      cmap="seismic", vmin=-vstd, vmax=vstd,
                      extent=(0, nt_c, 0, nch_c))

            # Draw GT boxes — model coords (x=ch, y=time), map to plot (x=time, y=ch)
            scale = 1.0 / 0.5  # resize_scale=0.5
            if "boxes" in target:
                for b in target["boxes"].numpy():
                    x1_ch, y1_t, x2_ch, y2_t = b * scale
                    # plot x=time, y=channel → swap
                    rect = patches.Rectangle((y1_t, x1_ch), y2_t - y1_t, x2_ch - x1_ch,
                                             linewidth=1.5, edgecolor="lime",
                                             facecolor="none")
                    ax.add_patch(rect)

            # Draw GT masks overlay — mask shape (N, time_r, channel_r)
            if "masks" in target:
                masks = target["masks"].numpy()  # (N, time_r, channel_r)
                combined = np.max(masks, axis=0).astype(np.float32)
                rgba = np.zeros((*combined.T.shape, 4), dtype=np.float32)
                rgba[..., 1] = 1.0
                rgba[..., 3] = combined.T * 0.4
                ax.imshow(rgba, aspect="auto", origin="lower",
                          extent=(0, nt_c, 0, nch_c))

            ax.set_xlim(0, nt_c)
            ax.set_ylim(0, nch_c)
            ax.set_title(f"DASTrainDataset (ch0, mapped to orig coords)")
            ax.set_xlabel("Time samples (decimated)")
            ax.set_ylabel("Channel")
        else:
            ax.set_title("DASTrainDataset: file not found in COCO")
    except Exception as e:
        ax.set_title(f"DASTrainDataset failed: {e}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
