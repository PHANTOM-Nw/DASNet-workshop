#!/usr/bin/env python3
"""Convert the Czech DAS-dataset to a DASNet-ingestible format.

Reads:
    <src_root>/<label>/<stem>.h5    (Acquisition/Raw[0]/RawData, int16, (nt, nch))
    <src_root>/<label>/<stem>.json  (polyline curves, x=channel, y=time-window)

Writes:
    <out_root>/<label>/<stem>.h5    (/data, float32, (nch, nt_dec), attr dt_s=2e-3)
    <coco_out>                       (COCO JSON with train/val split)

Output convention: z:/{origin_dataset_name}_dasnet/{data,annotations}/
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.signal import resample_poly

# ---- locked constants ------------------------------------------------------
RAW_DSET_PATH = "Acquisition/Raw[0]/RawData"
RAW_PULSE_RATE = 20000.0
ANN_SHIFT = 2048
DECIMATE = 40
CH_BLOCK = 64
OUT_DSET_NAME = "data"

DT_S_OUT = 1.0 / (RAW_PULSE_RATE / DECIMATE)           # 2e-3
ANN_Y_TO_T_DEC = ANN_SHIFT / DECIMATE                  # 51.2

CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "walk"},
    {"id": 3, "name": "running"},
    {"id": 4, "name": "longboard"},
    {"id": 5, "name": "fence"},
    {"id": 6, "name": "construction"},
    {"id": 7, "name": "manipulation"},
    {"id": 8, "name": "openclose"},
    {"id": 9, "name": "regular"},
]
LABEL_TO_CATID = {c["name"]: c["id"] for c in CATEGORIES}


def iter_source_triples(src_root: Path) -> Iterable[tuple[str, Path]]:
    """Yield (label, h5_path) pairs for every labelled .h5 under src_root."""
    for label_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        label = label_dir.name
        if label not in LABEL_TO_CATID:
            continue
        for h5_path in sorted(label_dir.glob("*.h5")):
            if not h5_path.with_suffix(".json").exists():
                continue
            yield label, h5_path


def decimate_and_rewrite_h5(src_h5: Path, dst_h5: Path) -> tuple[int, int]:
    """Read (nt, nch) raw int16 waveform, decimate time by DECIMATE, write (nch, nt_dec) float32.

    Processes CH_BLOCK channels at a time to cap peak memory. Returns (nch, nt_dec).
    """
    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_h5, "r") as f_src:
        dset_src = f_src[RAW_DSET_PATH]
        nt_raw, nch = dset_src.shape
        nt_dec = -(-nt_raw // DECIMATE)

        with h5py.File(dst_h5, "w") as f_dst:
            dset_dst = f_dst.create_dataset(
                OUT_DSET_NAME,
                shape=(nch, nt_dec),
                dtype=np.float32,
                chunks=(min(64, nch), min(4096, nt_dec)),
                compression="gzip",
                compression_opts=1,
            )
            dset_dst.attrs["dt_s"] = float(DT_S_OUT)
            dset_dst.attrs["source_file"] = str(src_h5)
            dset_dst.attrs["decimate_factor"] = int(DECIMATE)
            dset_dst.attrs["raw_pulse_rate"] = float(RAW_PULSE_RATE)

            for ch0 in range(0, nch, CH_BLOCK):
                ch1 = min(nch, ch0 + CH_BLOCK)
                block = dset_src[:, ch0:ch1].astype(np.float32, copy=False)
                decimated = resample_poly(block, up=1, down=DECIMATE, axis=0)
                if decimated.shape[0] != nt_dec:
                    fixed = np.zeros((nt_dec, decimated.shape[1]), dtype=np.float32)
                    k = min(nt_dec, decimated.shape[0])
                    fixed[:k] = decimated[:k]
                    decimated = fixed
                dset_dst[ch0:ch1, :] = decimated.T.astype(np.float32)

    return nch, nt_dec


def curves_json_to_coco_anns(
    json_path: Path,
    image_id: int,
    category_id: int,
    start_ann_id: int,
) -> tuple[list[dict], int]:
    """Convert one curves JSON into a list of COCO annotation dicts.

    Czech JSON convention (verified): curve.x = channel index (sub-pixel float),
    curve.y = time-window index (sub-pixel float, 0..nt/ANN_SHIFT). We convert
    time-window to decimated-sample index via ANN_Y_TO_T_DEC, then emit:

    * segmentation = [[t0, ch0, t1, ch1, ...]] (DASNet seg[i]=time, seg[i+1]=channel)
    * bbox         = [t_min, ch_min, t_max - t_min, ch_max - ch_min]
    * area         = bbox_w * bbox_h

    Curves with fewer than 5 points are dropped (DASNet drops len(seg) < 10).
    """
    j = json.loads(json_path.read_text())
    n_tw, n_ch = j["shape"]

    anns: list[dict] = []
    ann_id = start_ann_id

    for key in sorted(j.keys()):
        if not key.startswith("curve"):
            continue
        cx = np.asarray(j[key]["x"], dtype=np.float64)  # channel (float, sub-pixel)
        cy = np.asarray(j[key]["y"], dtype=np.float64)  # time-window (float)
        if len(cx) < 5 or len(cx) != len(cy):
            continue

        ch = np.clip(cx, 0, n_ch - 1)
        t_dec = np.clip(cy * ANN_Y_TO_T_DEC, 0, (n_tw - 1) * ANN_Y_TO_T_DEC)

        poly: list[float] = []
        for t_val, ch_val in zip(t_dec.tolist(), ch.tolist()):
            poly.append(float(t_val))
            poly.append(float(ch_val))

        t_min, t_max = float(t_dec.min()), float(t_dec.max())
        ch_min, ch_max = float(ch.min()), float(ch.max())
        bw = max(1.0, t_max - t_min)
        bh = max(1.0, ch_max - ch_min)

        anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(category_id),
            "iscrowd": 0,
            "bbox": [t_min, ch_min, bw, bh],
            "area": float(bw * bh),
            "segmentation": [poly],
        })
        ann_id += 1

    return anns, ann_id
