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
