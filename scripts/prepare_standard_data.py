#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp


# =========================================================
# Time parsing / formatting
# =========================================================

# Example:
# MBARI_F8_STU_GL20m_OCP5m_FS200Hz_2023-12-22T074710Z.h5
FNAME_TIME_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})Z\.h5$")


def parse_begin_time_from_filename(fname: str) -> datetime:
    """
    Parse begin_time from filename suffix: _YYYY-MM-DDTHHMMSSZ.h5  (UTC)
    Return timezone-aware datetime in UTC.
    """
    m = FNAME_TIME_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse begin_time from filename: {fname}")
    date_part, hh, mm, ss = m.group(1), m.group(2), m.group(3), m.group(4)
    dt = datetime.strptime(f"{date_part} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)


def format_time_like_example(dt_utc: datetime) -> str:
    """
    Format as: 2019-07-04T17:33:19.000000+00:00
    """
    dt_utc = dt_utc.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S") + ".000000+00:00"


# =========================================================
# Core processing
# =========================================================

def compute_strain_rate(strain: np.ndarray, dt: float) -> np.ndarray:
    """
    strain: (nt, nch)
    return strain_rate: (nt, nch) = d(strain)/dt
    """
    # diff along time axis
    sr = np.diff(strain, axis=0) / dt
    # pad first sample with zeros to keep same length
    sr = np.vstack([np.zeros((1, sr.shape[1]), dtype=sr.dtype), sr])
    return sr.astype(np.float32, copy=False)


def read_dt_from_rawdata_time(raw_data_time: np.ndarray) -> float:
    """
    raw_data_time: typically microseconds timestamps (int/float)
    Your original method: das_time = raw_data_time / 1e6; dt = das_time[1]-das_time[0]
    """
    das_time = raw_data_time.astype(np.float64) / 1e6
    if das_time.shape[0] < 2:
        raise ValueError("RawDataTime length < 2, cannot compute dt.")
    dt = float(das_time[1] - das_time[0])
    if dt <= 0:
        raise ValueError(f"Invalid dt computed from RawDataTime: {dt}")
    return dt


def process_one_row(row_dict: dict, channel_start: int, channel_end: int, save_dir: str) -> tuple[bool, str]:
    """
    Returns: (ok, message)
    """
    save_name = row_dict["File Name"]
    file_path = row_dict["File Path"]

    out_path = os.path.join(save_dir, save_name)
    if os.path.exists(out_path):
        return True, f"SKIP exists: {out_path}"

    # ---------- read input ----------
    try:
        with h5py.File(file_path, "r") as f:
            raw_data = f["Acquisition/Raw[0]/RawData"][:]          # (nt, n_total_ch)
            raw_data_time = f["Acquisition/Raw[0]/RawDataTime"][:] # (nt,)
    except Exception as e:
        return False, f"READ FAIL: {file_path} | {e}"

    # ---------- slice channels ----------
    # raw_data: (nt, n_total_ch) -> strain: (nt, nch)
    strain = raw_data[:, channel_start:channel_end]
    if strain.size == 0:
        return False, f"EMPTY strain after slicing [{channel_start}:{channel_end}] for {file_path}"

    # ---------- dt ----------
    try:
        dt = read_dt_from_rawdata_time(raw_data_time)
    except Exception as e:
        return False, f"DT FAIL: {file_path} | {e}"

    # ---------- compute strain_rate (microstrain/s) ----------
    strain_rate = compute_strain_rate(strain, dt)  # (nt, nch)

    # ---------- target format: data = (nch, nt) ----------
    data = np.ascontiguousarray(strain_rate.T, dtype=np.float32)  # (nch, nt)
    nch, nt = data.shape

    # ---------- attrs ----------
    try:
        begin_dt = parse_begin_time_from_filename(save_name)
    except Exception as e:
        return False, f"FNAME TIME FAIL: {save_name} | {e}"

    end_dt = begin_dt + timedelta(seconds=(nt - 1) * dt)

    attrs = {
        "event_id": os.path.splitext(save_name)[0],
        "event_time": "",                 # leave blank
        "event_time_index": int(-1),      # leave blank
        "begin_time": format_time_like_example(begin_dt),
        "end_time": format_time_like_example(end_dt),
        "dt_s": float(dt),
        "dx_m": float(5.2),
        "unit": "microstrain/s",
        "source": "monterey bay",
    }

    # ---------- write output ----------
    try:
        os.makedirs(save_dir, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            dset = f.create_dataset("data", data=data, dtype="float32")
            for k, v in attrs.items():
                dset.attrs[k] = v
    except Exception as e:
        return False, f"WRITE FAIL: {out_path} | {e}"

    # ---------- print attrs (per file) ----------
    # Note: parallel processes -> print order interleaves
    msg = (
        f"\n[OK] {save_name}\n"
        f"  data shape (nch, nt) = {data.shape}, dtype={data.dtype}\n"
        f"  attrs = {attrs}\n"
    )
    return True, msg


def prepare_data_parallel(
    df: pd.DataFrame,
    channel_start: int,
    channel_end: int,
    save_dir: str,
    max_workers: int = 8,
    chunksize: int = 1,
):
    rows = df.to_dict(orient="records")
    total = len(rows)
    if total == 0:
        print("No rows to process.")
        return

    # ProcessPool for CPU-bound + avoid GIL (also good for numpy ops)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one_row, row, channel_start, channel_end, save_dir)
            for row in rows
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preparing"):
            ok, msg = fut.result()
            if ok:
                print(msg)
            else:
                logging.error(msg)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # Recommended on Linux/HPC when using h5py in multiprocess
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s: %(message)s"
    )

    detect_name_path = "/home/chun/dasnet_workshop/DASNet-workshop/scripts/total_list.csv"
    save_dir = "/nfs2/group/chun/standard_data/monterey_bay_data"  # <- change if needed

    channel_start = 7400
    channel_end = 10245

    df = pd.read_csv(detect_name_path)
    # ensure required columns exist
    for col in ["File Name", "File Path"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {detect_name_path}. Columns={list(df.columns)}")

    print(f"Start preparing data... rows={len(df)}")
    prepare_data_parallel(
        df=df,
        channel_start=channel_start,
        channel_end=channel_end,
        save_dir=save_dir,
        max_workers=4,   # <- adjust
    )
    print("All done!")