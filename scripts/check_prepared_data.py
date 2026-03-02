#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt


def zscore_clip(data, clip_value=3.0):
    mean = np.mean(data)
    std = np.std(data) + 1e-6
    z = (data - mean) / std
    return np.clip(z, -clip_value, clip_value)


def random_plot_h5(folder_path):
    # -------- 找到所有 h5 文件 --------
    h5_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".h5")
    ]

    if len(h5_files) == 0:
        raise RuntimeError(f"No .h5 files found in {folder_path}")

    file_path = random.choice(h5_files)
    print(f"\nRandomly selected file:\n{file_path}")

    # -------- 读取数据 --------
    with h5py.File(file_path, "r") as f:
        data = f["data"][:]   # (nch, nt)
        attrs = dict(f["data"].attrs)

    print("\n=== ATTRIBUTES ===")
    for k, v in attrs.items():
        print(f"{k}: {v}")

    print(f"\nOriginal Data:")
    print(f"  shape: {data.shape}")
    print(f"  min={data.min():.3f}, max={data.max():.3f}, mean={data.mean():.3f}")

    # -------- Z-score + clip --------
    data_norm = zscore_clip(data, clip_value=3.0)

    print("\nAfter Z-score + clip:")
    print(f"  min={data_norm.min():.3f}, max={data_norm.max():.3f}")
    print(f"  mean={data_norm.mean():.3f}, std={data_norm.std():.3f}")

    # -------- 绘图 --------
    plt.figure(figsize=(12, 6))
    im = plt.imshow(
        data_norm,
        aspect="auto",
        cmap="seismic",
        origin="lower",
        vmin=-3,
        vmax=3
    )

    plt.colorbar(im, label="Z-score (clipped)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Channel")

    title_str = (
        f"{os.path.basename(file_path)}\n"
        f"dt={attrs.get('dt_s')} s | "
        f"dx={attrs.get('dx_m')} m | "
        f"begin={attrs.get('begin_time')} | "
        f"source={attrs.get('source')}"
    )
    plt.title(title_str)

    plt.tight_layout()
    plt.savefig('check.jpg')


if __name__ == "__main__":
    folder = "/nfs2/group/chun/standard_data/monterey_bay_data"  # 改成你的路径
    random_plot_h5(folder)