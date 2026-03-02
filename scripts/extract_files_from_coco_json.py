#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd


def extract_unique_filenames(coco_json_path, output_csv_path):
    # ---------- 读取 COCO ----------
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    if "images" not in coco:
        raise KeyError("No 'images' field found in COCO file.")

    # ---------- 提取 file_name ----------
    file_names = [img["file_name"] for img in coco["images"]]

    # ---------- 转 DataFrame ----------
    df = pd.DataFrame({"file_name": file_names})

    # ---------- 修改文件名 ----------
    # 把 ".h5_0.jpg" → ".h5"
    df["file_name"] = df["file_name"].str.replace(".h5_0.jpg", ".h5", regex=False)

    # ---------- 去重 ----------
    df = df.drop_duplicates(subset=["file_name"]).reset_index(drop=True)

    # ---------- 保存 ----------
    df.to_csv(output_csv_path, index=False)

    print(f"Total unique files: {len(df)}")
    print(f"Saved to: {output_csv_path}")


if __name__ == "__main__":
    coco_json_path = "/nfs2/group/chun/label_iter3re/val_data_iter3re.json"   # ← 改成你的 COCO 文件路径
    output_csv_path = "val_list.csv"   # 输出路径

    extract_unique_filenames(coco_json_path, output_csv_path)