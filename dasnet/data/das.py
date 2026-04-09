#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ============================================================================
# 本文件是 DASNet 项目的数据处理核心：
#   1. 从 HDF5 读取 DAS（分布式光纤声波传感）原始数据
#   2. 做带通/高通滤波，拼成 3 通道 "RGB" 张量给 CNN 用
#   3. 训练集：支持 COCO 标注、事件叠加、噪声叠加、高斯线掩膜、翻转增广等
#   4. 推理集：只做预处理和 resize
# 约定：
#   - 原始 raw 形状 (nch, nt)：nch=通道数(空间)，nt=时间采样点数
#   - RGB 形状 (H=通道, W=时间, 3)
#   - 送入模型时 permute 成 (C, W, H)，即 (3, 时间, 通道)
# ============================================================================

import os
import random
from typing import Optional
import h5py                                  # 读 HDF5 科学数据
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO             # 解析 COCO 格式标注
from torchvision.transforms import functional as F  # 图像 resize/flip
from scipy.signal import butter, sosfiltfilt # 巴特沃斯滤波器
import fsspec                                 # 统一文件系统访问（本地/GCS）


# =========================================================
# 基础工具函数
# =========================================================

def normalize(data: np.ndarray) -> np.ndarray:
    """Z-score 归一化 + 裁剪到 [-3, 3]，默认对 (N, T) 数组沿 axis=0 做标准化。"""
    # 逐列减均值除标准差；加 1e-6 防止除零
    data = (data - np.mean(data, axis=0, keepdims=True)) / (np.std(data, axis=0, keepdims=True) + 1e-6)
    # 裁掉极端值，抑制脉冲噪声对训练的影响
    return np.clip(data, -3, 3)


def interpolate_line_segments_int(points):
    """把折线的控制点按整数像素栅格化（Bresenham 风格）。"""
    line_points = []
    for i in range(len(points) - 1):                  # 遍历每一段
        start = points[i]
        end = points[i + 1]

        dx = end[0] - start[0]                        # x 方向增量
        dy = end[1] - start[1]                        # y 方向增量
        max_delta = max(abs(dx), abs(dy))             # 取较大步数，保证每个整数格都被访问
        if max_delta == 0:
            return points                             # 两端点重合：直接返回原 points

        step_x = dx / max_delta                       # 每一步在 x 上推进多少
        step_y = dy / max_delta

        for j in range(max_delta + 1):                # 线性插值出 max_delta+1 个整数点
            line_points.append((int(start[0] + step_x * j), int(start[1] + step_y * j)))

    unique_points = list(set(line_points))            # 去重（相邻段接头处会重复）
    unique_points.sort(key=lambda x: (x[0], x[1]))    # 排序便于后续顺序访问
    return unique_points


def gaussian_line(x_center, std_dev, amplitude, length):
    """生成一维高斯曲线：中心在 x_center，长度 length。用于把标注曲线“晕染”成高斯掩膜列。"""
    x = np.arange(length)
    gaussian = amplitude * np.exp(-(x - x_center) ** 2 / (2 * std_dev ** 2))
    return gaussian


def all_instances_in_allowed_bins(annotations, allowed_bins):
    """
    判断本样本的所有目标 (instance) 的 signal_index.bin 是否都落在 allowed_bins 允许的范围内。
    只要有一个不合格就返回 False —— 用来决定该样本是否“干净到可以做叠加增广”。
    """
    if not allowed_bins:                              # 没配置就当通过
        return True
    for ann in annotations:
        cat = ann.get("category_id")                  # 类别 id
        bin_idx = ann.get("signal_index", {}).get("bin")  # 信号强度分箱索引

        if cat not in allowed_bins:                   # 类别未被允许 -> 拒绝
            return False
        if bin_idx is None:                           # 缺少 bin 信息 -> 拒绝
            return False

        lo, hi = allowed_bins[cat]                    # 该类别允许的 bin 区间
        if not (lo <= bin_idx <= hi):
            return False

    return True


def filter_instances_by_min_bin(annotations, min_keep_bins):
    """
    删除“弱信号”实例：
    - 若该类别在 min_keep_bins 中且 bin < 阈值 -> 删除
    - 若该类别在 min_keep_bins 中但缺少 bin  -> 删除
    - 其余类别原样保留
    """
    if not min_keep_bins:
        return annotations

    kept = []
    for ann in annotations:
        cat = ann.get("category_id")
        if cat is None:
            continue                                  # 没类别，跳过

        if cat not in min_keep_bins:
            kept.append(ann)                          # 不在过滤规则中 -> 保留
            continue

        min_bin = min_keep_bins[cat]                  # 该类别最低需要的 bin
        bin_idx = ann.get("signal_index", {}).get("bin", None)
        if bin_idx is None:
            continue                                  # 缺 bin -> 丢弃
        if bin_idx >= min_bin:
            kept.append(ann)                          # 强度够 -> 保留

    return kept


# =========================================================
# 预处理：滤波器设计 & RGB 生成
# =========================================================

def _safe_design_sos_bandpass(dt: float, f1: float, f2: float, order: int = 4):
    """设计带通 Butterworth SOS 滤波器，做边界保护避免归一化频率越界。"""
    nyq = 0.5 / dt                                    # Nyquist 频率
    wn1 = f1 / nyq                                    # 归一化下截
    wn2 = f2 / nyq                                    # 归一化上截
    # 防止出现 <=0 或 >=1 的非法值
    wn1 = max(1e-6, min(wn1, 0.999999))
    wn2 = max(1e-6, min(wn2, 0.999999))
    if wn1 >= wn2:
        # 极端情况（采样率太低），给一个极小的通带
        wn1 = max(1e-6, min(0.49, wn2 * 0.5))
    return butter(order, [wn1, wn2], btype="bandpass", output="sos")


def _safe_design_sos_highpass(dt: float, f: float, order: int = 4):
    """设计高通 Butterworth SOS 滤波器。"""
    nyq = 0.5 / dt
    wn = f / nyq
    wn = max(1e-6, min(wn, 0.999999))
    return butter(order, wn, btype="highpass", output="sos")


def preprocess_data_rgb(
    h5_path: str,
    channel_range=None,                 # (ch_start, ch_end)：对通道(axis=0)做截取
    data_key: str = "data",             # HDF5 内数据集名
    data_is_strain_rate: bool = True,   # 输入是否已经是应变率
    f_band=(2.0, 10.0),                 # 带通频段（Hz）
    f_high=10.0,                        # 高通截止（Hz）
    storage_backend: str = "local",     # "local" 或 "gcs"
    gcs_key_path: Optional[str] = None, # GCS 认证密钥路径
):
    """
    读取 HDF5（本地或 GCS），输出 3 通道 "RGB" 数组：
      - ch0 = 原始应变率（归一化）
      - ch1 = 带通滤波后
      - ch2 = 高通滤波后
    形状约定：raw (nch, nt) -> rgb (H=nch, W=nt, 3)
    """
    # --------------------------------------------------
    # 1. 从本地文件系统或 GCS 打开 HDF5
    # --------------------------------------------------
    if storage_backend == "gcs" or (isinstance(h5_path, str) and h5_path.startswith("gs://")):
        # 走 GCS：用 fsspec 打开字节流，再让 h5py 解析
        fs = fsspec.filesystem("gcs", token=gcs_key_path)
        with fs.open(h5_path, "rb") as fp:
            with h5py.File(fp, "r") as f:
                if data_key not in f:
                    raise KeyError(f"'{data_key}' not found in {h5_path}. Keys={list(f.keys())}")
                dset = f[data_key]
                dt = float(dset.attrs.get("dt_s", None))  # 从 attrs 读采样间隔(秒)
                if dt is None:
                    raise KeyError(f"dt_s attribute not found in {h5_path}:{data_key}")

                raw = dset[:]                         # 全部读入内存，形状 (nch, nt)
                raw = raw.astype("float32")
    else:
        # 本地文件
        with h5py.File(h5_path, "r") as f:
            if data_key not in f:
                raise KeyError(f"'{data_key}' not found in {h5_path}. Keys={list(f.keys())}")
            dset = f[data_key]
            dt = float(dset.attrs.get("dt_s", None))
            if dt is None:
                raise KeyError(f"dt_s attribute not found in {h5_path}:{data_key}")

            raw = dset[:]                             # (nch, nt)
            raw = raw.astype("float32")

    # --------------------------------------------------
    # 2. 通道裁剪（可选：只取一段光纤通道）
    # --------------------------------------------------
    if channel_range is not None:
        ch0, ch1 = channel_range
        raw = raw[ch0:ch1, :]

    # --------------------------------------------------
    # 3. 统一成应变率 strain_rate
    # --------------------------------------------------
    if data_is_strain_rate:
        strain_rate = raw
    else:
        # 输入是应变 -> 沿时间轴差分近似求导
        sr = np.diff(raw, axis=1) / dt
        # diff 会少一列，在前面补 0 保持原长度
        sr = np.concatenate([np.zeros((sr.shape[0], 1), dtype=sr.dtype), sr], axis=1)
        strain_rate = sr

    # --------------------------------------------------
    # 4. 两种滤波：带通 + 高通（沿时间方向 axis=1，逐通道）
    # --------------------------------------------------
    sos_bp = _safe_design_sos_bandpass(dt, f_band[0], f_band[1], order=4)
    sos_hp = _safe_design_sos_highpass(dt, f_high, order=4)

    sr_bp = sosfiltfilt(sos_bp, strain_rate, axis=1)  # 零相位带通
    sr_hp = sosfiltfilt(sos_hp, strain_rate, axis=1)  # 零相位高通

    # --------------------------------------------------
    # 5. 归一化 + 组装 3 通道张量
    #    注意历史习惯：normalize 沿 axis=0，所以先 .T 再 .T 回来
    # --------------------------------------------------
    ch0 = normalize(strain_rate.T).T
    ch1 = normalize(sr_bp.T).T
    ch2 = normalize(sr_hp.T).T

    rgb = np.zeros((ch0.shape[0], ch0.shape[1], 3), dtype=np.float32)  # (H=通道, W=时间, 3)
    rgb[:, :, 0] = ch0                                # R = 原始应变率
    rgb[:, :, 1] = ch1                                # G = 带通
    rgb[:, :, 2] = ch2                                # B = 高通
    return rgb, dt


def preprocess_from_array(
    data_nch_nt: np.ndarray,
    dt_s: float,
    data_is_strain_rate: bool = True,
    f_band=(2.0, 10.0),
    f_high=10.0,
):
    """
    与 preprocess_data_rgb 相同的流水线，但直接从内存 numpy 数组开始（不读文件）。
    实时推理管线使用此函数。
    """
    raw = np.asarray(data_nch_nt, dtype=np.float32)

    # 统一成应变率（同上）
    if data_is_strain_rate:
        strain_rate = raw
    else:
        sr = np.diff(raw, axis=1) / dt_s
        sr = np.concatenate([np.zeros((sr.shape[0], 1), dtype=sr.dtype), sr], axis=1)
        strain_rate = sr

    # 滤波
    sos_bp = _safe_design_sos_bandpass(dt_s, f_band[0], f_band[1], order=4)
    sos_hp = _safe_design_sos_highpass(dt_s, f_high, order=4)

    sr_bp = sosfiltfilt(sos_bp, strain_rate, axis=1)
    sr_hp = sosfiltfilt(sos_hp, strain_rate, axis=1)

    # 归一化 + 打包
    ch0 = normalize(strain_rate.T).T
    ch1 = normalize(sr_bp.T).T
    ch2 = normalize(sr_hp.T).T

    rgb = np.zeros((ch0.shape[0], ch0.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = ch0
    rgb[:, :, 1] = ch1
    rgb[:, :, 2] = ch2
    return rgb, dt_s


# =========================================================
# 训练集 Dataset（完整功能）
# =========================================================

class DASTrainDataset(Dataset):
    """
    训练数据集，支持：
    - COCO 格式标注
    - 弱信号实例过滤 (min_keep_bins)
    - 事件叠加 / 噪声叠加（需要样本足够“干净”，由 allowed_bins 控制）
    - 高斯线掩膜生成（把折线标注晕染为概率图）
    - attention_mask 生成
    - 垂直翻转增广
    - 按固定比例缩放（而不是固定尺寸）
    - 中间表示为 RGB，送入模型前 permute 为 (C, W, H)
    """

    def __init__(
        self,
        ann_path: str,                    # COCO 标注 json 路径
        root_dir: str,                    # H5 原始数据根目录
        resize_scale: float = 0.5,        # 缩放比例（时间、通道同比例）

        # 弱信号过滤
        min_keep_bins=None,

        # 叠加增广的准入门槛
        allowed_bins=None,

        # 噪声叠加
        synthetic_noise: bool = True,
        noise_csv: str = "/work/zhu-stor1/group/chun/standard_data/monterey_bay_noise/noise_list_lambda3.csv",
        syn_prob: float = 0.5,            # 触发噪声叠加的概率
        syn_factor_range=(0.5, 1.5),      # 噪声幅度系数区间

        # 事件叠加
        enable_stack_event: bool = True,
        event_stack_prob: float = 0.3,    # 触发事件叠加的概率
        event_alpha_range=(0.5, 1.5),     # 伙伴样本权重

        # 预处理
        data_key: str = "data",
        data_is_strain_rate: bool = True,
        channel_range=None,
        f_band=(2.0, 10.0),
        f_high=10.0,

        # 增广
        enable_vflip: bool = True,
        vflip_prob: float = 0.2,

        # 高斯线掩膜参数
        gaussian_std_dev_x: float = 50.0, # 沿时间方向的高斯标准差（像素）
        gaussian_amplitude: float = 1.0,
    ):
        self.coco = COCO(ann_path)                    # 载入 COCO 标注
        self.root_dir = root_dir
        self.ids = list(self.coco.imgs.keys())        # 所有图像 id

        self.resize_scale = float(resize_scale)

        self.min_keep_bins = min_keep_bins or {}
        self.allowed_bins = allowed_bins or {}

        self.synthetic_noise = synthetic_noise
        self.noise_csv = noise_csv
        self.syn_prob = syn_prob
        self.syn_factor_range = syn_factor_range
        # 噪声库索引表：记录可用的“纯噪声”H5 文件路径
        self.noise_df = pd.read_csv(self.noise_csv) if synthetic_noise else None

        self.enable_stack_event = enable_stack_event
        self.event_stack_prob = event_stack_prob
        self.event_alpha_range = event_alpha_range

        self.data_key = data_key
        self.data_is_strain_rate = data_is_strain_rate
        self.channel_range = channel_range
        self.f_band = f_band
        self.f_high = f_high

        self.enable_vflip = enable_vflip
        self.vflip_prob = vflip_prob

        self.gaussian_std_dev_x = gaussian_std_dev_x
        self.gaussian_amplitude = gaussian_amplitude

    def __len__(self):
        return len(self.ids)

    # -----------------------------
    # COCO image id -> 真正的 H5 文件路径
    # -----------------------------
    def _imgid_to_h5_path(self, img_id: int) -> str:
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        # 历史原因：标注里写的是 *_0.jpg，实际要去掉后缀拼 H5 路径
        rel = file_name.replace("_0.jpg", "")
        return os.path.join(self.root_dir, rel)

    def _load_rgb_from_imgid(self, img_id: int) -> np.ndarray:
        """按 image id 加载并预处理成 RGB。"""
        h5_path = self._imgid_to_h5_path(img_id)
        rgb, _dt = preprocess_data_rgb(
            h5_path,
            channel_range=self.channel_range,
            data_key=self.data_key,
            data_is_strain_rate=self.data_is_strain_rate,
            f_band=self.f_band,
            f_high=self.f_high,
        )
        return rgb                                    # (H, W, 3)

    # -----------------------------
    # 事件叠加时：随机挑选一个合格的伙伴样本
    # -----------------------------
    def _select_partner(self, cur_index):
        N = len(self.ids)
        for _ in range(20):                           # 最多尝试 20 次
            j = random.randint(0, N - 1)
            if j == cur_index:                        # 跳过自己
                continue
            pid = self.ids[j]
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=pid))
            # 只有伙伴样本也“干净”才允许拿来叠
            if all_instances_in_allowed_bins(anns, self.allowed_bins):
                return j
        return None                                   # 20 次都没找到就放弃

    def _maybe_stack_event(self, rgb_base, anns_base, index):
        """按概率把另一个事件叠加到当前样本上，同时合并两者的标注。"""
        if not self.enable_stack_event:
            return rgb_base, anns_base, False

        # 当前样本必须足够“干净”才能参与叠加
        if not all_instances_in_allowed_bins(anns_base, self.allowed_bins):
            return rgb_base, anns_base, False

        if random.random() >= self.event_stack_prob:  # 概率门槛
            return rgb_base, anns_base, False

        partner_idx = self._select_partner(index)
        if partner_idx is None:
            return rgb_base, anns_base, False

        partner_id = self.ids[partner_idx]
        anns_partner = self.coco.loadAnns(self.coco.getAnnIds(imgIds=partner_id))
        rgb_partner = self._load_rgb_from_imgid(partner_id)

        # 两个样本尺寸可能不一致，裁到共同最小尺寸
        H = min(rgb_base.shape[0], rgb_partner.shape[0])
        W = min(rgb_base.shape[1], rgb_partner.shape[1])
        rgb_base = rgb_base[:H, :W]
        rgb_partner = rgb_partner[:H, :W]

        alpha = random.uniform(*self.event_alpha_range)   # 伙伴样本权重
        rgb_mix = rgb_base + alpha * rgb_partner          # 线性叠加

        # 叠加后每个通道重新归一化，维持数值量级稳定
        for c in range(3):
            rgb_mix[:, :, c] = normalize(rgb_mix[:, :, c])

        anns_new = anns_base + anns_partner               # 合并标注
        return rgb_mix, anns_new, True

    def _maybe_add_noise(self, rgb_input, annotations):
        """
        按概率叠加一段背景噪声。
        条件：
        - 开启 synthetic_noise
        - 当前样本满足 allowed_bins
        - 随机数通过 syn_prob
        - 噪声样本形状完全匹配（避免裁剪带来的标签错位）
        """
        if not self.synthetic_noise:
            return rgb_input

        if not all_instances_in_allowed_bins(annotations, self.allowed_bins):
            return rgb_input

        if np.random.rand() >= self.syn_prob:
            return rgb_input

        # 从噪声库随机挑一个 H5
        idx = random.randint(0, len(self.noise_df) - 1)
        selected_path = self.noise_df.iloc[idx]["File Path"]

        noise_rgb, _dt = preprocess_data_rgb(
            selected_path,
            channel_range=self.channel_range,
            data_key=self.data_key,
            data_is_strain_rate=self.data_is_strain_rate,
            f_band=self.f_band,
            f_high=self.f_high,
        )

        if noise_rgb.shape != rgb_input.shape:
            # 形状不匹配就放弃这次叠加（严格策略）
            return rgb_input

        syn_factor = random.uniform(*self.syn_factor_range)  # 随机噪声强度
        rgb_aug = rgb_input + noise_rgb * syn_factor

        # 叠加后重归一化
        for c in range(3):
            rgb_aug[:, :, c] = normalize(rgb_aug[:, :, c])

        return rgb_aug

    # -----------------------------
    # attention_mask 生成（把原图坐标系的矩形映射到 resize 后的网格）
    # -----------------------------
    def _generate_attention_mask(self, attention_mask_rects, mask_shape, original_shape):
        """
        attention_mask_rects: [[x(time), y(channel), w(time), h(channel)], ...]
        mask_shape: (new_t, new_ch) resize 之后
        original_shape: (orig_t, orig_ch) resize 之前
        """
        attention_mask = np.zeros(mask_shape, dtype=np.uint8)

        orig_t, orig_ch = original_shape
        new_t, new_ch = mask_shape
        scale_t = new_t / max(orig_t, 1)              # 时间方向缩放系数
        scale_ch = new_ch / max(orig_ch, 1)           # 通道方向缩放系数

        for rect in attention_mask_rects:
            x_min_t, y_min_ch, width_t, height_ch = rect

            # 原图坐标 -> resize 后坐标
            t0 = int(x_min_t * scale_t)
            ch0 = int(y_min_ch * scale_ch)
            t1 = int((x_min_t + width_t) * scale_t)
            ch1 = int((y_min_ch + height_ch) * scale_ch)

            # 边界保护
            t0 = max(0, t0)
            ch0 = max(0, ch0)
            t1 = min(new_t, t1)
            ch1 = min(new_ch, ch1)

            if t1 > t0 and ch1 > ch0:
                attention_mask[t0:t1, ch0:ch1] = 1    # 矩形区域置 1

        return attention_mask

    # -----------------------------
    # 核心 __getitem__
    # -----------------------------
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # ===== 第一步：弱信号过滤 =====
        annotations = filter_instances_by_min_bin(annotations, self.min_keep_bins)

        # 如果过滤后一个实例都不剩，就随机换一个样本（递归）
        if len(annotations) == 0:
            return self.__getitem__(random.randint(0, len(self.ids) - 1))

        # ===== 第二步：读 raw -> RGB =====
        rgb_input = self._load_rgb_from_imgid(img_id)     # (H=通道, W=时间, 3)

        # 事件叠加
        rgb_input, annotations, did_stack = self._maybe_stack_event(rgb_input, annotations, index)

        # 噪声叠加（和事件叠加互斥：一次只做一种）
        if not did_stack:
            rgb_input = self._maybe_add_noise(rgb_input, annotations)

        # ===== 第三步：转成 torch 张量 =====
        # rgb_input: (H=通道, W=时间, 3)
        # permute(2,1,0) -> (3, 时间, 通道) == (C, W, H)
        image = torch.from_numpy(rgb_input).permute(2, 1, 0)

        # image.shape = (C, W, H)
        orig_W = image.shape[1]                           # 时间维度长度
        orig_H = image.shape[2]                           # 通道维度长度
        original_size_for_labels = (orig_W, orig_H)       # 供 attention_mask 映射使用

        # ===== 第四步：按固定比例缩放 =====
        new_W = max(1, int(orig_W * self.resize_scale))   # 新的时间维度
        new_H = max(1, int(orig_H * self.resize_scale))   # 新的通道维度
        image = F.resize(image, (new_W, new_H))

        # 坐标系说明：
        # COCO 标注中 x=时间, y=通道
        # 模型张量  中 y=时间, x=通道 （因为我们 permute 成了 (C, W=time, H=ch)）
        scale_t = new_W / max(orig_W, 1)                  # 时间缩放
        scale_ch = new_H / max(orig_H, 1)                 # 通道缩放

        # mask 的形状按 (row=time, col=channel) 理解
        new_h, new_w = new_W, new_H

        # ===== 第五步：构建检测/分割 target =====
        boxes = []
        labels = []
        masks = []
        attention_masks = []
        areas = []
        iscrowd = []

        for ann in annotations:
            bbox = ann["bbox"]                            # COCO: [x(time), y(channel), w(time), h(channel)]
            x0, y0, bw, bh = bbox

            # 坐标交换 + 缩放：把 (time, channel) 的 COCO 坐标变换到 (channel, time) 的模型坐标
            x1 = y0 * scale_ch                            # 模型 x = channel
            y1 = x0 * scale_t                             # 模型 y = time
            x2 = (y0 + bh) * scale_ch
            y2 = (x0 + bw) * scale_t

            boxes.append([x1, y1, x2, y2])                # [xmin, ymin, xmax, ymax]
            labels.append(ann["category_id"])

            # 面积：缩放后时间方向 * 通道方向
            areas.append((bw * scale_t) * (bh * scale_ch))
            iscrowd.append(ann.get("iscrowd", 0))

            # ===== 高斯线掩膜：把多边形折线晕染成带宽度的高斯条带 =====
            mask = np.zeros((new_h, new_w), dtype=np.float32)  # (time, channel)

            std_dev = self.gaussian_std_dev_x * scale_t   # 高斯宽度也随缩放走
            amplitude = float(self.gaussian_amplitude)

            segs = ann.get("segmentation", [])
            for seg in segs:
                if len(seg) < 10:                         # 少于 5 个点的折线认为无效
                    continue

                # 把原始折线点 (time, channel) 转成 (channel, time) 像素坐标
                pts = []
                for i in range(0, len(seg), 2):
                    t = seg[i]                            # 时间
                    ch = seg[i + 1]                       # 通道
                    x = int(ch * scale_ch)                # 列 = 通道
                    y = int(t * scale_t)                  # 行 = 时间
                    pts.append((x, y))

                # 折线整数化
                pts = interpolate_line_segments_int(pts)
                for x, y in pts:
                    if 0 <= x < new_w:
                        # 每一列只保留“最强响应”：用 maximum 而不是累加，
                        # 避免折线拐点处高斯峰叠加造成多峰。
                        mask[:, x] = np.maximum(mask[:, x], gaussian_line(y, std_dev, amplitude, new_h))

            # 若某列峰值超过 amplitude，把该列整体压回 amplitude（防止 *255 溢出）
            col_max = np.max(mask, axis=0) if mask.size else np.zeros((new_w,), dtype=np.float32)
            cols = col_max > amplitude
            if np.any(cols):
                scales = col_max[cols] / amplitude
                mask[:, cols] /= scales[None, :]

            mask_u8 = (mask * 255).astype(np.uint8)       # 转成 uint8 存储
            masks.append(mask_u8)

            # ===== attention_mask（可选，每个实例一张） =====
            if "attention_mask" in ann:
                att = self._generate_attention_mask(
                    ann["attention_mask"],
                    mask_shape=(new_h, new_w),
                    original_shape=original_size_for_labels,  # (orig_W, orig_H)
                )
            else:
                att = np.zeros((new_h, new_w), dtype=np.uint8)

            attention_masks.append(att)

        # 全部打包成 tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in masks])
        attention_masks = torch.stack([torch.tensor(a, dtype=torch.uint8) for a in attention_masks])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # ===== 第六步：垂直翻转增广（沿通道轴翻）=====
        if self.enable_vflip and (random.random() < self.vflip_prob):
            # image shape = (C, W=time, H=channel)；vflip 翻 dim=1 (这里的 "height")
            image = F.vflip(image)

            # box 的 y 坐标（通道方向）同步翻转：y' = H - y
            H_img = image.shape[1]
            boxes[:, [1, 3]] = H_img - boxes[:, [3, 1]]

            # masks / attention_masks 沿通道维（dim=1）翻转
            masks = torch.flip(masks, dims=[1])
            attention_masks = torch.flip(attention_masks, dims=[1])

        target = {
            "boxes": boxes,                               # [N, 4]
            "labels": labels,                             # [N]
            "masks": masks,                               # [N, H, W] uint8
            "attention_masks": attention_masks,           # [N, H, W] uint8
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image, target


# =========================================================
# 推理集 Dataset（只做预处理 + resize，不需要标注）
# =========================================================

class DASInferDataset(Dataset):
    """预测阶段使用的数据集。"""

    def __init__(
        self,
        hdf5_files,                                      # list[str]：HDF5 路径列表（可为 gs:// URL）
        resize_scale: float = 0.5,
        data_key: str = "data",
        data_is_strain_rate: bool = True,
        channel_range=None,
        f_band=(2.0, 10.0),
        f_high=10.0,
        storage_backend: str = "auto",                   # "auto" / "local" / "gcs"
        gcs_key_path: Optional[str] = None,
    ):
        """
        storage_backend:
          - "local": 强制本地
          - "gcs":   强制 GCS
          - "auto":  按路径前缀自动判断（gs:// 走 GCS）
        """
        self.hdf5_files = list(hdf5_files)
        self.resize_scale = float(resize_scale)
        self.data_key = data_key
        self.data_is_strain_rate = data_is_strain_rate
        self.channel_range = channel_range
        self.f_band = f_band
        self.f_high = f_high
        self.storage_backend = storage_backend
        self.gcs_key_path = gcs_key_path

    def __len__(self):
        return len(self.hdf5_files)

    def _resolve_backend(self, path: str) -> str:
        """根据配置和路径前缀决定实际走的后端。"""
        if self.storage_backend == "local":
            return "local"
        if self.storage_backend == "gcs":
            return "gcs"
        if isinstance(path, str) and path.startswith("gs://"):
            return "gcs"
        return "local"

    def __getitem__(self, idx):
        file_path = self.hdf5_files[idx]
        backend = self._resolve_backend(file_path)

        # 预处理成 RGB
        rgb, _dt = preprocess_data_rgb(
            file_path,
            channel_range=self.channel_range,
            data_key=self.data_key,
            data_is_strain_rate=self.data_is_strain_rate,
            f_band=self.f_band,
            f_high=self.f_high,
            storage_backend=backend,
            gcs_key_path=self.gcs_key_path,
        )                                                 # (H, W, 3)

        # 和训练集相同的 permute：(H, W, 3) -> (3, W, H)
        image = torch.from_numpy(rgb).permute(2, 1, 0)

        # 按比例缩放
        orig_W = image.shape[1]
        orig_H = image.shape[2]
        new_W = max(1, int(orig_W * self.resize_scale))
        new_H = max(1, int(orig_H * self.resize_scale))
        image = F.resize(image, (new_W, new_H))

        # 返回张量 + 文件名，方便下游写结果
        return image, os.path.basename(file_path)
