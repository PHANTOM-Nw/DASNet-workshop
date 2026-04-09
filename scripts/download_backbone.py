#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-download the pretrained weights DASNet's build_model() needs, so training
can start offline afterwards.

Two checkpoints:
  1. ResNet50 ImageNet1K_V1 backbone
     (used when build_model(pretrained_backbone=True))
  2. MaskRCNN-ResNet50-FPN COCO weights
     (used when build_model(pretrained=True))

Files are saved into torch hub's default cache directory:
    ~/.cache/torch/hub/checkpoints/   (Linux/macOS)
    C:\\Users\\<you>\\.cache\\torch\\hub\\checkpoints\\   (Windows)

Usage:
    python scripts/download_backbone.py                # both
    python scripts/download_backbone.py --only resnet50
    python scripts/download_backbone.py --only maskrcnn
    python scripts/download_backbone.py --dest ./weights  # custom directory
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import torch

# Canonical URLs that torchvision itself uses. See:
#   torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1
#   torchvision.models.detection.mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
# Mirror first (fast in CN), official as fallback.
# SJTU mirror maps  download.pytorch.org/models/X  ->  mirror.sjtu.edu.cn/pytorch-models/X
MIRRORS = [
    "https://mirror.sjtu.edu.cn/pytorch-models",
    "https://download.pytorch.org/models",
]

# filename -> expected size in bytes (approx; used to spot corrupted partial downloads)
CHECKPOINTS: dict[str, tuple[str, int]] = {
    "resnet50": ("resnet50-0676ba61.pth", 102_530_333),
    "maskrcnn": ("maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth", 178_090_079),
}

MIN_SIZE_RATIO = 0.95  # anything smaller than 95% of expected is considered corrupt


def default_hub_dir() -> Path:
    """torch.hub default: $TORCH_HOME/hub or ~/.cache/torch/hub."""
    return Path(torch.hub.get_dir()) / "checkpoints"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(f"\r  {pct:6.2f}%  ({downloaded/1e6:7.1f} / {total_size/1e6:7.1f} MB)")
        sys.stdout.flush()


def download(name: str, dest_dir: Path) -> Path:
    filename, expected_size = CHECKPOINTS[name]
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename

    min_ok = int(expected_size * MIN_SIZE_RATIO)
    if target.exists():
        actual = target.stat().st_size
        if actual >= min_ok:
            print(f"[skip] {target}  already exists ({actual/1e6:.1f} MB)")
            return target
        print(f"[warn] {target} is only {actual/1e6:.1f} MB "
              f"(expected ~{expected_size/1e6:.1f} MB) — re-downloading")
        target.unlink()

    last_err = None
    for base in MIRRORS:
        url = f"{base}/{filename}"
        print(f"[get ] {url}")
        print(f"       -> {target}")
        try:
            urlretrieve(url, target, reporthook=_progress)
            print()
            print(f"[done] {target}  ({target.stat().st_size/1e6:.1f} MB)")
            return target
        except Exception as e:
            print()
            print(f"[fail] {base}: {e}")
            if target.exists():
                target.unlink()
            last_err = e

    print("All mirrors failed. Download the file manually in a browser and drop it into:")
    print(f"  {dest_dir}")
    print("Candidate URLs:")
    for base in MIRRORS:
        print(f"  {base}/{filename}")
    raise RuntimeError(f"download {name} failed") from last_err


def verify_load(name: str, path: Path) -> None:
    """Sanity-check the file is a valid torch checkpoint."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 2.4 has no weights_only kwarg
        state = torch.load(path, map_location="cpu")
    n = len(state) if hasattr(state, "__len__") else "?"
    print(f"[ok  ] {name}: loaded state_dict with {n} tensors")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained backbone weights for DASNet")
    parser.add_argument(
        "--only",
        choices=list(CHECKPOINTS.keys()),
        default=None,
        help="Download only one of: resnet50, maskrcnn. Default: both.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Target directory. Defaults to torch hub cache "
             "(so torchvision picks it up automatically at model build time).",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest).expanduser().resolve() if args.dest else default_hub_dir()
    print(f"Destination: {dest_dir}")

    names = [args.only] if args.only else list(CHECKPOINTS.keys())
    for name in names:
        path = download(name, dest_dir)
        verify_load(name, path)

    print()
    print("Done. Now you can build the model offline, e.g.:")
    print("    from dasnet.model.dasnet import build_model")
    print("    model = build_model(num_classes=2, pretrained_backbone=True)")


if __name__ == "__main__":
    main()
