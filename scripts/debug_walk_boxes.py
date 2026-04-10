#!/usr/bin/env python3
"""Debug: print boxes and masks for the walking sample from DASTrainDataset."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dasnet.data.das import DASTrainDataset

coco_path = "e:/dasdata/DAS-dataset_dasnet/annotations_bin/train.json"
data_root = "e:/dasdata/DAS-dataset_dasnet/data"

# Also print raw COCO annotations for comparison
coco = json.load(open(coco_path))
for img in coco["images"]:
    if "walking_2023-04-17T121933" in img["file_name"]:
        img_id = img["id"]
        print(f"COCO image: id={img_id} w={img['width']} h={img['height']} file={img['file_name']}")
        raw_anns = [a for a in coco["annotations"] if a["image_id"] == img_id]
        print(f"COCO annotations: {len(raw_anns)}")
        for a in raw_anns:
            print(f"  ann {a['id']}: bbox={a['bbox']}  cat={a['category_id']}")
        break

print("\n--- DASTrainDataset output ---")
ds = DASTrainDataset(
    ann_path=coco_path,
    root_dir=data_root,
    resize_scale=0.5,
    synthetic_noise=False,
    enable_stack_event=False,
    enable_vflip=False,
    allowed_bins={},
    min_keep_bins={},
)

for idx in range(len(ds)):
    info = ds.coco.imgs[ds.ids[idx]]
    if "walking_2023-04-17T121933" in info["file_name"]:
        print(f"idx={idx} file={info['file_name']}")
        image, target = ds[idx]
        print(f"image shape: {tuple(image.shape)}")
        boxes = target["boxes"]
        print(f"boxes ({boxes.shape[0]}):")
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.tolist()
            print(f"  box {i}: x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f}  w={x2-x1:.1f} h={y2-y1:.1f}")
        masks = target["masks"]
        print(f"masks shape: {tuple(masks.shape)}")
        for i in range(masks.shape[0]):
            m = masks[i]
            print(f"  mask {i}: sum={int(m.sum())} nonzero={int((m > 0).sum())}  "
                  f"frac={float((m > 0).sum()) / m.numel():.4f}")
        att = target.get("attention_masks", None)
        if att is not None:
            print(f"attention_masks shape: {tuple(att.shape)} sum={int(att.sum())}")
        break
