import json
import numpy as np
import pytest
from pathlib import Path

SRC_ROOT = Path("z:/DAS-dataset/data")
SAMPLE = SRC_ROOT / "car" / "auto_2023-04-17T124152+0100"
pytestmark = pytest.mark.skipif(
    not SAMPLE.with_suffix(".h5").exists(),
    reason="Czech DAS-dataset not mounted at z:/DAS-dataset",
)


def test_json_axis_convention_matches_npy():
    """curve.x must index channel (cols of bmp); curve.y must index time-window (rows of bmp)."""
    j = json.loads(SAMPLE.with_suffix(".json").read_text())
    bmp = np.load(SAMPLE.with_suffix(".npy"))
    n_tw, n_ch = j["shape"]
    assert bmp.shape == (n_tw, n_ch)

    rows_true, cols_true = np.where(bmp)
    bmp_row_lo, bmp_row_hi = int(rows_true.min()), int(rows_true.max())
    bmp_col_lo, bmp_col_hi = int(cols_true.min()), int(cols_true.max())

    for k, v in j.items():
        if not k.startswith("curve"):
            continue
        cx = np.asarray(v["x"])
        cy = np.asarray(v["y"])
        assert bmp_col_lo - 2 <= cx.min() and cx.max() <= bmp_col_hi + 2, (
            f"{k}.x out of channel range: [{cx.min()},{cx.max()}] vs bmp cols [{bmp_col_lo},{bmp_col_hi}]"
        )
        assert bmp_row_lo - 2 <= cy.min() and cy.max() <= bmp_row_hi + 2, (
            f"{k}.y out of time range: [{cy.min()},{cy.max()}] vs bmp rows [{bmp_row_lo},{bmp_row_hi}]"
        )
        assert 0 <= cx.min() and cx.max() < n_ch
        assert 0 <= cy.min() and cy.max() < n_tw


import h5py
from scripts.czech_das_to_dasnet import (
    decimate_and_rewrite_h5,
    DECIMATE, DT_S_OUT, OUT_DSET_NAME, RAW_DSET_PATH,
)


def test_decimate_and_rewrite_h5_shapes_and_attrs(tmp_path):
    src = SAMPLE.with_suffix(".h5")
    dst = tmp_path / "out.h5"

    decimate_and_rewrite_h5(src, dst)

    with h5py.File(src, "r") as f_src, h5py.File(dst, "r") as f_dst:
        nt_raw, nch = f_src[RAW_DSET_PATH].shape
        dset = f_dst[OUT_DSET_NAME]
        assert dset.shape[0] == nch, "axis 0 must be channels"
        expected_nt = -(-nt_raw // DECIMATE)
        assert abs(dset.shape[1] - expected_nt) <= 1
        assert dset.dtype == np.float32
        assert float(dset.attrs["dt_s"]) == pytest.approx(DT_S_OUT)
        sample = dset[dset.shape[0] // 2, :]
        assert np.all(np.isfinite(sample))


from scripts.czech_das_to_dasnet import (
    curves_json_to_coco_anns,
    ANN_Y_TO_T_DEC,
    LABEL_TO_CATID,
)


def test_curves_json_to_coco_anns_basic_shape():
    json_path = SAMPLE.with_suffix(".json")
    anns, next_id = curves_json_to_coco_anns(
        json_path=json_path,
        image_id=1,
        category_id=LABEL_TO_CATID["car"],
        start_ann_id=100,
    )
    assert len(anns) >= 1
    assert next_id == 100 + len(anns)

    j = json.loads(json_path.read_text())
    n_tw, n_ch = j["shape"]
    t_max = (n_tw - 1) * ANN_Y_TO_T_DEC + 1

    for i, a in enumerate(anns):
        assert a["id"] == 100 + i
        assert a["image_id"] == 1
        assert a["category_id"] == LABEL_TO_CATID["car"]
        assert a["iscrowd"] == 0
        x, y, w, h = a["bbox"]
        assert 0 <= x and x + w <= t_max + 1, f"bbox time overflow: {a['bbox']}, t_max={t_max}"
        assert 0 <= y and y + h <= n_ch + 1, f"bbox channel overflow: {a['bbox']}, n_ch={n_ch}"
        assert w > 0 and h > 0
        assert isinstance(a["segmentation"], list) and len(a["segmentation"]) == 1
        poly = a["segmentation"][0]
        assert len(poly) >= 10 and len(poly) % 2 == 0
        times = poly[0::2]
        chans = poly[1::2]
        assert min(times) >= 0 and max(times) <= t_max
        assert min(chans) >= 0 and max(chans) < n_ch
        assert a["area"] == pytest.approx(w * h)


def test_curves_json_drops_degenerate_curves(tmp_path):
    fake = {
        "shape": [100, 200],
        "curve0": {"x": [10, 10, 10], "y": [1, 2, 3]},
        "curve1": {"x": [10, 11, 12, 13, 14, 15], "y": [0, 1, 2, 3, 4, 5]},
    }
    p = tmp_path / "fake.json"
    p.write_text(json.dumps(fake))
    anns, _ = curves_json_to_coco_anns(p, image_id=7, category_id=1, start_ann_id=0)
    assert len(anns) == 1
    assert len(anns[0]["segmentation"][0]) == 12


from scripts.czech_das_to_dasnet import build_coco_and_convert


def test_build_coco_and_convert_one_sample(tmp_path):
    out_root = tmp_path / "czech_das"
    coco_path = tmp_path / "coco.json"

    manifest = [("car", SAMPLE.with_suffix(".h5"))]
    build_coco_and_convert(
        manifest=manifest,
        out_root=out_root,
        coco_out=coco_path,
    )

    out_h5 = out_root / "car" / (SAMPLE.name + ".h5")
    assert out_h5.exists()
    with h5py.File(out_h5, "r") as f:
        assert OUT_DSET_NAME in f
        assert f[OUT_DSET_NAME].shape[0] > 1000
        assert float(f[OUT_DSET_NAME].attrs["dt_s"]) == pytest.approx(DT_S_OUT)

    from pycocotools.coco import COCO
    coco = COCO(str(coco_path))
    assert len(coco.imgs) == 1
    img = list(coco.imgs.values())[0]
    assert img["file_name"].endswith("_0.jpg")
    assert img["file_name"].replace("_0.jpg", "") == f"car/{SAMPLE.name}.h5"
    assert img["width"] > 0 and img["height"] > 0

    anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"]))
    assert len(anns) >= 1
    assert all(a["category_id"] == LABEL_TO_CATID["car"] for a in anns)


def test_dasnet_train_dataset_consumes_converted_output(tmp_path):
    """The whole point: DASTrainDataset must load a converted sample without code changes."""
    import torch
    from dasnet.data.das import DASTrainDataset

    out_root = tmp_path / "czech_das"
    coco_path = tmp_path / "mini_coco.json"

    build_coco_and_convert(
        manifest=[("car", SAMPLE.with_suffix(".h5"))],
        out_root=out_root,
        coco_out=coco_path,
    )

    ds = DASTrainDataset(
        ann_path=str(coco_path),
        root_dir=str(out_root),
        resize_scale=0.25,
        synthetic_noise=False,
        enable_stack_event=False,
        enable_vflip=False,
    )
    assert len(ds) == 1
    image, target = ds[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3
    assert image.dtype == torch.float32
    assert torch.all(torch.isfinite(image))

    for key in ("boxes", "labels", "masks", "attention_masks", "image_id", "area", "iscrowd"):
        assert key in target
    assert target["boxes"].shape[0] >= 1
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
    assert target["masks"].shape[0] == target["boxes"].shape[0]
    assert target["masks"].shape[1:] == (image.shape[1], image.shape[2])
    assert int(target["masks"].sum()) > 0
