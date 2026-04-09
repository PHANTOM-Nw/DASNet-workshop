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
