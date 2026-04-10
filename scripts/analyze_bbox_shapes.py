#!/usr/bin/env python3
"""Print the H/W aspect ratio distribution of COCO bboxes."""
import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    args = ap.parse_args()

    j = json.load(open(args.coco))
    # bbox format in COCO: [x_time, y_channel, w_time, h_channel]
    # In MODEL space (after das.py axis swap): x=channel, y=time
    # so model H=time=w_time, model W=channel=h_channel
    # torchvision aspect_ratio = model_H / model_W = w_time / h_channel
    ratios, widths, heights = [], [], []
    for a in j["annotations"]:
        x, y, w, h = a["bbox"]
        # model-space: H=w_time, W=h_channel
        ratios.append(w / h)  # model H/W = time/channel
        widths.append(h)      # model W = channel
        heights.append(w)     # model H = time
    r = np.array(ratios)
    wt = np.array(widths)
    hc = np.array(heights)

    print(f"N = {len(r)} boxes")
    print(f"H/W ratio: min={r.min():.4f} max={r.max():.4f} "
          f"mean={r.mean():.4f} median={np.median(r):.4f}")
    for p in (10, 25, 50, 75, 90):
        print(f"  p{p:>2d} = {np.percentile(r, p):.4f}")

    in_range = ((r >= 0.5) & (r <= 2.0)).sum()
    below = (r < 0.5).sum()
    above = (r > 2.0).sum()
    N = len(r)
    print(f"\nCoverage by stock anchor ratios (0.5, 1.0, 2.0):")
    print(f"  in (0.5, 2.0):  {in_range}/{N} = {in_range/N*100:.1f}%")
    print(f"  < 0.5 (time-long, channel-short): {below}/{N} = {below/N*100:.1f}%")
    print(f"  > 2.0 (channel-long, time-short): {above}/{N} = {above/N*100:.1f}%")

    print(f"\nWidth (time, px):   min={wt.min():.0f} max={wt.max():.0f} "
          f"median={np.median(wt):.0f}")
    print(f"Height (channel, px):min={hc.min():.0f} max={hc.max():.0f} "
          f"median={np.median(hc):.0f}")

    print("\nH/W histogram:")
    buckets = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5),
               (0.5, 2.0), (2.0, 5.0), (5.0, 100.0)]
    for lo, hi in buckets:
        n = ((r >= lo) & (r < hi)).sum()
        bar = "#" * min(50, n)
        print(f"  [{lo:>5.2f} - {hi:>6.2f}): {n:4d} {bar}")

    # -------- Anchor suggestions --------
    print("\n" + "=" * 60)
    print("ANCHOR SUGGESTIONS (Czech-adapted)")
    print("=" * 60)

    # 1) Quantile-based log-spaced ratios, 5 values
    log_r = np.log10(r)
    lo, hi = np.percentile(log_r, 5), np.percentile(log_r, 95)
    qs = np.linspace(lo, hi, 5)
    ratios_qlog = np.round(10 ** qs, 4)
    print(f"\n(A) Log-quantile H/W ratios (p5 to p95, 5 points):")
    print(f"    aspect_ratios = ({tuple(float(x) for x in ratios_qlog)},) * 5")

    # 2) K-means on (log_w, log_h) — YOLO-style anchor derivation.
    # Hand-rolled k-means (no sklearn dep).
    from math import log
    log_w = np.log(wt + 1e-9)
    log_h = np.log(hc + 1e-9)
    pts = np.stack([log_w, log_h], axis=1)
    K = 5
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=K, replace=False)
    centers = pts[idx].copy()
    for _ in range(200):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        assign = d.argmin(axis=1)
        new_centers = np.stack([
            pts[assign == k].mean(axis=0) if (assign == k).any() else centers[k]
            for k in range(K)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    centers_wh = np.exp(centers)
    # Sort by area
    areas = centers_wh[:, 0] * centers_wh[:, 1]
    order = np.argsort(areas)
    centers_wh = centers_wh[order]
    print(f"\n(B) K-means {K} cluster centers in (W_time, H_channel) pixels:")
    for i, (w, h) in enumerate(centers_wh):
        s = float(np.sqrt(w * h))          # geometric-mean "size"
        r_i = float(h / w)                 # H/W
        print(f"    cluster {i}: W={w:7.1f}  H={h:6.1f}  size={s:6.1f}  H/W={r_i:.4f}")

    sizes = tuple(float(round(np.sqrt(w * h), 1)) for w, h in centers_wh)
    ratios = tuple(float(round(h / w, 4)) for w, h in centers_wh)
    print(f"\n    => suggested DASNet build_model kwargs:")
    print(f"       anchor_sizes = {tuple((s,) for s in sizes)}")
    print(f"       aspect_ratios = {tuple((r_,) for r_ in ratios)}")
    print("\n    NOTE: this uses one size + one ratio per FPN level (tight fit).")
    print("    For more robust coverage, use the (A) ratios × 5 sizes instead.")


if __name__ == "__main__":
    main()
