"""Crop every labeled foot from HVAngleEst into a fixed-size cache.

One sample per foot (a row of datasets.csv). Left feet are mirrored so every
crop looks like a right foot. The crop keeps its aspect ratio (letterboxed),
so angles in the crop equal angles in the original X-ray.

Usage: python prep.py --data /mnt/project-files/datasets/hvangleest --out cache.npz
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from common import IN_W, IN_H, LINES, canonical_points

MARGIN = 0.05  # extra border around the labeled foot box, as a fraction of box size


def parse(s):
    return np.array([float(v) for v in str(s).split(",")], dtype=np.float64)


def crop_foot(img, box_px, mirror):
    """Return the letterboxed crop and the affine map from image px to crop px."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_px
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - MARGIN * bw, x2 + MARGIN * bw
    y1, y2 = y1 - MARGIN * bh, y2 + MARGIN * bh
    s = min(IN_W / (x2 - x1), IN_H / (y2 - y1))
    ox = (IN_W - s * (x2 - x1)) / 2
    oy = (IN_H - s * (y2 - y1)) / 2
    # image px -> crop px: u = s*(x - x1) + ox (mirrored: IN_W - u)
    if mirror:
        M = np.array([[-s, 0, IN_W - (ox - s * x1)], [0, s, oy - s * y1]])
    else:
        M = np.array([[s, 0, ox - s * x1], [0, s, oy - s * y1]])
    crop = cv2.warpAffine(img, M, (IN_W, IN_H), flags=cv2.INTER_AREA, borderValue=0)
    return crop, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/project-files/datasets/hvangleest")
    ap.add_argument("--out", default="cache.npz")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.data, "HVAngleEst", "datasets.csv"))
    split_of = {}
    for name in ("train", "val", "test"):
        for f in pd.read_csv(os.path.join(a.data, "splits", f"{name}.csv")).filename:
            split_of[f] = name
    df["split"] = df.filename.map(split_of).fillna("none")

    imgs, pts, rows = [], [], []
    cache = {}
    for i, r in df.iterrows():
        if r.filename not in cache:
            g = cv2.imread(os.path.join(a.data, "HVAngleEst", "images", r.filename), cv2.IMREAD_GRAYSCALE)
            cache = {r.filename: g}  # rows of one image are consecutive
        g = cache[r.filename]
        W, H = r.image_width, r.image_height
        assert g.shape == (H, W), (r.filename, g.shape, W, H)
        scale = np.array([W, H, W, H])
        mirror = r.labels == "left"
        crop, M = crop_foot(g, parse(r.boxes) * scale, mirror)
        p = []
        for c in LINES:
            l = parse(r[c]) * scale
            for k in (0, 2):
                p.append(M @ np.array([l[k], l[k + 1], 1.0]))
        imgs.append(crop)
        pts.append(canonical_points(np.array(p)))
        rows.append(dict(filename=r.filename, patient_id=r.patient_id, side=r.labels, split=r.split,
                         source=r.source, HVA=r.HVA, IMA=r.IMA, M=M.tolist()))
        if i % 200 == 0:
            print(i, len(df))
    meta = pd.DataFrame(rows)
    np.savez_compressed(a.out, imgs=np.stack(imgs), pts=np.stack(pts).astype(np.float32),
                        meta=meta.to_json(orient="records"))
    print("saved", a.out, len(imgs), meta.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
