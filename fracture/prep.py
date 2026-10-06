"""Split the FracAtlas leg X-rays and cache the letterboxed images and fracture box masks.

All 2,273 leg images are used (fractures elsewhere in the leg still teach the model what a
break looks like); the score that matters is on the foot and ankle part of the test split,
using the hand-sorted regions in leg_region_labels.csv.

FracAtlas has no patient ids. Views of one patient sit under consecutive image ids, so
images are grouped in blocks of BLOCK ids (plus any near-identical copies by perceptual
hash) and whole groups go to one split. This is a guess at patients, not a guarantee.

Usage:
  python prep.py --data /path/to/FracAtlas --regions leg_region_labels.csv --split work/split.csv --out work/cache.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from common import SIZE, box_mask, letterbox, load_gray

BLOCK = 10
FOOT_ANKLE = {"foot", "ankle", "foot_ankle_view"}


def read_boxes(path, w, h):
    """YOLO txt (class cx cy bw bh, normalised) -> list of pixel boxes (x0, y0, x1, y1)."""
    out = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def load_one(args):
    path, ann, size = args
    gray = load_gray(path)
    h, w = gray.shape
    canvas, s, ox, oy = letterbox(gray, size)
    boxes = [(x0 * s + ox, y0 * s + oy, x1 * s + ox, y1 * s + oy) for x0, y0, x1, y1 in read_boxes(ann, w, h)]
    return canvas, box_mask(boxes, size), str(imagehash.phash(Image.fromarray(gray))), len(boxes)


def make_split(df, seed=0, frac=(0.65, 0.15, 0.20)):
    """Group-level split, stratified by what the group holds (foot/ankle fracture first)."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            a = parent[a]
        return a

    for b, h in zip(df["block"], df["phash"]):
        parent[find("b%d" % b)] = find("h" + h)
    df = df.assign(group=[find("b%d" % b) for b in df["block"]])
    kind = df.assign(k=np.select([df["fractured"] & df["foot_ankle"], df["fractured"], df["foot_ankle"]],
                                 [0, 1, 2], 3)).groupby("group")["k"].min()
    rng = np.random.default_rng(seed)
    n = df.groupby("group").size()
    out = {}
    for _, gs in kind.groupby(kind):
        gs = gs.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        cut = np.cumsum(n[gs.index].to_numpy()) / n[gs.index].sum()
        for g, c in zip(gs.index, cut):
            out[g] = "train" if c <= frac[0] else "val" if c <= frac[0] + frac[1] else "test"
    return df["group"].map(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FracAtlas folder (holds dataset.csv)")
    ap.add_argument("--regions", default=os.path.join(os.path.dirname(__file__), "leg_region_labels.csv"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=SIZE, help="letterbox side in pixels, a multiple of 32")
    a = ap.parse_args()
    assert a.size % 32 == 0

    meta = pd.read_csv(os.path.join(a.data, "dataset.csv"))
    df = meta[meta["leg"] == 1].merge(pd.read_csv(a.regions)[["image_id", "region"]], on="image_id", how="left")
    assert df["region"].notna().all(), "leg_region_labels.csv is missing some leg images"
    df["fractured"] = df["fractured"] == 1
    df["foot_ankle"] = df["region"].isin(FOOT_ANKLE)
    df["block"] = df["image_id"].str.extract(r"(\d+)")[0].astype(int) // BLOCK
    folder = np.where(df["fractured"], "Fractured", "Non_fractured")
    jobs = [(os.path.join(a.data, "images", f, i), os.path.join(a.data, "Annotations", "YOLO", i[:-4] + ".txt"), a.size)
            for f, i in zip(folder, df["image_id"])]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(load_one, jobs, chunksize=16))
    df["phash"] = [r[2] for r in res]
    df["n_boxes"] = [r[3] for r in res]
    df["split"] = make_split(df)
    df[["image_id", "fractured", "region", "n_boxes", "phash", "split"]].to_csv(a.split, index=False)
    print(pd.crosstab(df["split"], [df["fractured"], df["foot_ankle"]]))
    print(pd.crosstab(df["split"], df["region"]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, names=df["image_id"].to_numpy(), y=df["fractured"].to_numpy(np.int64),
             region=df["region"].to_numpy(), split=df["split"].to_numpy(),
             img=np.stack([r[0] for r in res]), mask=np.stack([r[1] for r in res]))


if __name__ == "__main__":
    main()
