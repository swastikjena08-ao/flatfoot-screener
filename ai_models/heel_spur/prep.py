"""Split the Kaggle heel dataset and cache the inputs for both models.

Near-identical images (same perceptual hash) are kept in the same split so the
test set never contains a copy of a training X-ray.

Usage:
  python prep.py --data /path/to/heel-spur --split split.csv --out work/cache.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from common import CLASSES, LOC_SIZE, heel_crop, letterbox, load_gray

LABELS = {"Normal": "normal", "heel spur": "spur", "Sever": "sever"}
FOLDERS = {"normal": "normal", "spur": "heelspur", "sever": "sever"}


def load_one(args):
    path, box = args
    gray = load_gray(path)
    canvas, s, ox, oy = letterbox(gray, LOC_SIZE)
    b = np.array(box, np.float32)
    loc_box = np.array([b[0] * s + ox, b[1] * s + oy, b[2] * s + ox, b[3] * s + oy], np.float32) / LOC_SIZE
    return canvas, loc_box, heel_crop(gray, b), str(imagehash.phash(Image.fromarray(gray)))


def make_split(df, seed=0, frac=(0.70, 0.15, 0.15)):
    """Group-level split stratified by class; a group is a set of images sharing a phash."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("phash").agg(label=("label", "first"), n=("label", "size")).reset_index()
    out = {}
    for _, g in groups.groupby("label"):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        cut = np.cumsum(g["n"].to_numpy()) / g["n"].sum()
        for h, c in zip(g["phash"], cut):
            out[h] = "train" if c <= frac[0] else "val" if c <= frac[0] + frac[1] else "test"
    return df["phash"].map(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.data, "boundingbox.csv"), encoding="utf-8-sig")
    df["label"] = df["label"].map(LABELS)
    df["path"] = [os.path.join(a.data, "all_category", "all_category", FOLDERS[l], n)
                  for l, n in zip(df["label"], df["unique_name"])]
    jobs = list(zip(df["path"], df[["xmin", "ymin", "xmax", "ymax"]].to_numpy()))
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(load_one, jobs, chunksize=32))
    df["phash"] = [r[3] for r in res]
    df["split"] = make_split(df)
    df[["unique_name", "label", "phash", "split"]].to_csv(a.split, index=False)
    print(pd.crosstab(df["split"], df["label"]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, names=df["unique_name"].to_numpy(), y=df["label"].map(CLASSES.index).to_numpy(),
             split=df["split"].to_numpy(), loc_img=np.stack([r[0] for r in res]),
             loc_box=np.stack([r[1] for r in res]), crop=np.stack([r[2] for r in res]))


if __name__ == "__main__":
    main()
