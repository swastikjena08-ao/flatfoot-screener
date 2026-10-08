"""Cache letterboxed X-rays for the view classifier from the combined label table.

The table (labels.csv, built from five public datasets) has one row per image with its
view, a duplicate flag and a patient-level split; see README. Only the four trained views
are cached, plus the hand-sorted images the model never trains on (kept for reporting).

Usage:
  python prep.py --labels D:/datasets/xray-check/labels.csv --root D:/datasets --out work/cache.npz
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import ImageFile

from common import CACHE, CLASSES, letterbox_u8, load_gray

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few FracAtlas JPGs are cut short
EXTRA = ["ankle_front", "end_on", "foot_unsorted"]  # never trained on, scored for the README


def load_one(path):
    return letterbox_u8(load_gray(path), CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", required=True, help="folder the label table's paths are relative to")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.labels, keep_default_na=False)
    df = df[(df["duplicate"].astype(str) == "False") & df["view"].isin(CLASSES + EXTRA)].reset_index(drop=True)
    paths = [os.path.join(a.root, p) for p in df["path"]]
    with ProcessPoolExecutor() as ex:
        imgs = np.stack(list(ex.map(load_one, paths, chunksize=32)))
    y = np.array([CLASSES.index(v) if v in CLASSES else -1 for v in df["view"]])
    split = np.where((y >= 0) | (df["split"] == "outside_test"), df["split"], "extra")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, img=imgs, y=y, split=split, view=df["view"].to_numpy(),
             source=df["source"].to_numpy(), path=df["path"].to_numpy())
    print(pd.crosstab(df["view"], split))


if __name__ == "__main__":
    main()
