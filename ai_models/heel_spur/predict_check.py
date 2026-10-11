"""Check that the exported ONNX pipeline (predict.py) gives the same answers as the
PyTorch evaluation (evaluate.py) on the test X-rays.

Usage:
  python predict_check.py --data D:/heel-spur --split work/split.csv --locate work/locate/best.pt \
      --classify work/classify/best.pt --onnx-dir . [--n 60]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import CLASSES, CROP_CACHE, LOC_SIZE, HeelFinder, heel_crop, letterbox, load_gray, resnet18
from predict import HeelSpur
from prep import FOLDERS, LABELS
from train import center, predict, to_xyxy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--locate", required=True)
    ap.add_argument("--classify", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.data, "boundingbox.csv"), encoding="utf-8-sig")
    df["label"] = df["label"].map(LABELS)
    df = df.merge(pd.read_csv(a.split)[["unique_name", "split"]], on="unique_name")
    df = df[df["split"] == "test"].sample(a.n, random_state=0).reset_index(drop=True)
    paths = [os.path.join(a.data, "all_category", "all_category", FOLDERS[l], n)
             for l, n in zip(df["label"], df["unique_name"])]

    loc = HeelFinder()
    loc.load_state_dict(torch.load(a.locate, map_location="cpu"))
    clf = resnet18(len(CLASSES))
    clf.load_state_dict(torch.load(a.classify, map_location="cpu"))
    grays = [load_gray(p) for p in paths]
    lb = [letterbox(g, LOC_SIZE) for g in grays]
    pb = to_xyxy(predict(loc, np.stack([b[0] for b in lb]), "locate")).numpy() * LOC_SIZE
    boxes = [((x[0] - ox) / s, (x[1] - oy) / s, (x[2] - ox) / s, (x[3] - oy) / s) for x, (_, s, ox, oy) in zip(pb, lb)]
    pt = predict(clf, center(np.stack([heel_crop(g, b, CROP_CACHE) for g, b in zip(grays, boxes)])), "classify").numpy()

    hs = HeelSpur(a.onnx_dir)
    res = [hs.predict(Image.open(p)) for p in paths]
    ox = np.array([[r["probs"][c] for c in CLASSES] for r in res])
    box_err = np.abs(np.array([r["box"] for r in res]) - np.array(boxes)).max()
    print(f"{a.n} test X-rays: max prob difference {np.abs(ox - pt).max():.5f}, "
          f"max box difference {box_err:.3f} px, same label {int((ox.argmax(1) == pt.argmax(1)).sum())}/{a.n}")
    ok = np.abs(ox - pt).max() < 1e-3 and (ox.argmax(1) == pt.argmax(1)).all()
    print("PASS" if ok else "FAIL")
