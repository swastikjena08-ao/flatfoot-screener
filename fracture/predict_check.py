"""Check that the exported ONNX pipeline (predict.py) gives the same answers as the
PyTorch evaluation (evaluate.py) on the test X-rays.

Usage:
  python predict_check.py --data D:/fracatlas/FracAtlas --cache work/cache.npz --model work/run/best.pt \
      --onnx-dir ../flatfoot_app [--n 60]
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from common import FractureNet
from predict import Fracture
from train import predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FracAtlas folder (holds dataset.csv)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    d = np.load(a.cache, allow_pickle=True)
    te = np.flatnonzero(d["split"] == "test")
    pick = np.random.default_rng(0).choice(te, min(a.n, len(te)), replace=False)
    names = d["names"][pick]
    paths = [os.path.join(a.data, "images", "Fractured" if y else "Non_fractured", n)
             for n, y in zip(names, d["y"][pick])]

    m = FractureNet()
    m.load_state_dict(torch.load(a.model, map_location="cpu"))
    pt, mt = predict(m, d["img"][pick])

    fr = Fracture(a.onnx_dir)
    res = [fr.predict(Image.open(p)) for p in paths]
    po = np.array([r["p"] for r in res])
    mo = np.stack([r["heatmap"] for r in res])
    same = int(((po >= fr.threshold) == (pt >= fr.threshold)).sum())
    print(f"{len(pick)} test X-rays: max prob difference {np.abs(po - pt).max():.5f}, "
          f"max map difference {np.abs(mo - mt).max():.5f}, same flag {same}/{len(pick)}")
    ok = np.abs(po - pt).max() < 1e-3 and same == len(pick)
    print("PASS" if ok else "FAIL")
