"""Check that predict.py (the app's onnxruntime pipeline) scores like the training code.

Runs the two passes on the official test feet with the dataset's foot boxes and compares with
results/ensemble/test_scores.json. With --pass1-csv it also compares the first pass foot by foot
with evaluate.py's saved predictions.

Usage: python predict_check.py --data /path/to/hvangleest --models ../../flatfoot_app/models/bunion \
           [--pass1-csv pass1_test_predictions.csv]
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from predict import Bunion, angles, canonical, crop_map

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(s):
    return np.array([float(v) for v in str(s).split(",")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--pass1-csv")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.data, "HVAngleEst", "datasets.csv"))
    test = set(pd.read_csv(os.path.join(a.data, "splits", "test.csv")).filename)
    df = df[df.filename.isin(test)].reset_index(drop=True)
    b = Bunion(a.models)
    rows, cur, g = [], None, None
    for _, r in df.iterrows():
        if r.filename != cur:
            cur = r.filename
            g = cv2.imread(os.path.join(a.data, "HVAngleEst", "images", r.filename), cv2.IMREAD_GRAYSCALE)
        box = parse(r.boxes) * [r.image_width, r.image_height, r.image_width, r.image_height]
        f = b.measure(g, box, r.labels)
        # first pass alone, for the foot-by-foot comparison
        h1, w1 = b.size1
        p1 = canonical(b._points(b.pass1, g, crop_map(box, r.labels == "left", w1, h1), b.size1))
        rows.append(dict(filename=r.filename, side=r.labels, HVA=r.HVA, IMA=r.IMA, hva=f["hva"], ima=f["ima"],
                         hva1=angles(p1)[0], ima1=angles(p1)[1]))
    out = pd.DataFrame(rows)
    eh, ei = (out.hva - out.HVA).abs(), (out.ima - out.IMA).abs()
    mine = dict(n=len(out), hva_mae=eh.mean(), hva_within3=(eh <= 3).mean(), ima_mae=ei.mean(), ima_within3=(ei <= 3).mean())
    ref = json.load(open(os.path.join(HERE, "results", "ensemble", "test_scores.json")))["test_official"]
    print("app pipeline :", {k: round(float(v), 4) for k, v in mine.items()})
    print("ensemble.py  :", ref)
    if a.pass1_csv:
        p = pd.read_csv(a.pass1_csv).merge(out, on=["filename", "side"])
        d = np.maximum((p.HVA_pred - p.hva1).abs(), (p.IMA_pred - p.ima1).abs())
        print(f"first pass vs evaluate.py, {len(p)} feet: max angle difference {d.max():.3f} deg, "
              f"{(d <= 0.05).mean():.1%} within 0.05 deg")


if __name__ == "__main__":
    main()
