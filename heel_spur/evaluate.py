"""Score the full pipeline (heel finder -> crop -> spur classifier) on the test X-rays.

The crop comes from the heel finder's box, not the hand-drawn one, so this is what the app would see.

Usage:
  python evaluate.py --data /path/to/heel-spur --split split.csv --locate work/locate/best.pt \
      --classify work/classify/best.pt --out work/eval
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from common import CLASSES, CROP_CACHE, CROP_SIZE, LOC_SIZE, box_iou, heel_crop, letterbox, load_gray, resnet18
from prep import FOLDERS, LABELS
from train import center, predict, to_xyxy


def spur_vs_normal(p, y):
    m = y != CLASSES.index("sever")
    s = p[m, 1] / (p[m, 0] + p[m, 1])
    t = y[m] == 1
    pred = s >= 0.5
    return {"auc": float(roc_auc_score(t, s)), "sensitivity": float(pred[t].mean()),
            "specificity": float((~pred[~t]).mean()), "n_spur": int(t.sum()), "n_normal": int((~t).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--locate", required=True)
    ap.add_argument("--classify", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(os.path.join(a.data, "boundingbox.csv"), encoding="utf-8-sig")
    df["label"] = df["label"].map(LABELS)
    df = df.merge(pd.read_csv(a.split)[["unique_name", "split"]], on="unique_name")
    df = df[df["split"] == a.subset].reset_index(drop=True)
    y = df["label"].map(CLASSES.index).to_numpy()

    loc = resnet18(4)
    loc.load_state_dict(torch.load(a.locate, map_location="cpu"))
    clf = resnet18(len(CLASSES))
    clf.load_state_dict(torch.load(a.classify, map_location="cpu"))

    grays = [load_gray(os.path.join(a.data, "all_category", "all_category", FOLDERS[l], n))
             for l, n in zip(df["label"], df["unique_name"])]
    lb = [letterbox(g, LOC_SIZE) for g in grays]
    pbox = to_xyxy(predict(loc, np.stack([b[0] for b in lb]), "locate")).numpy() * LOC_SIZE
    boxes = np.stack([[(bx[0] - ox) / s, (bx[1] - oy) / s, (bx[2] - ox) / s, (bx[3] - oy) / s]
                      for bx, (_, s, ox, oy) in zip(pbox, lb)])
    gt = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)
    iou = box_iou(torch.from_numpy(boxes).float(), torch.from_numpy(gt)).numpy()

    res = {}
    for name, bx in [("finder_box", boxes), ("hand_box", gt)]:
        crops = center(np.stack([heel_crop(g, b, CROP_CACHE) for g, b in zip(grays, bx)]))
        p = predict(clf, crops, "classify").numpy()
        res[name] = {"accuracy_3class": float((p.argmax(1) == y).mean()),
                     "confusion_rows_true_cols_pred": confusion_matrix(y, p.argmax(1)).tolist(),
                     "spur_vs_normal": spur_vs_normal(p, y),
                     "sever_vs_rest_auc": float(roc_auc_score(y == 2, p[:, 2]))}
        if name == "finder_box":
            out = df[["unique_name", "label"]].copy()
            out[[f"p_{c}" for c in CLASSES]] = p
            out["finder_iou"] = iou
            out.to_csv(os.path.join(a.out, f"{a.subset}_predictions.csv"), index=False)
    res["finder"] = {"mean_iou": float(iou.mean()), "iou_over_0.5": float((iou > 0.5).mean()),
                     "n": len(df), "classes": CLASSES}
    json.dump(res, open(os.path.join(a.out, f"{a.subset}_scores.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
