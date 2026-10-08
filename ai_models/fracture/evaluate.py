"""Score a trained fracture model on the test split, with the foot and ankle X-rays reported
on their own. The flag threshold is picked on the validation split (lowest score that still
catches 90% of its fractures), never on the test split.

"Points at the break" counts a fractured test X-ray as found when the model's hottest
12x12 cell overlaps a FracAtlas fracture box.

Usage:
  python evaluate.py --cache work/cache.npz --model work/run/best.pt --out work/eval [--overlays DIR]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from common import FractureNet, cell_targets
from train import predict

TARGET_SENS = 0.90


def threshold_for(p, y, sens=TARGET_SENS):
    pos = np.sort(p[y == 1])
    return float(pos[int(np.floor((1 - sens) * len(pos)))])


def report(p, y, maps, cells, t):
    flag = p >= t
    hot = maps.reshape(len(maps), -1).argmax(1)
    pos = y == 1
    hit = cells.reshape(len(cells), -1)[np.arange(len(cells)), hot] > 0
    return {"n": int(len(y)), "fractured": int(pos.sum()),
            "auc": float(roc_auc_score(y, p)) if 0 < pos.sum() < len(y) else None,
            "sensitivity": float(flag[pos].mean()) if pos.any() else None,
            "specificity": float((~flag[~pos]).mean()) if (~pos).any() else None,
            "flagged": int(flag.sum()),
            "points_at_break": float(hit[pos].mean()) if pos.any() else None}


def overlay(img, cmap, path):
    """Gray X-ray with the cell map in red on top."""
    n = img.shape[0]
    heat = np.asarray(Image.fromarray((cmap * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR),
                      np.float32) / 255
    rgb = np.repeat(img[..., None], 3, 2).astype(np.float32)
    rgb[..., 0] = np.clip(rgb[..., 0] + 200 * heat, 0, 255)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlays", help="folder for test overlay PNGs (keep it outside the repo)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.cache, allow_pickle=True)
    model = FractureNet()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    region, y = d["region"], d["y"]
    fa = np.isin(region, ["foot", "ankle", "foot_ankle_view"])
    res = {}
    va, te = d["split"] == "val", d["split"] == "test"
    pv, _ = predict(model, d["img"][va])
    t = threshold_for(pv, y[va])
    pt, mt = predict(model, d["img"][te])
    ct = cell_targets(d["mask"][te]).numpy()
    yt, fat, rt = y[te], fa[te], region[te]
    res["threshold"] = t
    res["test_all_leg"] = report(pt, yt, mt, ct, t)
    res["test_foot_ankle"] = report(pt[fat], yt[fat], mt[fat], ct[fat], t)
    for r in ["foot", "ankle", "distal_leg", "other_leg"]:
        m = (rt == r) & (yt == 1)
        res[f"found_{r}_fractures"] = f"{int((pt[m] >= t).sum())}/{int(m.sum())}"
    res["val_auc"] = float(roc_auc_score(y[va], pv))
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    rows = ["image_id,region,fractured,prob"] + [f"{n},{r},{int(v)},{q:.4f}" for n, r, v, q in
                                                 zip(d["names"][te], rt, yt, pt)]
    open(os.path.join(a.out, "test_predictions.csv"), "w").write("\n".join(rows) + "\n")
    print(json.dumps(res, indent=1))

    if a.overlays:
        os.makedirs(a.overlays, exist_ok=True)
        imgs = d["img"][te]
        for i in np.flatnonzero(fat & ((yt == 1) | (pt >= t))):
            tag = ("TP" if pt[i] >= t else "FN") if yt[i] else "FP"
            overlay(imgs[i], mt[i], os.path.join(a.overlays, f"{tag}_{pt[i]:.2f}_{d['names'][te][i][:-4]}.png"))


if __name__ == "__main__":
    main()
