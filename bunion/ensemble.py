"""Average the whole-foot and zoomed models' points, then score HVA/IMA.

Both models' points are mapped back to original X-ray pixels, averaged per
point (with a weight on the zoomed model), and the angles are measured from
the averaged lines.

Usage: python ensemble.py --zoom-cache work/zoom_cache.npz --ckpt2 work/zoom1/best.pt --out work/ensemble
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch

from model import PointNet
from train import predict, score


def to_image(M, p):
    inv = cv2.invertAffineTransform(np.array(M))
    return np.c_[p, np.ones(len(p))] @ inv.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom-cache", required=True, help="from zoom_prep.py (holds the first-pass points)")
    ap.add_argument("--ckpt2", required=True, help="zoomed-pass checkpoint")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    z = np.load(a.zoom_cache)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    model = PointNet(arch=a.arch)
    model.load_state_dict(torch.load(a.ckpt2, map_location="cpu"))
    if torch.cuda.is_available():
        model = model.cuda()
    pred2 = predict(model, z["imgs"])

    p1 = np.stack([to_image(m, p) for m, p in zip(meta.M1, z["pred1"])])
    p2 = np.stack([to_image(m, p) for m, p in zip(meta.M, pred2)])

    seen = set(meta[meta.split.isin(["train", "val"])].patient_id)
    sets = {"val": meta.split == "val", "test_official": meta.split == "test",
            "test_clean": (meta.split == "test") & ~meta.patient_id.isin(seen)}
    results = {}
    # pick the zoomed-model weight on val only, then report test with that weight
    best_w, best = None, 1e9
    for w in np.linspace(0, 1, 11):
        s = score(w * p2[sets["val"]] + (1 - w) * p1[sets["val"]], meta[sets["val"]])
        if s["hva_mae"] + s["ima_mae"] < best:
            best, best_w = s["hva_mae"] + s["ima_mae"], float(w)
    results["zoom_weight"] = round(best_w, 2)
    for name, mask in sets.items():
        s = score(best_w * p2[mask] + (1 - best_w) * p1[mask], meta[mask])
        results[name] = {k: round(v, 4) for k, v in s.items() if not k.endswith("_pred")}
        print(name, json.dumps(results[name]))
    print("zoom weight", best_w)
    with open(os.path.join(a.out, "test_scores.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
