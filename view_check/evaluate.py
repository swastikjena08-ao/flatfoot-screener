"""Score the view classifier on the test split and on the hand-sorted extras.

Also scores FracAtlas test X-rays on their own: almost every top-down training image
comes from one hospital's bunion set, so this checks the model learned the view and not
the hospital.

Usage:
  python evaluate.py --cache work/cache.npz --model work/view/best.pt --out work/eval
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from common import CLASSES, resnet18, shrink
from train import predict, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.cache, allow_pickle=True)
    keep = np.isin(d["split"], ["test", "extra", "outside_test"])
    imgs, y, split, view, source, path = (d[k][keep] for k in ["img", "y", "split", "view", "source", "path"])
    model = resnet18(len(CLASSES))
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    p = predict(model, shrink(imgs))
    pred = np.array(CLASSES)[p.argmax(1).numpy()]

    t = split == "test"
    out = {"test": scores(p[t], y[t]), "test_n": int(t.sum())}
    for src in ["fracatlas", "unifesp"]:
        m = t & (source == src)
        if m.any():
            out[f"test_{src}_only"] = {"accuracy": float((pred[m] == view[m]).mean()), "n": int(m.sum()),
                                       "recall": {c: float((pred[m & (view == c)] == c).mean())
                                                  for c in CLASSES if (m & (view == c)).any()}}
    o = (split == "outside_test") & (y >= 0)
    if o.any():
        out["outside_test"] = scores(p[o], y[o])
        out["outside_test_n"] = int(o.sum())
        out["outside_confusion"] = pd.crosstab(pd.Series(view[o], name="true"),
                                               pd.Series(pred[o], name="predicted")).to_dict()
    cm = pd.crosstab(pd.Series(view[t], name="true"), pd.Series(pred[t], name="predicted"))
    out["confusion"] = cm.to_dict()
    e = (split == "extra") | ((split == "outside_test") & (y < 0))
    out["extras"] = pd.crosstab(pd.Series(view[e], name="hand_sorted"), pd.Series(pred[e], name="predicted")).to_dict()
    json.dump(out, open(os.path.join(a.out, "test_scores.json"), "w"), indent=1)
    pd.DataFrame({"path": path, "split": split, "true": view, "predicted": pred,
                  **{f"p_{c}": p[:, k].numpy() for k, c in enumerate(CLASSES)}}).to_csv(
        os.path.join(a.out, "predictions.csv"), index=False)
    print(cm)
    print(json.dumps({k: v for k, v in out.items() if k not in ("confusion", "extras")}, indent=1))
    print(pd.DataFrame(out["extras"]).fillna(0).astype(int))


if __name__ == "__main__":
    main()
