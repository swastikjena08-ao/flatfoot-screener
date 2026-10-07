"""Score the app's own model pipeline (assistant/engine.py) on held-out X-rays.

This runs exactly what the app runs when a doctor adds an X-ray, so the scores shown in
the app (assistant/model_scores.json) are the app's scores, not only the training code's.
Each check needs its dataset on disk; checks without data are skipped.

  python tools/check_models.py --pesplanus "D:/data/Pesplanus Dataset" --pesplanus-split pesplanus_split.csv
                               --heel D:/data/heel-spur --hvangleest D:/data/hvangleest
                               --fracatlas D:/data/FracAtlas --fracture-split split.csv
                               [--out scores.json]

* Flat foot: Pesplanus X-rays from patients never used for the landmark labels
  (group never_labeled_patient in pesplanus_split.csv); heel bone angle under 17.7 deg
  counts as flat, compared with the dataset folder.
* Heel spur: the heel_spur/prep.py test split (rebuilt here from the same perceptual
  hashes and seed), heel finder box -> classifier, as in heel_spur/evaluate.py.
* Bunion: HVAngleEst official test feet with the dataset's foot boxes, as in
  bunion/ensemble.py; also with the foot finder when footfinder.onnx is installed.
* Fracture: the fracture/prep.py test split (its split.csv), whole X-ray, flag threshold
  from fracture.json; scored on the foot and ankle X-rays, as in fracture/evaluate.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assistant import engine  # noqa: E402


def check_flat(eng, folder, split_csv):
    df = pd.read_csv(split_csv)
    df = df[df["group"] == "never_labeled_patient"]
    sub = {"flat": "pesplanus", "normal": "notpesplanus"}
    hits, rows = [], []
    t = eng.lm_meta["threshold_deg"]
    for _, r in df.iterrows():
        img = engine.open_xray(os.path.join(folder, sub[r.folder], r.file))
        res = eng.analyse_side(img)
        a = engine.arch_angle(res, {})
        pred = "flat" if a < t else "normal"
        hits.append(pred == r.folder)
        rows.append((r.folder, pred))
    rows = pd.DataFrame(rows, columns=["truth", "pred"])
    by = {k: float((g.pred == k).mean()) for k, g in rows.groupby("truth")}
    return {"set": "Pesplanus X-rays from patients never used in training labels", "n": len(df),
            "angle_matches_folder": float(np.mean(hits)), "by_folder": by,
            "n_flat": int((rows.truth == "flat").sum()), "n_normal": int((rows.truth == "normal").sum())}


def heel_test_split(folder):
    """heel_spur/prep.py make_split, without the training caches."""
    import imagehash
    from heel_spur.predict import to_gray
    labels = {"Normal": "normal", "heel spur": "spur", "Sever": "sever"}
    dirs = {"normal": "normal", "spur": "heelspur", "sever": "sever"}
    df = pd.read_csv(os.path.join(folder, "boundingbox.csv"), encoding="utf-8-sig")
    df["label"] = df["label"].map(labels)
    df["path"] = [os.path.join(folder, "all_category", "all_category", dirs[l], n) for l, n in zip(df.label, df.unique_name)]
    df["phash"] = [str(imagehash.phash(Image.fromarray(to_gray(Image.open(p))))) for p in df.path]
    rng = np.random.default_rng(0)
    groups = df.groupby("phash").agg(label=("label", "first"), n=("label", "size")).reset_index()
    split = {}
    for _, g in groups.groupby("label"):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        cut = np.cumsum(g["n"].to_numpy()) / g["n"].sum()
        for h, c in zip(g["phash"], cut):
            split[h] = "train" if c <= 0.70 else "val" if c <= 0.85 else "test"
    df["split"] = df.phash.map(split)
    return df[df.split == "test"].reset_index(drop=True)


def check_spur(eng, folder):
    from sklearn.metrics import roc_auc_score
    df = heel_test_split(folder)
    classes = ["normal", "spur", "sever"]
    p = np.array([[eng.spur.predict(Image.open(x))["probs"][c] for c in classes] for x in df.path])
    y = df.label.map(classes.index).to_numpy()
    m = y != 2
    s = p[m, 1] / (p[m, 0] + p[m, 1])
    t = y[m] == 1
    return {"set": "Kaggle Heel Dataset test split (prep.py, seed 0)", "n": len(df),
            "accuracy_3class": float((p.argmax(1) == y).mean()),
            "spur_vs_normal_auc": float(roc_auc_score(t, s)),
            "spurs_caught": float((s[t] >= 0.5).mean()), "n_spur": int(t.sum()),
            "normal_cleared": float((s[~t] < 0.5).mean()), "n_normal": int((~t).sum())}


def check_bunion(eng, folder):
    import cv2  # noqa: F401  (bunion.predict needs it)
    df = pd.read_csv(os.path.join(folder, "HVAngleEst", "datasets.csv"))
    test = set(pd.read_csv(os.path.join(folder, "splits", "test.csv")).filename)
    df = df[df.filename.isin(test)].reset_index(drop=True)
    out, ff_rows, cur, gray, found = [], [], None, None, {}
    for _, r in df.iterrows():
        if r.filename != cur:
            cur = r.filename
            gray = engine.to_gray(engine.open_xray(os.path.join(folder, "HVAngleEst", "images", r.filename)))
            found = eng.bunion.find_feet(gray) if eng.bunion.finder else {}
        wh = [r.image_width, r.image_height] * 2
        box = np.array([float(v) for v in r.boxes.split(",")]) * wh
        f = eng.bunion.measure(gray, box, r.labels)
        out.append((abs(f["hva"] - r.HVA), abs(f["ima"] - r.IMA)))
        if eng.bunion.finder:
            if r.labels in found:
                f2 = eng.bunion.measure(gray, found[r.labels], r.labels)
                ff_rows.append((abs(f2["hva"] - r.HVA), abs(f2["ima"] - r.IMA)))
            else:
                ff_rows.append((np.inf, np.inf))
    e = np.array(out)
    res = {"set": "HVAngleEst official test split (dataset foot boxes)", "n": len(e),
           "hva_within3": float((e[:, 0] <= 3).mean()), "ima_within3": float((e[:, 1] <= 3).mean()),
           "hva_mae": float(e[:, 0].mean()), "ima_mae": float(e[:, 1].mean())}
    if ff_rows:
        e2 = np.array(ff_rows)
        ok = np.isfinite(e2[:, 0])
        res["with_foot_finder"] = {"n": len(e2), "missed_feet": int((~ok).sum()),
                                   "hva_within3": float((e2[:, 0] <= 3).mean()), "ima_within3": float((e2[:, 1] <= 3).mean()),
                                   "hva_mae": float(e2[ok, 0].mean()), "ima_mae": float(e2[ok, 1].mean())}
    return res


def check_fracture(eng, folder, split_csv):
    from PIL import ImageFile
    from sklearn.metrics import roc_auc_score
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few FracAtlas JPGs are cut short, as in fracture/common.py
    df = pd.read_csv(split_csv)
    df = df[(df.split == "test") & df.region.isin(["foot", "ankle", "foot_ankle_view"])].reset_index(drop=True)
    p = []
    for _, r in df.iterrows():
        out = {}
        eng.check_fracture(engine.open_xray(os.path.join(folder, "images", "Fractured" if r.fractured else "Non_fractured",
                                                         r.image_id)), out)
        p.append(out["fracture"]["p"])
    p, y = np.array(p), df.fractured.to_numpy(bool)
    flag = p >= eng.fracture.threshold
    return {"set": "FracAtlas foot and ankle X-rays, fracture/prep.py test split", "n": len(df),
            "auc": float(roc_auc_score(y, p)), "fractures_caught": float(flag[y].mean()), "n_fractured": int(y.sum()),
            "normal_cleared": float((~flag[~y]).mean()), "n_normal": int((~y).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesplanus")
    ap.add_argument("--pesplanus-split")
    ap.add_argument("--heel")
    ap.add_argument("--hvangleest")
    ap.add_argument("--bunion-models")
    ap.add_argument("--fracatlas")
    ap.add_argument("--fracture-split")
    ap.add_argument("--out")
    a = ap.parse_args()
    eng = engine.Engine(a.bunion_models)
    print(json.dumps(eng.status, indent=1))
    res = {}
    if a.pesplanus and a.pesplanus_split:
        res["flat"] = check_flat(eng, a.pesplanus, a.pesplanus_split)
        print("flat foot", json.dumps(res["flat"]), flush=True)
    if a.heel:
        res["spur"] = check_spur(eng, a.heel)
        print("heel spur", json.dumps(res["spur"]), flush=True)
    if a.hvangleest and eng.bunion:
        res["bunion"] = check_bunion(eng, a.hvangleest)
        print("bunion", json.dumps(res["bunion"]), flush=True)
    if a.fracatlas and a.fracture_split and eng.fracture:
        res["fracture"] = check_fracture(eng, a.fracatlas, a.fracture_split)
        print("fracture", json.dumps(res["fracture"]), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
