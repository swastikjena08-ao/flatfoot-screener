"""Runs every collected X-ray through Remi AI the way the app does and records each outcome.

For each X-ray:
  1. the X-ray check decides the view (side, top-down, angled, not a foot, or unsure),
  2. the app engine runs only the checks that fit the true view (an unsure view is the case
     where the app asks the doctor to pick, so the doctor's pick is the true view),
  3. the outcome is saved next to the dataset's own label.

Nothing is trained or tuned here. Images, labels and per-image outputs stay outside the repo;
summarise.py turns the per-image CSVs into the scores.

  python full_test.py --app <checkout with remi_ai/ and ai_models/> --data <datasets folder>
                      --out <output folder> [--workers 4] [--only routing,flat_foot,heel_spur,bunion]
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

ENGINE = None
DATA = None


def _init(app, data):
    """One engine per worker, each ONNX session limited to one thread so workers don't compete."""
    global ENGINE, DATA
    import onnxruntime as ort
    real = ort.InferenceSession

    def single_thread(*args, **kw):
        if "sess_options" not in kw and len(args) < 2:
            so = ort.SessionOptions()
            so.intra_op_num_threads = so.inter_op_num_threads = 1
            kw["sess_options"] = so
        return real(*args, **kw)

    ort.InferenceSession = single_thread
    sys.path.insert(0, app)
    from remi_ai.engine import Engine
    ENGINE, DATA = Engine(os.path.join(app, "ai_models")), data


def _view(img):
    vc = ENGINE._load("view_check")
    r = vc.predict(img)
    return r["view"], r["probs"]


def run_one(job):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    kind, path, meta = job
    full = os.path.join(DATA, path)
    out = dict(meta, path=path)
    t0 = time.time()
    try:
        img = Image.open(full)
        img.load()
        view, probs = _view(img)
        out.update(pred_view=view, **{f"p_{k}": round(v, 4) for k, v in probs.items()})
        if kind != "routing":
            res = ENGINE.analyse(full, meta["true_view"])
            c, s = res["checks"], res["skipped"]
            if "flat_foot" in c:
                f = c["flat_foot"]
                out.update(ff_category=f["category"], ff_angle=f["angle"], ff_p_flat=f["p_flat"],
                           ff_tilt=f["tilt"], ff_left=f["left"])
            elif "flat_foot" in ROUTES_FOR(meta["true_view"]):
                out.update(ff_category="Error: " + s.get("flat_foot", ""))
            if "heel_spur" in c:
                h = c["heel_spur"]
                out.update(hs_label=h["label"], **{f"hs_p_{k}": v for k, v in h["probs"].items()})
            elif "heel_spur" in ROUTES_FOR(meta["true_view"]):
                out.update(hs_label="Error: " + s.get("heel_spur", ""))
            if "bunion" in c:
                out["bn_feet"] = json.dumps([{k: f[k] for k in ("side", "hva", "ima", "grade", "box")}
                                             for f in c["bunion"]["feet"]])
            elif "bunion" in ROUTES_FOR(meta["true_view"]):
                out["bn_feet"] = "Error: " + s.get("bunion", "")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["seconds"] = round(time.time() - t0, 3)
    return kind, out


def ROUTES_FOR(view):
    from remi_ai.engine import ROUTES
    return ROUTES.get(view, [])


# ---- job lists ------------------------------------------------------------------------------
def routing_jobs(data):
    """Every distinct image in the X-ray check table with a usable view label."""
    keep = {"side", "top_down", "oblique", "not_foot"}
    for r in csv.DictReader(open(os.path.join(data, "xray-check", "labels.csv"), encoding="utf-8")):
        if r["view"] in keep and r["duplicate"] != "True":
            yield ("routing", r["path"], {"source": r["source"], "true_view": r["view"], "split": r["split"]})


def flat_foot_jobs(data, split_csv):
    folders = {"flat": "pesplanus", "normal": "notpesplanus"}
    base = os.path.join(data, "pesplanus", "Pesplanus Dataset")
    for r in csv.DictReader(open(split_csv, encoding="utf-8")):
        folder = os.path.join(base, folders[r["folder"]])
        name = next((f for f in os.listdir(folder) if f.lower() == r["file"].lower()), None)
        if name:
            yield ("flat_foot", os.path.relpath(os.path.join(folder, name), data),
                   {"file": r["file"], "truth": r["folder"], "group": r["group"], "true_view": "side"})


def heel_spur_jobs(data, split_csv):
    folders = {"normal": "normal", "spur": "heelspur", "sever": "sever"}
    base = os.path.join("heel-spur", "all_category", "all_category")
    for r in csv.DictReader(open(split_csv, encoding="utf-8")):
        yield ("heel_spur", os.path.join(base, folders[r["label"]], r["unique_name"]),
               {"file": r["unique_name"], "truth": r["label"], "split": r["split"], "true_view": "side"})


def bunion_jobs(data):
    root = os.path.join(data, "hvangleest")
    split = {}
    for s in ("train", "val", "test"):
        for r in csv.DictReader(open(os.path.join(root, "splits", f"{s}.csv"), encoding="utf-8")):
            split[r["filename"]] = s
    feet = {}
    for r in csv.DictReader(open(os.path.join(root, "HVAngleEst", "datasets.csv"), encoding="utf-8")):
        feet.setdefault(r["filename"], []).append({"side": r["labels"], "hva": float(r["HVA"]),
                                                   "ima": float(r["IMA"])})
    for name, f in feet.items():
        yield ("bunion", os.path.join("hvangleest", "HVAngleEst", "images", name),
               {"file": name, "split": split.get(name, "none"), "truth_feet": json.dumps(f),
                "true_view": "top_down"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pesplanus-split", required=True)
    ap.add_argument("--heel-spur-split", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="routing,flat_foot,heel_spur,bunion")
    ap.add_argument("--limit", type=int, default=0, help="first N jobs per kind, for a quick trial")
    a = ap.parse_args()

    makers = {"routing": lambda: routing_jobs(a.data),
              "flat_foot": lambda: flat_foot_jobs(a.data, a.pesplanus_split),
              "heel_spur": lambda: heel_spur_jobs(a.data, a.heel_spur_split),
              "bunion": lambda: bunion_jobs(a.data)}
    os.makedirs(a.out, exist_ok=True)
    for kind in a.only.split(","):
        jobs = list(makers[kind]())
        if a.limit:
            jobs = jobs[:a.limit]
        rows, t0 = [], time.time()
        with Pool(a.workers, initializer=_init, initargs=(a.app, a.data)) as pool:
            for i, (_, row) in enumerate(pool.imap(run_one, jobs, chunksize=4), 1):
                rows.append(row)
                if i % 200 == 0 or i == len(jobs):
                    print(f"{kind}: {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(a.out, f"{kind}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
