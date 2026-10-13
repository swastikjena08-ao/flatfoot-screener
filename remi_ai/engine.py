"""Runs the trained models in ai_models/ on one X-ray.

Each model only runs on the X-ray view it was trained for:
    side (lateral)   -> flat foot, heel spur
    top-down (AP)    -> bunion angles
    anything else    -> nothing, with the reason recorded

The view comes from the X-ray check model (ai_models/view_check) when its weights are
installed; otherwise the doctor picks it when adding the X-ray.
"""
import os
import sys

import numpy as np
from PIL import Image

# Repo root when run from source; the bundle folder when packaged with PyInstaller.
REPO = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("REMI_MODELS", os.path.join(REPO, "ai_models"))

VIEWS = {
    "side": "Side view",
    "top_down": "Top-down view",
    "oblique": "Angled view",
    "not_foot": "Not a foot X-ray",
    "unsure": "Unclear view",
}

# Which checks fit which view.
ROUTES = {"side": ["flat_foot", "heel_spur"], "top_down": ["bunion"]}

CHECKS = {
    "flat_foot": {"name": "Flat foot", "view": "side",
                  "score": "93.8% correct on 211 unseen X-rays"},
    "heel_spur": {"name": "Heel spur", "view": "side",
                  "score": "97.5% correct on 595 unseen X-rays"},
    "bunion": {"name": "Bunion angles", "view": "top_down",
               "score": "Within 3° of doctors: HVA 85.0%, IMA 95.6% (160 feet)"},
}

FLAT_BELOW = 17.7   # heel bone angle under this = flat foot (landmarks.json threshold)
HIGH_ABOVE = 30.0   # over this = high arch


def flat_category(angle):
    if angle is None:
        return "Not measured"
    if angle < FLAT_BELOW:
        return "Flat foot"
    if angle > HIGH_ABOVE:
        return "High arch"
    return "Normal arch"


def bunion_grade(hva, ima):
    """Common grading: HVA <15 normal, 15-20 mild, 20-40 moderate, >40 severe;
    IMA <9 normal, 9-11 mild, 11-16 moderate, >16 severe. The worse of the two wins."""
    h = 0 if hva < 15 else 1 if hva <= 20 else 2 if hva <= 40 else 3
    i = 0 if ima < 9 else 1 if ima <= 11 else 2 if ima <= 16 else 3
    return ["No bunion", "Mild bunion", "Moderate bunion", "Severe bunion"][max(h, i)]


def _round_map(a, digits=3):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


class Engine:
    def __init__(self, models_dir=MODELS_DIR):
        self.dir = models_dir
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        self._models = {}
        self.errors = {}

    # ---- loading -----------------------------------------------------------------
    def _load(self, key):
        if key in self._models:
            return self._models[key]
        model = None
        try:
            folder = os.path.join(self.dir, key)
            if key == "flat_foot":
                from flat_foot.predict import FlatFoot
                model = FlatFoot(folder)
            elif key == "heel_spur":
                from heel_spur.predict import HeelSpur
                model = HeelSpur(folder)
            elif key == "bunion":
                from bunion.predict import Bunion
                model = Bunion(folder)
            elif key == "view_check":
                from view_check.predict import ViewCheck
                model = ViewCheck(folder)
        except Exception as e:  # missing weights or packages: the check is reported as unavailable
            self.errors[key] = f"{type(e).__name__}: {e}"
            model = None
        self._models[key] = model
        return model

    def available(self, key):
        return self._load(key) is not None

    def status(self):
        """{key: True/False} for every model, for the settings and assistant panels."""
        return {k: self.available(k) for k in ("view_check", *CHECKS)}

    # ---- X-ray type check ------------------------------------------------------------
    def detect_view(self, img):
        """(view, confidence) from the X-ray check, or (None, None) if it isn't installed."""
        vc = self._load("view_check")
        if vc is None:
            return None, None
        r = vc.predict(img)
        conf = max(r["probs"].values())
        return r["view"], conf

    # ---- analysis --------------------------------------------------------------------
    def analyse(self, path, view, progress=None):
        """Run only the checks that fit `view`. Returns a JSON-safe dict."""
        img = Image.open(path)
        img.load()
        out = {"view": view, "checks": {}, "skipped": {}}
        wanted = ROUTES.get(view, [])
        for key, info in CHECKS.items():
            if key not in wanted:
                need = VIEWS[info["view"]].lower()
                out["skipped"][key] = (f"Not run: needs a {need}" if view in ROUTES
                                       else f"Not run: {VIEWS.get(view, view).lower()}")
                continue
            model = self._load(key)
            if model is None:
                out["skipped"][key] = "Model files not installed"
                continue
            if progress:
                progress(info["name"])
            try:
                out["checks"][key] = getattr(self, "_run_" + key)(model, img)
            except Exception as e:
                out["skipped"][key] = f"Could not read this X-ray ({type(e).__name__})"
        return out

    def _run_flat_foot(self, model, img):
        r = model.predict(img)
        angle = None if r["angle"] is None else round(float(r["angle"]), 1)
        cat = flat_category(angle)
        if angle is not None and not (0 <= angle <= 45):
            # outside anything a standing foot shows: the heel points were probably misplaced
            cat = "Not measured"
        return {
            "angle": angle,
            "category": cat,
            "p_flat": round(float(r["p_flat"]), 4),
            "left": bool(r["left"]),
            "points": {k: [float(v[0]), float(v[1])] for k, v in (r["points"] or {}).items()},
            "heatmap": _round_map(r["heatmap"]),
            "severity": "ok" if cat in ("Normal arch", "Not measured") else "warn",
        }

    def _run_heel_spur(self, model, img):
        r = model.predict(img)
        label = r["label"]
        headline = {"normal": "No heel spur", "spur": "Heel spur found",
                    "sever": "Heel growth plate irritation (Sever's)"}[label]
        return {
            "label": label,
            "headline": headline,
            "probs": {k: round(v, 4) for k, v in r["probs"].items()},
            "box": list(r["box"]),
            "crop_box": list(r["crop_box"]),
            "heatmap": _round_map(r["heatmap"]),
            "severity": "ok" if label == "normal" else "alert",
        }

    def _run_bunion(self, model, img):
        gray = np.asarray(img.convert("L"))
        if model.finder is not None:
            feet = model.predict(gray)
        else:
            h, w = gray.shape
            feet = model.predict(gray, boxes={"right": (0.0, 0.0, float(w), float(h))})
        if not feet:
            raise ValueError("no foot found")
        out = []
        for f in feet:
            grade = bunion_grade(f["hva"], f["ima"])
            out.append({"side": f["side"], "hva": round(float(f["hva"]), 1), "ima": round(float(f["ima"]), 1),
                        "points": [[float(x), float(y)] for x, y in f["points"]], "box": list(f["box"]),
                        "grade": grade,
                        "severity": "ok" if grade == "No bunion" else "warn" if grade == "Mild bunion" else "alert"})
        return {"feet": out, "severity": max((f["severity"] for f in out), key=["ok", "warn", "alert"].index)}


def findings(results):
    """Short list of (title, detail, severity) for a result dict, for lists and the report."""
    out = []
    c = results.get("checks", {})
    if "flat_foot" in c:
        f = c["flat_foot"]
        measured = f["angle"] is not None and f["category"] != "Not measured"
        detail = f"Heel bone angle {f['angle']}°" if measured else "Angle could not be measured"
        out.append((f["category"], detail, f["severity"]))
    if "heel_spur" in c:
        h = c["heel_spur"]
        out.append((h["headline"], f"{h['probs'][h['label']] * 100:.0f}% model confidence", h["severity"]))
    if "bunion" in c:
        for f in c["bunion"]["feet"]:
            out.append((f"{f['grade']} ({f['side']})", f"HVA {f['hva']}°, IMA {f['ima']}°", f["severity"]))
    return out
