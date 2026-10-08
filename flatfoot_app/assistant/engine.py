"""Runs the three models on one X-ray. No UI here, so it is safe off the main thread
and easy to test.

* Side view: flat foot (flatfoot_app.py: classifier + L10/L11 heel bone angle) and
  heel spur (ai_models/heel_spur/predict.py).
* Top-down view: bunion angles (ai_models/bunion/predict.py), one result per foot.

Each model's own predict code is imported and called unchanged, so the app gives the
same answers as the model's training repo (tools/check_models.py checks this).
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)            # flatfoot_app/
REPO = os.path.dirname(APP_DIR)            # repo root
MODELS_DIR = os.path.join(REPO, "ai_models")  # holds heel_spur/ and bunion/
for p in (APP_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import flatfoot_app as ff  # noqa: E402  the original app module: models, preprocessing, angle maths


def resource(*parts):
    """Bundled file: next to the .exe when frozen with PyInstaller, else in flatfoot_app/."""
    base = getattr(sys, "_MEIPASS", APP_DIR)
    return os.path.join(base, *parts)


def load_scores():
    with open(resource("assistant", "model_scores.json"), encoding="utf-8") as f:
        return json.load(f)


def to_gray(img):
    """8-bit grayscale numpy array, same as heel_spur.predict.to_gray."""
    from heel_spur.predict import to_gray as tg
    return tg(img)


def open_xray(path):
    img = Image.open(path)
    img.load()
    return ff.normalize_input(img)


def guess_view(img):
    """Side views are usually wider than tall, top-down views taller than wide. Only a default."""
    return "side" if img.width >= img.height else "top"


class Engine:
    def __init__(self, bunion_dir=None):
        self.status = {}
        self.flat_clf = self.flat_lm = self.spur = self.bunion = None
        try:
            import onnxruntime as ort
            self.flat_clf = ort.InferenceSession(resource("model.onnx"), providers=["CPUExecutionProvider"])
            self.flat_lm = ff.Landmarks(resource("landmarks.onnx"), resource("landmarks.json"))
            self.status["flat"] = "ready"
        except Exception as e:  # noqa: BLE001
            self.status["flat"] = f"not loaded: {e}"
        try:
            from heel_spur.predict import HeelSpur
            self.spur = HeelSpur(resource())
            self.status["spur"] = "ready"
        except Exception as e:  # noqa: BLE001
            self.status["spur"] = f"not loaded: {e}"
        bunion_dir = bunion_dir or resource("models", "bunion")
        try:
            from bunion.predict import Bunion
            self.bunion = Bunion(bunion_dir)
            self.status["bunion"] = "ready" if self.bunion.finder else "ready (no foot finder: you pick the foot)"
        except FileNotFoundError:
            self.status["bunion"] = f"not installed: copy the bunion models into {bunion_dir}"
        except Exception as e:  # noqa: BLE001
            self.status["bunion"] = f"not loaded: {e}"
        self.lm_meta = self.flat_lm.meta if self.flat_lm else {"threshold_deg": 17.7}

    def ready(self, name):
        return self.status.get(name, "").startswith("ready")

    # ---------- side view ----------

    def analyse_side(self, img, side=None):
        """side: None (auto), 'left' or 'right'."""
        out = {"view": "side"}
        if self.flat_clf is not None:
            choice = ff.AUTO_SIDE if side is None else (ff.LEFT_SIDE if side == "left" else ff.RIGHT_SIDE)
            a = ff.analyse_xray(img, choice, self.flat_clf, self.flat_lm)
            out["flat"] = {"p": a.p, "cam": a.cam.tolist(), "left": a.left, "how": a.how,
                           "points": {k: list(v) for k, v in a.points.items()} if a.points else None}
            out["side"] = "left" if a.left else "right"
        if self.spur is not None:
            r = self.spur.predict(img)
            out["spur"] = {"probs": r["probs"], "label": r["label"], "box": list(r["box"]),
                           "crop_box": list(r["crop_box"]), "heatmap": np.asarray(r["heatmap"]).tolist()}
        return out

    # ---------- top-down view ----------

    def analyse_top(self, img, side=None):
        """side: None to let the foot finder find both feet, or 'left'/'right' for a one-foot X-ray."""
        out = {"view": "top", "feet": []}
        if self.bunion is None:
            return out
        gray = to_gray(img)
        h, w = gray.shape
        whole = (0.0, 0.0, float(w), float(h))
        boxes = None
        if self.bunion.finder is None:
            boxes = {side or "right": whole}
        elif side:
            # the person named the foot: still box it with the finder, so a wide X-ray isn't
            # measured with all its background (one foot found on the wrong side = that foot)
            found = self.bunion.find_feet(gray)
            box = found.get(side) or (next(iter(found.values())) if len(found) == 1 else whole)
            boxes = {side: box}
        feet = self.bunion.predict(gray, boxes)
        out["feet"] = feet
        out["side"] = feet[0]["side"] if len(feet) == 1 else ("both" if feet else "")
        return out

    def analyse(self, img, view, side=None):
        return self.analyse_side(img, side) if view == "side" else self.analyse_top(img, side)


# ---------- reading results (with the doctor's point edits applied) ----------

def arch_angle(results, edits):
    f = (results or {}).get("flat")
    if not f or not f.get("points"):
        return None
    pts = (edits or {}).get("arch") or f["points"]
    return ff.line_angle(pts["L10"], pts["L11"], f["left"])


def bunion_feet(results, edits):
    """Feet with angles recomputed from edited points where the doctor moved them."""
    from bunion.predict import angles
    out = []
    for i, foot in enumerate((results or {}).get("feet", [])):
        pts = ((edits or {}).get("bunion") or {}).get(str(i)) or foot["points"]
        hva, ima = angles(np.asarray(pts, float))
        out.append({**foot, "points": pts, "hva": hva, "ima": ima, "edited": pts is not foot["points"]})
    return out


def arch_class(angle, threshold=17.7):
    if angle is None:
        return None, ""
    if angle < threshold:
        return "Flat foot range", "amber"
    if angle > 30:
        return "High arch flag", "red"
    return "Normal arch", "green"


def bunion_class(hva, ima):
    if hva > 40 or ima > 16:
        return "Severe bunion", "red"
    if hva >= 20 or ima > 11:
        return "Moderate bunion", "amber"
    if hva >= 15 or ima >= 9:
        return "Mild bunion", "amber"
    return "No bunion", "green"


def spur_class(spur):
    if not spur:
        return None, ""
    p = spur["probs"]
    if spur["label"] == "spur":
        return "Heel spur likely", "red"
    if spur["label"] == "sever":
        return "Child's heel (Sever's)", "amber"
    return ("Possible heel spur", "amber") if p["spur"] >= 0.3 else ("No heel spur", "green")


def findings(results, edits, threshold=17.7):
    """Short labels for lists and the report: [(text, colour)]."""
    out = []
    if not results:
        return out
    a = arch_angle(results, edits)
    if a is not None:
        t, c = arch_class(a, threshold)
        out.append((f"{t} {a:.1f}°", c))
    t, c = spur_class(results.get("spur"))
    if t and c != "green":
        out.append((t, c))
    for f in bunion_feet(results, edits):
        t, c = bunion_class(f["hva"], f["ima"])
        out.append((f"{t} ({f['side']})" if len(results.get("feet", [])) > 1 else t, c))
    return out


def gray_of(img):
    return np.asarray(ImageOps.grayscale(ff.flatten_alpha(img)))
