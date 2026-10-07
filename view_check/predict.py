"""X-ray check for the app: says what kind of X-ray was uploaded and which tools can read it.

Needs only numpy, Pillow and onnxruntime (no PyTorch), like the flat foot models (flatfoot_app/assistant/flatfoot.py).

    from view_check.predict import ViewCheck
    vc = ViewCheck("flatfoot_app")      # folder with view_check.onnx and view_check.json
    r = vc.predict(Image.open("xray.jpg"))
    r["view"]    # "side", "top_down", "oblique", "not_foot" or "unsure"
    r["probs"]   # {"side": 0.97, "top_down": 0.01, "oblique": 0.01, "not_foot": 0.01}
    r["tools"]   # ["flat_foot", "heel_spur"] for a side view, ["bunion"] for top-down, [] otherwise

CLI: python predict.py --models ../flatfoot_app xray.jpg [more.jpg ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heel_spur.predict import normalise, to_gray  # noqa: E402

TOOLS = {"side": ["flat_foot", "heel_spur"], "top_down": ["bunion"]}


def letterbox(gray, size):
    h, w = gray.shape
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.zeros((size, size), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(Image.fromarray(gray).resize((nw, nh), Image.BILINEAR))
    return canvas


class ViewCheck:
    def __init__(self, folder):
        meta = json.load(open(os.path.join(folder, "view_check.json"), encoding="utf-8"))
        self.classes, self.size, self.min_conf = meta["classes"], meta["size"], meta["min_confidence"]
        self.sess = ort.InferenceSession(os.path.join(folder, "view_check.onnx"), providers=["CPUExecutionProvider"])

    def predict(self, img):
        x = normalise(letterbox(to_gray(img), self.size))
        p = self.sess.run(None, {"image": x})[0][0]
        k = int(p.argmax())
        view = self.classes[k] if p[k] >= self.min_conf else "unsure"
        return {"view": view, "probs": {c: float(v) for c, v in zip(self.classes, p)},
                "tools": TOOLS.get(view, [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "flatfoot_app"))
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    vc = ViewCheck(a.models)
    for path in a.images:
        r = vc.predict(Image.open(path))
        print(path, r["view"], {k: round(v, 3) for k, v in r["probs"].items()}, r["tools"])


if __name__ == "__main__":
    main()
