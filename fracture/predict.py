"""Fracture flag for the app: one foot or ankle X-ray in, a result out.

Needs only numpy, Pillow and onnxruntime (no PyTorch), like flatfoot_app.py.

    from fracture.predict import Fracture
    fr = Fracture("flatfoot_app")          # folder with fracture.onnx, fracture.json
    r = fr.predict(Image.open("xray.jpg"))
    r["p"]        # 0.83, fracture probability
    r["flag"]     # True when p >= the threshold picked on the validation split
    r["heatmap"]  # 16x16 map (at 512 px) of "fracture here", 0..1
    r["map_box"]  # (x0, y0, x1, y1) in image pixels the heatmap covers (the letterbox square)

CLI: python predict.py --models ../flatfoot_app xray.jpg
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_gray(img):
    """8-bit grayscale, with EXIF rotation applied and 16-bit / float X-rays rescaled."""
    img = ImageOps.exif_transpose(img)
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(rgba, mask=rgba.split()[3])
        img = bg
    return np.asarray(img.convert("L"))


def letterbox(gray, size):
    """Same as common.letterbox: whole image fitted into a size x size square."""
    h, w = gray.shape
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.zeros((size, size), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(Image.fromarray(gray).resize((nw, nh), Image.BILINEAR))
    return canvas, s, ox, oy


def normalise(u8):
    g = u8.astype(np.float32) / 255
    return ((g[None] - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)


class Fracture:
    def __init__(self, folder):
        meta = json.load(open(os.path.join(folder, "fracture.json"), encoding="utf-8"))
        self.size, self.threshold = meta["size"], meta["threshold"]
        self.model = ort.InferenceSession(os.path.join(folder, "fracture.onnx"), providers=["CPUExecutionProvider"])

    def predict(self, img):
        gray = img if isinstance(img, np.ndarray) else to_gray(img)
        canvas, s, ox, oy = letterbox(gray, self.size)
        x = normalise(canvas)
        p1, c1 = self.model.run(None, {"image": x})
        p2, c2 = self.model.run(None, {"image": x[..., ::-1].copy()})  # mirrored, as in evaluation
        p = float((p1[0] + p2[0]) / 2)
        heat = (c1[0] + c2[0][:, ::-1]) / 2
        box = (-ox / s, -oy / s, (self.size - ox) / s, (self.size - oy) / s)
        return {"p": p, "flag": p >= self.threshold, "threshold": self.threshold,
                "heatmap": heat, "map_box": tuple(map(float, box))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "flatfoot_app"))
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    fr = Fracture(a.models)
    for path in a.images:
        r = fr.predict(Image.open(path))
        print(path, "FLAG" if r["flag"] else "-", round(r["p"], 3))
