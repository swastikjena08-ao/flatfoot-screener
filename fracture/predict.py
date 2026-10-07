"""Fracture flag for the app: one foot or ankle X-ray in, a result out.

Needs only numpy, Pillow and onnxruntime (no PyTorch), like heel_spur/predict.py.

    from fracture.predict import Fracture
    fr = Fracture("flatfoot_app")      # folder with fracture.onnx and fracture.json (export.py)
    r = fr.predict(Image.open("xray.jpg"))
    r["p"]       # fracture probability, 0..1
    r["flag"]    # True when p reaches the threshold picked on the validation split
    r["cells"]   # (S/32) x (S/32) map of "fracture here" probabilities over the letterbox square
    r["square"]  # (x0, y0, x1, y1) in image pixels that the map covers (it overhangs the
                 # short side of the X-ray, where the letterbox padding was)

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
    return np.asarray(img.convert("L"))


class Fracture:
    def __init__(self, folder):
        meta = json.load(open(os.path.join(folder, "fracture.json"), encoding="utf-8"))
        self.size, self.threshold = meta["size"], meta["threshold"]
        self.model = ort.InferenceSession(os.path.join(folder, "fracture.onnx"), providers=["CPUExecutionProvider"])

    def predict(self, img):
        gray = to_gray(img)
        h, w = gray.shape
        # same letterbox as training (common.letterbox)
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = np.zeros((self.size, self.size), np.uint8)
        ox, oy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = np.asarray(Image.fromarray(gray).resize((nw, nh), Image.BILINEAR))
        g = canvas.astype(np.float32) / 255
        x = ((g[None] - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)
        # averaged with the mirrored image, as in evaluation
        p1, c1 = self.model.run(None, {"image": x})
        p2, c2 = self.model.run(None, {"image": x[..., ::-1].copy()})
        p = float((p1[0] + p2[0]) / 2)
        cells = (c1[0] + c2[0][:, ::-1]) / 2
        square = (-ox / s, -oy / s, (self.size - ox) / s, (self.size - oy) / s)
        return {"p": p, "flag": p >= self.threshold, "threshold": float(self.threshold),
                "cells": cells.astype(float), "square": tuple(map(float, square))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "flatfoot_app"))
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    fr = Fracture(a.models)
    for path in a.images:
        r = fr.predict(Image.open(path))
        print(path, "FLAG" if r["flag"] else "-", round(r["p"], 3))
