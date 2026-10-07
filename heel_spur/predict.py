"""Heel spur check for the app: one lateral foot X-ray in, a result out.

Needs only numpy, Pillow and onnxruntime (no PyTorch), like the flat foot models (flatfoot_app/assistant/flatfoot.py).

    from heel_spur.predict import HeelSpur
    hs = HeelSpur("flatfoot_app")          # folder with heel_finder.onnx, heel_spur.onnx, heel_spur.json
    r = hs.predict(Image.open("xray.jpg"))
    r["probs"]    # {"normal": 0.03, "spur": 0.95, "sever": 0.02}
    r["label"]    # "spur"
    r["box"]      # heel box (x0, y0, x1, y1) in image pixels
    r["heatmap"]  # 7x7 map over the heel crop for the predicted class, 0..1
    r["crop_box"] # square (x0, y0, x1, y1) in image pixels the heatmap covers

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


def normalise(u8):
    g = u8.astype(np.float32) / 255
    return ((g[None] - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)


class HeelSpur:
    def __init__(self, folder):
        meta = json.load(open(os.path.join(folder, "heel_spur.json"), encoding="utf-8"))
        self.classes, self.size = meta["classes"], meta["finder_size"]
        self.crop_size, self.pad = meta["crop_size"], meta["crop_pad"]
        opts = dict(providers=["CPUExecutionProvider"])
        self.finder = ort.InferenceSession(os.path.join(folder, "heel_finder.onnx"), **opts)
        self.spur = ort.InferenceSession(os.path.join(folder, "heel_spur.onnx"), **opts)

    def find_heel(self, gray):
        h, w = gray.shape
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = np.zeros((self.size, self.size), np.uint8)
        ox, oy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = np.asarray(Image.fromarray(gray).resize((nw, nh), Image.BILINEAR))
        b = self.finder.run(None, {"image": normalise(canvas)})[0][0] * self.size
        return ((b[0] - ox) / s, (b[1] - oy) / s, (b[2] - ox) / s, (b[3] - oy) / s)

    def crop(self, gray, box, out):
        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0) * self.pad
        a, b = (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2
        img = Image.fromarray(gray).transform((out, out), Image.AFFINE,
                                              [side / out, 0, a, 0, side / out, b], resample=Image.BILINEAR)
        return np.asarray(img), (a, b, a + side, b + side)

    def predict(self, img):
        gray = to_gray(img)
        box = self.find_heel(gray)
        # same as training: crop at 256, keep the central 224
        big, (a, b, a1, _) = self.crop(gray, box, 256)
        o = (256 - self.crop_size) // 2
        x = normalise(big[o:o + self.crop_size, o:o + self.crop_size])
        probs, cam = self.spur.run(None, {"image": x})
        probs2, _ = self.spur.run(None, {"image": x[..., ::-1].copy()})  # mirrored, as in evaluation
        p = (probs[0] + probs2[0]) / 2
        k = int(p.argmax())
        heat = np.maximum(cam[0, k], 0)
        heat = heat / (heat.max() or 1)
        step = (a1 - a) / 256
        crop_box = (a + o * step, b + o * step, a + (o + self.crop_size) * step, b + (o + self.crop_size) * step)
        return {"probs": dict(zip(self.classes, map(float, p))), "label": self.classes[k],
                "box": tuple(map(float, box)), "heatmap": heat, "crop_box": tuple(map(float, crop_box))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "flatfoot_app"))
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    hs = HeelSpur(a.models)
    for path in a.images:
        r = hs.predict(Image.open(path))
        print(path, r["label"], {k: round(v, 3) for k, v in r["probs"].items()})
