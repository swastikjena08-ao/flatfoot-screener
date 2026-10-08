"""Bunion angles for the app: one front-view (AP) foot X-ray in, HVA and IMA per foot out.

Needs only numpy, OpenCV and onnxruntime (no PyTorch). The crops are made with the
same OpenCV calls as prep.py and zoom_prep.py, so the app sees what the models saw.

    from bunion.predict import Bunion
    b = Bunion("flatfoot_app/models/bunion")
    for foot in b.predict(gray_uint8):            # or b.predict(gray, boxes={"left": (x0, y0, x1, y1)})
        foot["side"], foot["hva"], foot["ima"], foot["points"]   # points: 6 x 2, image pixels

The model folder holds:
    bunion_points.onnx   first pass, whole-foot crop (required)
    bunion_zoom.onnx     second pass, zoomed forefoot crop (optional)
    footfinder.onnx      finds the left/right foot boxes (optional; without it pass `boxes`)
    bunion.json          {"zoom_weight": 0.8} and the test scores to show in the app

CLI: python predict.py --models ../../flatfoot_app/models/bunion xray.jpg
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MARGIN = 0.05    # prep.py: border around the foot box
ZOOM_PAD = 1.6   # zoom_prep.py: region size relative to the points' bounding box
SIDES = ("left", "right")


def to_tensor(u8):
    """N x H x W uint8 -> N x 3 x H x W float32, ImageNet-normalised gray (train.to_tensor)."""
    x = u8.astype(np.float32)[:, None] / 255.0
    return ((x - MEAN[None, :, None, None]) / STD[None, :, None, None]).astype(np.float32)


def apply(M, p):
    return np.c_[p, np.ones(len(p))] @ np.asarray(M).T


def canonical(p):
    """Each line's distal end (smaller y) first, as in common.canonical_points."""
    p = p.copy()
    for i in range(0, 6, 2):
        if p[i, 1] > p[i + 1, 1]:
            p[[i, i + 1]] = p[[i + 1, i]]
    return p


def line_angle(a0, a1, b0, b1):
    v1, v2 = np.subtract(a1, a0), np.subtract(b1, b0)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return float("nan")
    t = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / n))))
    return min(t, 180 - t)


def angles(p):
    """HVA (big toe vs 1st metatarsal) and IMA (1st vs 2nd metatarsal) from 6 points."""
    return line_angle(p[0], p[1], p[2], p[3]), line_angle(p[2], p[3], p[4], p[5])


def crop_map(box, mirror, w, h):
    """prep.crop_foot: affine map from image px to a letterboxed w x h foot crop."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - MARGIN * bw, x2 + MARGIN * bw
    y1, y2 = y1 - MARGIN * bh, y2 + MARGIN * bh
    s = min(w / (x2 - x1), h / (y2 - y1))
    ox, oy = (w - s * (x2 - x1)) / 2, (h - s * (y2 - y1)) / 2
    if mirror:
        return np.array([[-s, 0, w - (ox - s * x1)], [0, s, oy - s * y1]])
    return np.array([[s, 0, ox - s * x1], [0, s, oy - s * y1]])


def region_map(pts, mirror, w, h):
    """zoom_prep.region_map: affine map from image px to a w x h crop framing the points."""
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = h / max((y2 - y1) * ZOOM_PAD, (x2 - x1) * ZOOM_PAD * h / w)
    if mirror:
        return np.array([[-s, 0, w / 2 + s * cx], [0, s, h / 2 - s * cy]])
    return np.array([[s, 0, w / 2 - s * cx], [0, s, h / 2 - s * cy]])


class Bunion:
    def __init__(self, folder):
        opts = dict(providers=["CPUExecutionProvider"])

        def load(name):
            path = os.path.join(folder, name)
            return ort.InferenceSession(path, **opts) if os.path.exists(path) else None

        self.pass1 = load("bunion_points.onnx")
        if self.pass1 is None:
            raise FileNotFoundError(os.path.join(folder, "bunion_points.onnx"))
        self.zoom = load("bunion_zoom.onnx")
        self.finder = load("footfinder.onnx")
        meta_path = os.path.join(folder, "bunion.json")
        self.meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        self.zoom_weight = float(self.meta.get("zoom_weight", 0.8)) if self.zoom else 0.0
        self.size1 = tuple(self.pass1.get_inputs()[0].shape[2:4])          # (h, w)
        self.size2 = tuple(self.zoom.get_inputs()[0].shape[2:4]) if self.zoom else None
        self.size_f = tuple(self.finder.get_inputs()[0].shape[2:4]) if self.finder else None

    def find_feet(self, gray):
        """{"left": box, "right": box} for each foot the finder sees (box in image px)."""
        fh, fw = self.size_f
        H, W = gray.shape
        s = min(fw / W, fh / H)
        nw, nh = max(1, round(W * s)), max(1, round(H * s))
        ox, oy = (fw - nw) // 2, (fh - nh) // 2
        canvas = np.zeros((fh, fw), np.uint8)
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        boxes, present = self.finder.run(None, {"image": to_tensor(canvas[None])})
        out = {}
        for k, side in enumerate(SIDES):
            if present[0, k] >= 0.5:
                out[side] = tuple(float(v) for v in (boxes[0, k] - [ox, oy, ox, oy]) / s)
        return out

    def _points(self, session, gray, M, size):
        h, w = size
        crop = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_AREA, borderValue=0)
        p = session.run(None, {"image": to_tensor(crop[None])})[0][0]
        return apply(cv2.invertAffineTransform(M), p)

    def measure(self, gray, box, side):
        """6 points (image px) and angles for one foot inside `box`."""
        mirror = side == "left"
        h1, w1 = self.size1
        p = p1 = self._points(self.pass1, gray, crop_map(box, mirror, w1, h1), self.size1)
        if self.zoom is not None:
            h2, w2 = self.size2
            p2 = self._points(self.zoom, gray, region_map(p1, mirror, w2, h2), self.size2)
            p = self.zoom_weight * p2 + (1 - self.zoom_weight) * p1
        p = canonical(p)
        hva, ima = angles(p)
        return {"side": side, "box": tuple(map(float, box)), "points": p.tolist(), "hva": hva, "ima": ima}

    def predict(self, gray, boxes=None):
        """gray: H x W uint8. boxes: {"left"/"right": (x0, y0, x1, y1)} or None to use the foot finder."""
        if boxes is None:
            if self.finder is None:
                raise ValueError("No foot finder model: pass the foot box and side.")
            boxes = self.find_feet(gray)
        return [self.measure(gray, boxes[s], s) for s in SIDES if s in boxes]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "flatfoot_app", "models", "bunion"))
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    b = Bunion(a.models)
    for path in a.images:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        for f in b.predict(g):
            print(path, f["side"], f"HVA {f['hva']:.1f}", f"IMA {f['ima']:.1f}")
