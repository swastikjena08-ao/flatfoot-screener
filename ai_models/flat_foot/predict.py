"""Flat foot models: classifier + heel bone angle (L10/L11 landmarks). No UI.

Model code copied from flatfoot_app/flatfoot_app.py (minus the window), so it gives the
same answers as the app. Two ONNX models, both in flatfoot_app/:
  * model.onnx      ResNet-18 classifier: flat foot probability + 7x7 heatmap.
  * landmarks.onnx  ResNet-18 heatmap model that finds L10 and L11; the L10->L11 angle
                    against the horizontal is the heel bone angle (flat if under 17.7 deg).

Both were trained (Google Colab) on right-foot side X-rays only, toes pointing left.
Left feet are mirrored before inference and the points mapped back.

CLI: python predict.py --models ../../flatfoot_app xray.jpg [more.jpg ...]
"""
import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

SIZE, THRESH, BORDER = 224, 0.5, (0.35, 0.65)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalize_input(img):
    """Upright, 8-bit image. Applies EXIF rotation (phone photos) and rescales
    16-bit / float X-rays (TIFF exports), which Pillow would otherwise clip to white."""
    img = ImageOps.exif_transpose(img)
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        a = np.clip((a - lo) / (hi - lo if hi > lo else 1), 0, 1)
        img = Image.fromarray((a * 255).astype(np.uint8), "L")
    return img


def flatten_alpha(img):
    rgba = img.convert("RGBA")
    rgb = Image.new("RGB", img.size, (0, 0, 0))
    rgb.paste(rgba, mask=rgba.split()[3])
    return rgb


# ---------- classifier ----------

def to_tensor(img):
    d = np.asarray(flatten_alpha(img).resize((SIZE, SIZE), Image.Resampling.LANCZOS), np.float32)
    gray = (0.299 * d[..., 0] + 0.587 * d[..., 1] + 0.114 * d[..., 2]) / 255
    t = (gray[None] - MEAN[:, None, None]) / STD[:, None, None]
    return t[None].astype(np.float32)


def analyse(session, img):
    out = session.run(None, {"image": to_tensor(img)})
    res = dict(zip([o.name for o in session.get_outputs()], out))
    p = float(res["prob"].reshape(-1)[0])
    cam = res["cam"].reshape(res["cam"].shape[-2:])
    sign = 1 if p >= THRESH else -1
    cam = np.maximum(0, sign * cam)
    cam = cam / (cam.max() or 1)
    return p, cam


# ---------- landmarks ----------

class Landmarks:
    def __init__(self, onnx_path, meta_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.meta = json.load(open(meta_path, encoding="utf-8"))
        self.w, self.h, self.stride = self.meta["input_w"], self.meta["input_h"], self.meta["stride"]

    def letterbox(self, w, h):
        s = min(self.w / w, self.h / h)
        return s, (self.w - w * s) / 2, (self.h - h * s) / 2

    def predict(self, img):
        """Returns ({'L10': (x, y), 'L11': (x, y)} in image pixels, confidence).

        The model was trained on right feet only (toes pointing left), so pass
        left feet mirrored. Confidence is the weaker of the two heatmap peaks.
        """
        gray = ImageOps.grayscale(flatten_alpha(img))
        s, ox, oy = self.letterbox(*gray.size)
        # same affine letterbox as training
        canvas = gray.transform((self.w, self.h), Image.AFFINE,
                                [1 / s, 0, -ox / s, 0, 1 / s, -oy / s], resample=Image.BILINEAR)
        g = np.asarray(canvas, np.float32) / 255
        x = ((g[None] - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)
        hm = self.session.run(None, {"image": x})[0][0]
        conf = float(hm.reshape(len(hm), -1).max(1).min())
        pts = {}
        for name, h in zip(self.meta["points"], hm):
            yy, xx = np.unravel_index(np.argmax(h), h.shape)
            y0, y1 = max(0, yy - 2), min(h.shape[0], yy + 3)
            x0, x1 = max(0, xx - 2), min(h.shape[1], xx + 3)
            w = np.clip(h[y0:y1, x0:x1], 0, None)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            sw = w.sum() or 1
            px, py = (gx * w).sum() / sw * self.stride, (gy * w).sum() / sw * self.stride
            pts[name] = ((px - ox) / s, (py - oy) / s)
        return pts, conf


def mirror(img):
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def line_angle(p10, p11, left=False):
    """Angle of the L10→L11 line against the horizontal, in degrees (positive = L11 lower).

    Measured in right-foot orientation, so a left foot gives the same value as its mirror image.
    """
    dx = p11[0] - p10[0]
    return math.degrees(math.atan2(p11[1] - p10[1], -dx if left else dx))


AUTO_SIDE, RIGHT_SIDE, LEFT_SIDE = 0, 1, 2  # order of the "Foot side" combo box


@dataclass
class Analysis:
    p: float
    cam: np.ndarray
    points: dict | None  # {'L10': (x, y), 'L11': (x, y)} on the original image, or None
    left: bool
    how: str


def analyse_xray(img, side, session, landmarks):
    """Classifier + landmarks for one X-ray. No UI; safe to run off the main thread.

    Both models were trained on right feet (toes pointing left), so a left foot is
    mirrored first and the heatmap and points are mapped back onto `img`.
    """
    pts = None
    if side == AUTO_SIDE and landmarks is not None:
        pts, conf_r = landmarks.predict(img)
        pts_l, conf_l = landmarks.predict(mirror(img))
        left = conf_l > conf_r
        if left:
            pts = pts_l
        how = "detected automatically"
    else:
        left = side == LEFT_SIDE
        how = "set by you" if side != AUTO_SIDE else "assumed (landmark model not found)"
    work = mirror(img) if left else img
    if landmarks is not None and pts is None:
        pts, _ = landmarks.predict(work)
    if pts is not None:
        # Classify the foot only, framed like the training X-rays, so a calibration ball,
        # the other leg or a long shin can't sway the verdict or pull the heatmap off the foot.
        box = foot_box(pts, work.size)
        p, cam = analyse(session, work.crop(box))
        cam = cam_on_image(cam, box, *work.size)
    else:
        p, cam = analyse(session, work)
    if left:
        cam = cam[:, ::-1]
        pts = {k: (img.width - 1 - x, y) for k, (x, y) in pts.items()} if pts else pts
    return Analysis(p, cam, pts, left, how)


def foot_box(pts, size):
    """Crop (x0, y0, x1, y1) that frames the foot like the Pesplanus training X-rays.

    In those, measured on the 163 hand-labelled ones, the L10–L11 line's midpoint sits 76%
    across and 83% down, and the image is about 8 line-lengths wide and 5.4 tall.
    Parts outside the X-ray come out black, like the training images' borders.
    """
    (x1, y1), (x2, y2) = pts["L10"], pts["L11"]
    d = max(math.hypot(x2 - x1, y2 - y1), 0.02 * max(size))
    mx, my, w, h = (x1 + x2) / 2, (y1 + y2) / 2, 8 * d, 5.4 * d
    return tuple(round(v) for v in (mx - 0.76 * w, my - 0.83 * h, mx + 0.24 * w, my + 0.17 * h))


def cam_on_image(cam, box, w, h, grid=64):
    """Place a heatmap computed on crop `box` onto a coarse grid over the whole w x h image
    (zero outside the crop), so callers can keep stretching it over the full X-ray."""
    s = grid / max(w, h)
    gw, gh = max(2, round(w * s)), max(2, round(h * s))
    ch, cw = cam.shape
    x0, y0, x1, y1 = box
    xs = ((np.arange(gw) + 0.5) / s - x0) / (x1 - x0)  # cell centres, 0..1 inside the crop
    ys = ((np.arange(gh) + 0.5) / s - y0) / (y1 - y0)
    inside = ((ys >= 0) & (ys <= 1))[:, None] & ((xs >= 0) & (xs <= 1))[None, :]
    sx = np.clip(xs * cw - 0.5, 0, cw - 1)
    sy = np.clip(ys * ch - 0.5, 0, ch - 1)
    ix0, iy0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    ix1, iy1 = np.minimum(cw - 1, ix0 + 1), np.minimum(ch - 1, iy0 + 1)
    fx, fy = (sx - ix0)[None, :], (sy - iy0)[:, None]
    out = ((cam[np.ix_(iy0, ix0)] * (1 - fx) + cam[np.ix_(iy0, ix1)] * fx) * (1 - fy)
           + (cam[np.ix_(iy1, ix0)] * (1 - fx) + cam[np.ix_(iy1, ix1)] * fx) * fy)
    return np.where(inside, out, 0).astype(np.float32)


class FlatFoot:
    """Loads model.onnx, landmarks.onnx and landmarks.json from one folder."""

    def __init__(self, models_dir):
        self.session = ort.InferenceSession(os.path.join(models_dir, "model.onnx"),
                                            providers=["CPUExecutionProvider"])
        self.landmarks = Landmarks(os.path.join(models_dir, "landmarks.onnx"),
                                   os.path.join(models_dir, "landmarks.json"))
        self.threshold = self.landmarks.meta["threshold_deg"]

    def predict(self, img, side=AUTO_SIDE):
        a = analyse_xray(normalize_input(img), side, self.session, self.landmarks)
        angle = line_angle(a.points["L10"], a.points["L11"], a.left) if a.points else None
        return {"p_flat": a.p, "angle": angle, "flat_by_angle": angle is not None and angle < self.threshold,
                "points": a.points, "left": a.left, "heatmap": a.cam}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "..", "flatfoot_app"))
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    ff = FlatFoot(args.models)
    for path in args.images:
        r = ff.predict(Image.open(path))
        angle = f"{r['angle']:.1f} deg" if r["angle"] is not None else "n/a"
        print(f"{path}: {'left' if r['left'] else 'right'} foot, classifier p(flat)={r['p_flat']:.3f}, "
              f"heel bone angle {angle} -> {'flat foot' if r['flat_by_angle'] else 'normal arch'}")


if __name__ == "__main__":
    main()
