"""Second pass: cut a zoomed forefoot crop around the 6 points, from the original X-ray.

Train feet are cropped around their true points (training adds random shifts).
Val, test and unsplit feet are cropped around the first-pass model's predicted
points, the same way a new X-ray would be handled.

Usage: python zoom_prep.py --data /path/to/hvangleest --cache work/cache.npz \
           --ckpt work/run1/best.pt --out work/zoom_cache.npz
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch

from common import canonical_points
from model import PointNet
from train import predict

Z_W, Z_H = 256, 384  # zoomed input (width, height)
PAD = 1.6            # region size relative to the points' bounding box


def region_map(pts_img, mirror, Z_W=Z_W, Z_H=Z_H):
    """Affine map from original-image px to the zoomed crop, framing the given points."""
    x1, y1 = pts_img.min(0)
    x2, y2 = pts_img.max(0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = max((y2 - y1) * PAD, (x2 - x1) * PAD * Z_H / Z_W)
    s = Z_H / h
    if mirror:
        return np.array([[-s, 0, Z_W / 2 + s * cx], [0, s, Z_H / 2 - s * cy]])
    return np.array([[s, 0, Z_W / 2 - s * cx], [0, s, Z_H / 2 - s * cy]])


def apply(M, p):
    return np.c_[p, np.ones(len(p))] @ M.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True, help="first-pass cache from prep.py")
    ap.add_argument("--ckpt", required=True, help="first-pass checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18", help="must match the checkpoint")
    ap.add_argument("--width", type=int, default=Z_W)
    ap.add_argument("--height", type=int, default=Z_H)
    a = ap.parse_args()

    z = np.load(a.cache)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    model = PointNet(arch=a.arch)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    if torch.cuda.is_available():
        model = model.cuda()
    pred1 = predict(model, z["imgs"])

    imgs, pts, rows = [], [], []
    cur, g = None, None
    for i, r in meta.iterrows():
        if r.filename != cur:
            cur = r.filename
            g = cv2.imread(os.path.join(a.data, "HVAngleEst", "images", r.filename), cv2.IMREAD_GRAYSCALE)
        M1 = np.array(r.M)
        inv1 = cv2.invertAffineTransform(M1)
        true_img = apply(inv1, z["pts"][i])
        frame = true_img if r.split == "train" else apply(inv1, pred1[i])
        M2 = region_map(frame, r.side == "left", a.width, a.height)
        crop = cv2.warpAffine(g, M2, (a.width, a.height), flags=cv2.INTER_AREA, borderValue=0)
        imgs.append(crop)
        pts.append(canonical_points(apply(M2, true_img)))
        rows.append({**r.to_dict(), "M": M2.tolist(), "M1": r.M})
    out = pd.DataFrame(rows)
    np.savez_compressed(a.out, imgs=np.stack(imgs), pts=np.stack(pts).astype(np.float32),
                        pred1=pred1.astype(np.float32), meta=out.to_json(orient="records"))
    print("saved", a.out, len(imgs))


if __name__ == "__main__":
    main()
