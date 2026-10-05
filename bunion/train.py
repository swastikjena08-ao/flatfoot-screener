"""Train the bunion angle point model and score HVA/IMA.

Usage:
  python prep.py --out work/cache.npz
  python train.py --cache work/cache.npz --weights work/resnet18_imagenet.pth --out work/run1
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import N_PTS, STRIDE, angles
from model import PointNet, soft_argmax

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def to_tensor(imgs):
    x = torch.from_numpy(imgs).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    m = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - m) / s


def augment(img, pts, rng):
    """Random rotation, scale and shift (applied to image and points) plus contrast changes."""
    ang = rng.uniform(-15, 15)
    sc = rng.uniform(0.85, 1.15)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-16, 16, 2)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    pts = np.c_[pts, np.ones(len(pts))] @ M.T
    f = img.astype(np.float32) / 255.0
    f = np.clip(f ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.2))
    return (f * 255).astype(np.uint8), pts.astype(np.float32)


def heatmap_targets(pts, h, w, sigma=1.5):
    """Normalized Gaussian per point on the heatmap grid, for the auxiliary loss."""
    ys = (torch.arange(h, device=pts.device).float() + 0.5) * STRIDE
    xs = (torch.arange(w, device=pts.device).float() + 0.5) * STRIDE
    dx = (xs.view(1, 1, 1, w) - pts[..., 0].view(*pts.shape[:2], 1, 1)) / STRIDE
    dy = (ys.view(1, 1, h, 1) - pts[..., 1].view(*pts.shape[:2], 1, 1)) / STRIDE
    g = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    return g / g.sum((-1, -2), keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def predict(model, imgs, bs=32):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(imgs), bs):
        out.append(soft_argmax(model(to_tensor(imgs[i:i + bs]).to(dev))).cpu().numpy())
    return np.concatenate(out)


def score(pred, meta):
    """Angle errors against the dataset's own HVA/IMA labels."""
    a = np.array([angles(p) for p in pred])
    eh, ei = np.abs(a[:, 0] - meta.HVA.values), np.abs(a[:, 1] - meta.IMA.values)
    return dict(n=len(meta),
                hva_mae=float(np.mean(eh)), hva_within3=float(np.mean(eh <= 3)),
                ima_mae=float(np.mean(ei)), ima_within3=float(np.mean(ei <= 3)),
                hva_pred=a[:, 0].tolist(), ima_pred=a[:, 1].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--weights", default=None, help='local .pth, or "imagenet" to download')
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="train on only this many feet (smoke test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    z = np.load(a.cache)
    imgs, pts = z["imgs"], z["pts"]
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    tr = np.where(meta.split == "train")[0]
    va = np.where(meta.split == "val")[0]
    if a.limit:
        tr, va = tr[:a.limit], va[:max(8, a.limit // 4)]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNet(a.weights, a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    hm_h, hm_w = imgs.shape[1] // STRIDE, imgs.shape[2] // STRIDE

    best, log = 1e9, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(tr)
        for i in range(0, len(order), a.bs):
            idx = order[i:i + a.bs]
            batch = [augment(imgs[j], pts[j], rng) for j in idx]
            x = to_tensor(np.stack([b[0] for b in batch])).to(dev)
            y = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
            logits = model(x)
            p = soft_argmax(logits)
            l_pts = F.smooth_l1_loss(p, y, beta=2.0)
            logp = F.log_softmax(logits.flatten(2), -1)
            l_hm = -(heatmap_targets(y, hm_h, hm_w).flatten(2) * logp).sum(-1).mean()
            loss = l_pts + 0.5 * l_hm
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        s = score(predict(model, imgs[va]), meta.iloc[va])
        mean_mae = (s["hva_mae"] + s["ima_mae"]) / 2
        row = dict(epoch=ep + 1, loss=tot / len(tr), sec=round(time.time() - t0),
                   **{k: round(v, 3) for k, v in s.items() if not k.endswith("_pred")})
        log.append(row)
        print(json.dumps(row), flush=True)
        if mean_mae < best:
            best = mean_mae
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    pd.DataFrame(log).to_csv(os.path.join(a.out, "log.csv"), index=False)


if __name__ == "__main__":
    main()
