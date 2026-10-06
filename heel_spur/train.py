"""Train the heel finder (--task locate) or the spur classifier (--task classify) on CPU or GPU.

Usage:
  python train.py --task locate   --cache work/cache.npz --init ../flatfoot_app/model.onnx --out work/locate
  python train.py --task classify --cache work/cache.npz --init ../flatfoot_app/model.onnx --out work/classify
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import CLASSES, CROP_CACHE, CROP_SIZE, LOC_SIZE, box_iou, resnet18, to_tensor


def photometric(f, rng):
    f = np.clip(f ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.2))
    return f


def aug_locate(img, box, rng):
    """Flip, rotate, scale and shift the whole letterboxed image; the box follows its corners."""
    n = LOC_SIZE
    M = cv2.getRotationMatrix2D((n / 2, n / 2), rng.uniform(-12, 12), rng.uniform(0.85, 1.15))
    M[:, 2] += rng.uniform(-20, 20, 2)
    if rng.random() < 0.5:
        M = M @ np.array([[-1, 0, n], [0, 1, 0], [0, 0, 1]], np.float64)
    img = cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderValue=0)
    b = box * n
    corners = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[0], b[3], 1], [b[2], b[3], 1]]) @ M.T
    nb = np.r_[corners.min(0), corners.max(0)] / n
    f = photometric(img.astype(np.float32) / 255, rng)
    return (f * 255).astype(np.uint8), np.clip(nb, 0, 1).astype(np.float32)


def aug_classify(img, rng):
    """Random rotation, scale and shift (stand-ins for heel finder error), flip, then a 224 crop."""
    n = CROP_CACHE
    M = cv2.getRotationMatrix2D((n / 2, n / 2), rng.uniform(-15, 15), rng.uniform(0.85, 1.15))
    M[:, 2] += rng.uniform(-12, 12, 2)
    img = cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    o = (n - CROP_SIZE) // 2
    f = photometric(img[o:o + CROP_SIZE, o:o + CROP_SIZE].astype(np.float32) / 255, rng)
    return (f * 255).astype(np.uint8)


def center(crops):
    o = (CROP_CACHE - CROP_SIZE) // 2
    return crops[:, o:o + CROP_SIZE, o:o + CROP_SIZE]


@torch.no_grad()
def predict(model, imgs, task, bs=64):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(imgs), bs):
        x = to_tensor(imgs[i:i + bs]).to(dev)
        y = model(x)
        if task == "classify":  # average with the mirrored image
            y = (y.softmax(1) + model(x.flip(3)).softmax(1)) / 2
        else:
            y = y.sigmoid()
        out.append(y.cpu())
    return torch.cat(out)


def to_xyxy(p):
    """Model output (cx, cy, w, h) in 0..1 -> x0, y0, x1, y1."""
    return torch.cat([p[:, :2] - p[:, 2:] / 2, p[:, :2] + p[:, 2:] / 2], 1)


def scores(task, pred, target):
    if task == "locate":
        iou = box_iou(to_xyxy(pred), torch.from_numpy(target))
        return {"mean_iou": float(iou.mean()), "iou_over_0.5": float((iou > 0.5).float().mean())}
    p, y = pred.numpy(), target
    m = y != CLASSES.index("sever")
    return {"accuracy": float((p.argmax(1) == y).mean()),
            "spur_vs_normal_auc": float(roc_auc_score(y[m] == 1, p[m, 1] / (p[m, 0] + p[m, 1]))),
            "macro_auc": float(roc_auc_score(y, p, multi_class="ovr"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["locate", "classify"], required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--init", help="flat foot model.onnx; its backbone is the starting point")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    d = np.load(a.cache, allow_pickle=True)
    tr, va = d["split"] == "train", d["split"] == "val"
    if a.task == "locate":
        imgs, target = d["loc_img"], d["loc_box"]
        model, key = resnet18(4, a.init), "mean_iou"
    else:
        imgs, target = d["crop"], d["y"]
        model, key = resnet18(len(CLASSES), a.init), "macro_auc"
    val_imgs = imgs[va] if a.task == "locate" else center(imgs[va])
    tr_idx = np.flatnonzero(tr)
    dev = torch.device(a.device)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_idx) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    best, log = -1, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            if a.task == "locate":
                pairs = [aug_locate(imgs[j], target[j], rng) for j in idx]
                x = to_tensor(np.stack([p[0] for p in pairs])).to(dev)
                tb = torch.from_numpy(np.stack([p[1] for p in pairs])).to(dev)
                pb = to_xyxy(model(x).sigmoid())
                loss = F.l1_loss(pb, tb) * 5 + (1 - box_iou(pb, tb)).mean()
            else:
                x = to_tensor(np.stack([aug_classify(imgs[j], rng) for j in idx])).to(dev)
                yt = torch.from_numpy(target[idx]).long().to(dev)
                loss = F.cross_entropy(model(x), yt, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        s = scores(a.task, predict(model, val_imgs, a.task), target[va])
        s.update(epoch=ep + 1, loss=tot / len(perm), minutes=(time.time() - t0) / 60)
        log.append(s)
        print(json.dumps(s), flush=True)
        if s[key] > best:
            best = s[key]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
