"""Train the view classifier (side / top_down / oblique / not_foot) on CPU or GPU.

Each epoch draws the same number of X-rays from every class, so the 217 angled views
are not drowned out by the thousands of side views.

Usage:
  python train.py --cache work/cache.npz --init ../flat_foot/model.onnx --out work/view
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, CLASSES, SIZE, resnet18, shrink, to_tensor


def augment(img, rng):
    """Rotate, scale and shift the whole letterboxed X-ray into SIZE, flip, vary contrast,
    sometimes invert (some systems export bone as dark)."""
    s = SIZE / CACHE * rng.uniform(0.85, 1.1)
    M = cv2.getRotationMatrix2D((CACHE / 2, CACHE / 2), rng.uniform(-25, 25), s)
    M[:, 2] += (SIZE - CACHE) / 2 + rng.uniform(-15, 15, 2)
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    f = img.astype(np.float32) / 255
    f = np.clip(f ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.1:
        f = 1 - f
    return (f * 255).astype(np.uint8)


@torch.no_grad()
def predict(model, imgs, bs=64):
    model.eval()
    dev = next(model.parameters()).device
    return torch.cat([model(to_tensor(imgs[i:i + bs]).to(dev)).softmax(1).cpu()
                      for i in range(0, len(imgs), bs)])


def scores(p, y):
    pred = p.argmax(1).numpy()
    recall = {c: float((pred[y == k] == k).mean()) for k, c in enumerate(CLASSES)}
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(np.mean(list(recall.values()))),
            "recall": recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--init", help="flat foot model.onnx; its backbone is the starting point")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=1200, help="X-rays drawn per class each epoch")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    d = np.load(a.cache, allow_pickle=True)
    imgs, y, split = d["img"], d["y"], d["split"]
    by_class = [np.flatnonzero((split == "train") & (y == k)) for k in range(len(CLASSES))]
    va = split == "val"
    val_imgs, val_y = shrink(imgs[va]), y[va]
    dev = torch.device(a.device)
    model = resnet18(len(CLASSES), a.init).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    per_epoch = a.per_class * len(CLASSES) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * per_epoch, pct_start=0.1)
    best, log = -1, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        draw = np.concatenate([rng.choice(ix, a.per_class, replace=len(ix) < a.per_class) for ix in by_class])
        draw = rng.permutation(draw)
        for i in range(per_epoch):
            idx = draw[i * a.bs:(i + 1) * a.bs]
            x = to_tensor(np.stack([augment(imgs[j], rng) for j in idx])).to(dev)
            loss = F.cross_entropy(model(x), torch.from_numpy(y[idx]).long().to(dev), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        s = scores(predict(model, val_imgs), val_y)
        s.update(epoch=ep + 1, loss=tot / per_epoch, minutes=(time.time() - t0) / 60)
        log.append(s)
        print(json.dumps(s), flush=True)
        if s["balanced_accuracy"] > best:
            best = s["balanced_accuracy"]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
