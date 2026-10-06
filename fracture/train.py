"""Train the fracture model on CPU or GPU.

Two losses: whether the X-ray has a fracture (pooled score), and which 12x12 cells hold one
(the FracAtlas boxes). The box loss is what lets ~260 fractures go further than labels alone.

Usage:
  python train.py --cache work/cache.npz --init ../flatfoot_app/model.onnx --out work/run
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

from common import FractureNet, cell_targets, to_tensor


def augment(img, mask, rng):
    """Same flip, rotation, scale and shift for image and box mask, then brightness/contrast."""
    n = img.shape[0]
    M = cv2.getRotationMatrix2D((n / 2, n / 2), rng.uniform(-15, 15), rng.uniform(0.85, 1.2))
    M[:, 2] += rng.uniform(-25, 25, 2)
    if rng.random() < 0.5:
        M = M @ np.array([[-1, 0, n], [0, 1, 0], [0, 0, 1]], np.float64)
    img = cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (n, n), flags=cv2.INTER_NEAREST, borderValue=0)
    f = img.astype(np.float32) / 255
    f = np.clip(f ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.2))
    return (f * 255).astype(np.uint8), mask


@torch.no_grad()
def predict(model, imgs, bs=32):
    """Fracture probability per image and 12x12 cell map, averaged with the mirrored image."""
    model.eval()
    dev = next(model.parameters()).device
    ps, maps = [], []
    for i in range(0, len(imgs), bs):
        x = to_tensor(imgs[i:i + bs]).to(dev)
        y, c = model(x)
        y2, c2 = model(x.flip(3))
        ps.append(((y.sigmoid() + y2.sigmoid()) / 2).cpu())
        maps.append(((c.sigmoid() + c2.flip(2).sigmoid()) / 2).cpu())
    return torch.cat(ps).numpy(), torch.cat(maps).numpy()


def scores(p, y, foot_ankle):
    s = {"auc": float(roc_auc_score(y, p))}
    if foot_ankle.any() and 0 < y[foot_ankle].sum() < foot_ankle.sum():
        s["foot_ankle_auc"] = float(roc_auc_score(y[foot_ankle], p[foot_ankle]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--init", help="flat foot model.onnx; its backbone is the starting point")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cell-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)

    d = np.load(a.cache, allow_pickle=True)
    imgs, masks, y = d["img"], d["mask"], d["y"]
    fa = np.isin(d["region"], ["foot", "ankle", "foot_ankle_view"])
    tr, va = d["split"] == "train", d["split"] == "val"
    tr_idx = np.flatnonzero(tr)
    # draw fractured X-rays more often so a batch of 16 usually holds a few
    w = np.where(y[tr_idx] == 1, 3.0, 1.0)
    w /= w.sum()
    dev = torch.device(a.device)
    model = FractureNet(a.init).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    per_epoch = len(tr_idx) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * per_epoch, pct_start=0.1)
    best, log = -1, []
    print(f"train {len(tr_idx)} ({int(y[tr].sum())} fractured), val {int(va.sum())} ({int(y[va].sum())}), "
          f"device {dev}", flush=True)
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(per_epoch):
            idx = rng.choice(tr_idx, a.bs, p=w)
            pairs = [augment(imgs[j], masks[j], rng) for j in idx]
            x = to_tensor(np.stack([p[0] for p in pairs])).to(dev)
            ct = (cell_targets(np.stack([p[1] for p in pairs])) > 0.1).float().to(dev)
            yt = torch.from_numpy(y[idx]).float().to(dev)
            logit, cells = model(x)
            # an X-ray marked fractured whose box was rotated out of frame keeps its image label only
            loss = F.binary_cross_entropy_with_logits(logit, yt) + a.cell_weight * \
                F.binary_cross_entropy_with_logits(cells, ct, pos_weight=torch.tensor(5.0, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        p, _ = predict(model, imgs[va])
        s = scores(p, y[va], fa[va])
        s.update(epoch=ep + 1, loss=tot / per_epoch, minutes=(time.time() - t0) / 60)
        log.append(s)
        print(json.dumps(s), flush=True)
        if s["auc"] > best:
            best = s["auc"]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
