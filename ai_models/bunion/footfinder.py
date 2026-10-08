"""Foot finder: locate the left and/or right foot in a whole AP foot X-ray.

The bunion model needs one foot, cropped and mirrored to look like a right foot.
This model takes the whole X-ray (letterboxed to 320x512) and predicts, for each
side, whether that foot is present and its box (two corner points read out with
the same heatmap + soft-argmax head as the bunion model).

Usage:
  python footfinder.py prep  --data /path/to/hvangleest --out work/feet.npz
  python footfinder.py train --cache work/feet.npz --out work/footfinder --epochs 40
  python footfinder.py eval  --cache work/feet.npz --ckpt work/footfinder/best.pt --data /path/to/hvangleest \
                             --bunion-cache work/cache.npz --bunion-ckpt work/run/best.pt --bunion-arch resnet34 --out work/footfinder
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model import soft_argmax
from train import to_tensor

FW, FH = 320, 512   # foot-finder input (width, height)
SIDES = ("left", "right")


def parse(s):
    return np.array([float(v) for v in str(s).split(",")], dtype=np.float64)


def letterbox(img, w=FW, h=FH):
    """Resize keeping aspect ratio and pad to (h, w). Returns canvas and (scale, ox, oy)."""
    H, W = img.shape[:2]
    s = min(w / W, h / H)
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out = np.zeros((h, w), np.uint8)
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, (s, ox, oy)


class FootNet(nn.Module):
    """ResNet-18 + upsampling head: 4 heatmaps (left tl/br, right tl/br) and 2 presence logits."""

    def __init__(self, weights=None):
        super().__init__()
        r = torchvision.models.resnet18(weights="DEFAULT" if weights == "imagenet" else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4

        def up(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.u3, self.u2, self.u1 = up(512 + 256, 256), up(256 + 128, 128), up(128 + 64, 64)
        self.out = nn.Conv2d(64, 4, 1)
        self.pres = nn.Linear(512, 2)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        y = self.u3(torch.cat([F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False), f3], 1))
        y = self.u2(torch.cat([F.interpolate(y, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
        y = self.u1(torch.cat([F.interpolate(y, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
        return self.out(y), self.pres(f4.mean((-1, -2)))


# ---------------------------------------------------------------- prep

def prep(a):
    df = pd.read_csv(os.path.join(a.data, "HVAngleEst", "datasets.csv"))
    split_of = {}
    for name in ("train", "val", "test"):
        for f in pd.read_csv(os.path.join(a.data, "splits", f"{name}.csv")).filename:
            split_of[f] = name
    imgs, boxes, pres, meta = [], [], [], []
    for fn, g in df.groupby("filename", sort=False):
        img = cv2.imread(os.path.join(a.data, "HVAngleEst", "images", fn), cv2.IMREAD_GRAYSCALE)
        canvas, (s, ox, oy) = letterbox(img)
        W, H = img.shape[1], img.shape[0]
        b = np.zeros((2, 4), np.float32)
        p = np.zeros(2, np.float32)
        for _, r in g.iterrows():
            k = SIDES.index(r.labels)
            if p[k]:
                continue  # one image has two "left" boxes; keep the first
            x1, y1, x2, y2 = parse(r.boxes) * [W, H, W, H]
            b[k] = [x1 * s + ox, y1 * s + oy, x2 * s + ox, y2 * s + oy]
            p[k] = 1
        imgs.append(canvas)
        boxes.append(b)
        pres.append(p)
        meta.append(dict(filename=fn, split=split_of.get(fn, "none"), scale=s, ox=ox, oy=oy, W=W, H=H))
        if len(meta) % 200 == 0:
            print(len(meta), flush=True)
    np.savez_compressed(a.out, imgs=np.stack(imgs), boxes=np.stack(boxes), pres=np.stack(pres), meta=json.dumps(meta))
    print("saved", a.out, len(meta), pd.DataFrame(meta).split.value_counts().to_dict())


# ---------------------------------------------------------------- train

def augment(img, box, pres, rng):
    """Random horizontal flip (swaps left/right), rotation, scale, shift and contrast."""
    box, pres = box.copy(), pres.copy()
    h, w = img.shape
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        box = box[::-1].copy()
        pres = pres[::-1].copy()
        box[:, [0, 2]] = w - box[:, [2, 0]]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.8, 1.15))
    M[:, 2] += rng.uniform(-24, 24, 2)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    for k in range(2):
        x1, y1, x2, y2 = box[k]
        c = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ M.T
        box[k] = [c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()]
    f = img.astype(np.float32) / 255.0
    f = np.clip(f ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
    return (f * 255).astype(np.uint8), box.astype(np.float32), pres


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@torch.no_grad()
def predict(model, imgs, bs=32):
    """-> boxes (N, 2, 4) in canvas px, presence probabilities (N, 2)."""
    model.eval()
    dev = next(model.parameters()).device
    bx, pr = [], []
    for i in range(0, len(imgs), bs):
        logits, p = model(to_tensor(imgs[i:i + bs]).to(dev))
        bx.append(soft_argmax(logits).reshape(-1, 2, 4).cpu().numpy())
        pr.append(torch.sigmoid(p).cpu().numpy())
    return np.concatenate(bx), np.concatenate(pr)


def det_score(boxes, probs, gt, pres):
    """Side presence accuracy and box IoU for present feet."""
    hit = (probs > 0.5) == (pres > 0.5)
    ious = [box_iou(boxes[i, k], gt[i, k]) for i in range(len(gt)) for k in range(2) if pres[i, k]]
    return dict(n=len(gt), side_acc=float(hit.all(1).mean()), iou_mean=float(np.mean(ious)),
                iou_min=float(np.min(ious)), iou_over_0_8=float(np.mean(np.array(ious) > 0.8)))


def train(a):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    z = np.load(a.cache)
    imgs, boxes, pres = z["imgs"], z["boxes"], z["pres"]
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    tr, va = np.where(meta.split == "train")[0], np.where(meta.split == "val")[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FootNet("imagenet").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * math.ceil(len(tr) / a.bs), pct_start=0.1)
    os.makedirs(a.out, exist_ok=True)
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(tr)
        for i in range(0, len(order), a.bs):
            batch = [augment(imgs[j], boxes[j], pres[j], rng) for j in order[i:i + a.bs]]
            x = to_tensor(np.stack([b[0] for b in batch])).to(dev)
            y = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev).reshape(-1, 4, 2)  # 4 corner points
            pm = torch.from_numpy(np.stack([b[2] for b in batch])).to(dev)
            logits, pl = model(x)
            p = soft_argmax(logits)
            m = pm.repeat_interleave(2, 1)  # (B, 4): corner mask from side presence
            l_box = (F.smooth_l1_loss(p, y, beta=2.0, reduction="none").mean(-1) * m).sum() / m.sum().clamp_min(1)
            l_pres = F.binary_cross_entropy_with_logits(pl, pm)
            loss = l_box + 10 * l_pres
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
        bx, pr = predict(model, imgs[va])
        s = det_score(bx, pr, boxes[va], pres[va])
        row = dict(epoch=ep + 1, loss=tot / len(tr), sec=round(time.time() - t0), **{k: round(v, 4) for k, v in s.items()})
        log.append(row)
        print(json.dumps(row), flush=True)
        if s["side_acc"] + s["iou_mean"] > best:
            best = s["side_acc"] + s["iou_mean"]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    pd.DataFrame(log).to_csv(os.path.join(a.out, "log.csv"), index=False)


# ---------------------------------------------------------------- eval (end to end)

def evaluate(a):
    """Foot-finder test scores, then HVA/IMA when the bunion model gets foot-finder crops
    instead of the dataset's hand-drawn boxes."""
    import common
    import prep as bprep
    import train as btrain
    from model import PointNet

    z = np.load(a.cache)
    imgs, boxes, pres = z["imgs"], z["boxes"], z["pres"]
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    te = np.where(meta.split == "test")[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ff = FootNet().to(dev)
    ff.load_state_dict(torch.load(a.ckpt, map_location=dev))
    bx, pr = predict(ff, imgs[te])
    det = det_score(bx, pr, boxes[te], pres[te])
    print("footfinder test", json.dumps(det), flush=True)

    # bunion model on foot-finder crops, one per labeled test foot
    bz = np.load(a.bunion_cache)
    common.IN_W, common.IN_H = bz["imgs"].shape[2], bz["imgs"].shape[1]
    bmeta = pd.DataFrame(json.loads(str(bz["meta"])))
    bm = PointNet(None, a.bunion_arch).to(dev)
    bm.load_state_dict(torch.load(a.bunion_ckpt, map_location=dev))
    pos = {fn: i for i, fn in enumerate(meta.filename.values[te])}
    crops, rows, missed = [], [], 0
    cache = {}
    for _, r in bmeta[bmeta.split == "test"].iterrows():
        i = pos[r.filename]
        k = SIDES.index(r.side)
        if pr[i, k] < 0.5:
            missed += 1
            continue
        if r.filename not in cache:
            cache = {r.filename: cv2.imread(os.path.join(a.data, "HVAngleEst", "images", r.filename), cv2.IMREAD_GRAYSCALE)}
        m = meta.iloc[te[i]]
        x1, y1, x2, y2 = (bx[i, k] - [m.ox, m.oy, m.ox, m.oy]) / m.scale  # canvas px -> X-ray px
        crop, _ = bprep.crop_foot(cache[r.filename], (x1, y1, x2, y2), k == 0)
        crops.append(crop)
        rows.append(r)
    sub = pd.DataFrame(rows)
    s = btrain.score(btrain.predict(bm, np.stack(crops)), sub)
    n_all = int((bmeta.split == "test").sum())
    # a missed foot counts as a miss in the "within 3" shares
    e2e = dict(n=n_all, missed_feet=missed, hva_mae=s["hva_mae"], ima_mae=s["ima_mae"],
               hva_within3=s["hva_within3"] * len(sub) / n_all, ima_within3=s["ima_within3"] * len(sub) / n_all)
    print("end_to_end test_official", json.dumps(e2e), flush=True)
    json.dump(dict(footfinder_test=det, end_to_end_test_official=e2e), open(os.path.join(a.out, "test_scores.json"), "w"), indent=2)

    class Exported(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            logits, p = self.m(x)
            return soft_argmax(logits).reshape(-1, 2, 4), torch.sigmoid(p)

    ff_cpu = FootNet()
    ff_cpu.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    torch.onnx.export(Exported(ff_cpu).eval(), torch.zeros(1, 3, FH, FW), os.path.join(a.out, "footfinder.onnx"),
                      input_names=["image"], output_names=["boxes", "present"], opset_version=17, dynamo=False)
    print("exported", os.path.join(a.out, "footfinder.onnx"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--data")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--ckpt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bunion-cache")
    ap.add_argument("--bunion-ckpt")
    ap.add_argument("--bunion-arch", default="resnet34")
    a = ap.parse_args()
    {"prep": prep, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
