"""Shared pieces for the heel spur feature: image loading, letterbox, heel crop,
models, and the weight transfer from the flat foot classifier (model.onnx)."""
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps

CLASSES = ["normal", "spur", "sever"]  # sever = child's heel growth plate (Sever's disease), see README
LOC_SIZE = 256    # heel finder input (square letterbox)
CROP_SIZE = 224   # spur classifier input
CROP_CACHE = 256  # cached crop size, random-cropped to CROP_SIZE in training
CROP_PAD = 1.6    # crop side = CROP_PAD * longer side of the heel box
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def load_gray(path):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.convert("L"))


def letterbox(gray, size=LOC_SIZE):
    """Fit the whole image into a size x size square. Returns canvas, scale, x/y offset."""
    h, w = gray.shape
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(Image.fromarray(gray).resize((nw, nh), Image.BILINEAR))
    canvas = np.zeros((size, size), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas, s, ox, oy


def heel_crop(gray, box, out=CROP_CACHE, pad=CROP_PAD):
    """Square crop centred on the heel box (x0, y0, x1, y1 in image pixels), zero padded."""
    x0, y0, x1, y1 = box
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * pad
    a, b = cx - side / 2, cy - side / 2
    img = Image.fromarray(gray).transform((out, out), Image.AFFINE,
                                          [side / out, 0, a, 0, side / out, b], resample=Image.BILINEAR)
    return np.asarray(img)


def to_tensor(batch_u8):
    """N x H x W uint8 -> N x 3 x H x W float, ImageNet-normalised gray."""
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).float().div_(255).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - torch.from_numpy(MEAN).view(1, 3, 1, 1)) / torch.from_numpy(STD).view(1, 3, 1, 1)


def resnet18(n_out, init_onnx=None):
    m = torchvision.models.resnet18(weights=None)
    if init_onnx:
        load_folded_onnx(m, init_onnx)
    m.fc = nn.Linear(512, n_out)
    return m


def load_folded_onnx(model, path):
    """Copy the backbone of an exported torchvision ResNet-18 into `model`.

    The export folded each BatchNorm into the conv before it, so the conv gets the
    weight and its BatchNorm becomes a pass-through holding the bias."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(path).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    sd, n = model.state_dict(), 0
    for node in g.node:
        if node.op_type != "Conv":
            continue
        # e.g. /conv1/Conv, /layer1/layer1.0/conv1/Conv, /layer2/layer2.0/downsample/downsample.0/Conv
        parts = node.name.strip("/").split("/")[:-1]
        mod = parts[0] if len(parts) == 1 else parts[1] + "." + (
            "downsample.0" if parts[2] == "downsample" else parts[2])
        bn = re.sub(r"conv(\d)$", r"bn\1", mod).replace("downsample.0", "downsample.1")
        w, b = inits[node.input[1]], inits[node.input[2]]
        sd[mod + ".weight"] = torch.from_numpy(w.copy())
        sd[bn + ".weight"] = torch.ones(len(b))
        sd[bn + ".bias"] = torch.from_numpy(b.copy())
        sd[bn + ".running_mean"] = torch.zeros(len(b))
        sd[bn + ".running_var"] = torch.ones(len(b)) - 1e-5
        n += 1
    assert n == 20, f"expected 20 convs, got {n}"
    model.load_state_dict(sd)


def box_iou(a, b):
    """IoU of N x 4 boxes (x0, y0, x1, y1)."""
    lt, rb = torch.maximum(a[:, :2], b[:, :2]), torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda t: (t[:, 2:] - t[:, :2]).clamp(min=0).prod(1)
    return inter / (area(a) + area(b) - inter + 1e-7)
