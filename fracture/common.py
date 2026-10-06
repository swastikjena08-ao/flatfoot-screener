"""Shared pieces for the fracture flag: image loading, letterbox, the fracture model,
and the weight transfer from the flat foot classifier (model.onnx)."""
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few FracAtlas JPGs are cut short by a few bytes

SIZE = 384  # default: whole X-ray letterboxed to SIZE x SIZE, giving a SIZE/32 = 12 x 12 fracture map
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def load_gray(path):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.convert("L"))


def letterbox(gray, size=SIZE):
    """Fit the whole image into a size x size square. Returns canvas, scale, x/y offset."""
    h, w = gray.shape
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(Image.fromarray(gray).resize((nw, nh), Image.BILINEAR))
    canvas = np.zeros((size, size), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas, s, ox, oy


def box_mask(boxes, size=SIZE):
    """uint8 mask with 255 inside each (x0, y0, x1, y1) box given in canvas pixels."""
    m = np.zeros((size, size), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[int(max(0, y0)):int(np.ceil(min(size, y1))), int(max(0, x0)):int(np.ceil(min(size, x1)))] = 255
    return m


def to_tensor(batch_u8):
    """N x H x W uint8 -> N x 3 x H x W float, ImageNet-normalised gray."""
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).float().div_(255).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - torch.from_numpy(MEAN).view(1, 3, 1, 1)) / torch.from_numpy(STD).view(1, 3, 1, 1)


class FractureNet(nn.Module):
    """ResNet-18 body plus a 1x1 conv that scores every cell of the 12x12 grid for
    "fracture here" (12x12 at 384 px). The image score pools the cells (log-sum-exp, a soft max), so the
    map shows where and the pooled score says whether."""

    def __init__(self, init_onnx=None):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        if init_onnx:
            load_folded_onnx(m, init_onnx)
        self.body = nn.Sequential(*list(m.children())[:-2])
        self.drop = nn.Dropout2d(0.2)
        self.cell = nn.Conv2d(512, 1, 1)
        nn.init.constant_(self.cell.bias, -4.0)  # start with "no fracture anywhere"

    def forward(self, x):
        cells = self.cell(self.drop(self.body(x)))[:, 0]   # N x (side/32) x (side/32) logits
        return pool(cells), cells


def pool(cells, r=4.0):
    """Log-sum-exp pooling of N x g x g cell logits into one logit per image."""
    flat = cells.flatten(1)
    return (torch.logsumexp(r * flat, 1) - np.log(flat.shape[1])) / r


def cell_targets(masks_u8):
    """N x S x S box masks -> N x S/32 x S/32 fraction of each cell inside a box."""
    m = torch.from_numpy(np.ascontiguousarray(masks_u8)).float().div_(255).unsqueeze(1)
    return F.adaptive_avg_pool2d(m, m.shape[-1] // 32)[:, 0]


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
