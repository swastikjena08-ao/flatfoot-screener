"""Shared pieces for the X-ray check: classes, image loading, letterbox and the model.

Reuses the image helpers and the flat foot backbone loader from heel_spur/common.py."""
import importlib.util
import os

import numpy as np

# heel_spur/common.py is also called common, so load it under its own name
_spec = importlib.util.spec_from_file_location(
    "heel_spur_common", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heel_spur", "common.py"))
_hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hs)
letterbox, load_gray, resnet18, to_tensor = _hs.letterbox, _hs.load_gray, _hs.resnet18, _hs.to_tensor

CLASSES = ["side", "top_down", "oblique", "not_foot"]
SIZE = 224   # model input (whole X-ray letterboxed into a square)
CACHE = 256  # cached letterbox size, random-cropped / scaled down to SIZE in training


def letterbox_u8(gray, size=CACHE):
    return letterbox(gray, size)[0]


def center(imgs):
    o = (CACHE - SIZE) // 2
    return imgs[:, o:o + SIZE, o:o + SIZE] if imgs.shape[1] == CACHE else imgs


def shrink(imgs):
    """Whole cached letterbox scaled to SIZE (used for evaluation, so nothing is cropped off)."""
    import cv2
    return np.stack([cv2.resize(i, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for i in imgs])
