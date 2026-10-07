"""Tests for the flat foot models' input handling and left/right logic (assistant/flatfoot.py).

Run from the flatfoot_app folder:  python -m pytest tests
The analyse_xray tests need tests/data/10L.JPG and 10R.JPG, which are kept out of the repo.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PIL import Image

from assistant import flatfoot as ff

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reopen(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


# ---------- normalize_input ----------

def test_16bit_tiff_keeps_contrast():
    a = np.linspace(0, 65535, 256 * 256).reshape(256, 256).astype(np.uint16)
    img = _reopen(Image.fromarray(a), "TIFF")
    assert img.mode.startswith("I")

    out = np.asarray(ff.flatten_alpha(ff.normalize_input(img)))[..., 0]

    # before the fix ~99.6% of pixels were clipped to 255
    assert (out == 255).mean() < 0.05
    assert out.min() == 0 and out.max() == 255


def test_constant_16bit_image_does_not_divide_by_zero():
    img = Image.fromarray(np.full((10, 10), 1000, np.uint16))
    out = ff.normalize_input(img)
    assert out.mode == "L"


def test_exif_rotation_is_applied():
    img = Image.new("RGB", (400, 200))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90° CW on display
    rotated = _reopen(img, "JPEG", exif=exif)

    assert ff.normalize_input(rotated).size == (200, 400)


def test_plain_8bit_image_is_unchanged():
    img = Image.new("RGB", (30, 20), (10, 20, 30))
    out = ff.normalize_input(img)
    assert out.mode == "RGB" and out.size == (30, 20)
    assert out.getpixel((0, 0)) == (10, 20, 30)


# ---------- ModelLoader ----------

# ---------- analyse_xray (real models) ----------

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO, RIGHT, LEFT = 0, 1, 2


@pytest.fixture(scope="module")
def models():
    import onnxruntime as ort
    clf = ort.InferenceSession(os.path.join(APP, "model.onnx"), providers=["CPUExecutionProvider"])
    lm = ff.Landmarks(os.path.join(APP, "landmarks.onnx"), os.path.join(APP, "landmarks.json"))
    return clf, lm


def _xray(name):
    path = os.path.join(HERE, "data", name)
    if not os.path.exists(path):
        pytest.skip(f"tests/data/{name} not found (kept out of the repo)")
    return ff.normalize_input(Image.open(path))


def test_auto_detects_left_foot(models):
    res = ff.analyse_xray(_xray("10L.JPG"), AUTO, *models)
    assert res.left is True
    assert res.how == "detected automatically"


def test_auto_detects_right_foot(models):
    res = ff.analyse_xray(_xray("10R.JPG"), AUTO, *models)
    assert res.left is False


def test_mirrored_right_foot_gives_same_answer_as_original(models):
    # The models only know right feet; a left foot is handled by mirroring. So a
    # mirrored right foot must give the identical verdict and angle, with the
    # heatmap and points mirrored back onto the image it was given.
    img = _xray("10R.JPG")
    right = ff.analyse_xray(img, AUTO, *models)
    left = ff.analyse_xray(ff.mirror(img), AUTO, *models)

    assert left.left is True
    assert left.p == pytest.approx(right.p)
    np.testing.assert_allclose(left.cam, right.cam[:, ::-1])
    for name, (x, y) in right.points.items():
        assert left.points[name] == pytest.approx((img.width - 1 - x, y))
    assert ff.line_angle(left.points["L10"], left.points["L11"], left=True) == pytest.approx(
        ff.line_angle(right.points["L10"], right.points["L11"]))


def test_side_chosen_by_user_overrides_detection(models):
    res = ff.analyse_xray(_xray("10L.JPG"), RIGHT, *models)
    assert res.left is False
    assert res.how == "set by you"


def test_without_landmark_model_side_is_assumed(models):
    res = ff.analyse_xray(_xray("10L.JPG"), AUTO, models[0], None)
    assert res.left is False and res.points is None
    assert res.how == "assumed (landmark model not found)"
