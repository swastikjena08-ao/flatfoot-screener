"""Regression tests for input handling and failure paths in flatfoot_app.

Run from the flatfoot_app folder:  python -m pytest tests
"""
import io
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

import flatfoot_app as ff


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    # don't load the real 100 MB of models in the background
    monkeypatch.setattr(ff.ModelLoader, "start", lambda self: None)
    win = ff.MainWindow()
    yield win
    win.close()


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

def test_landmark_failure_keeps_classifier(monkeypatch):
    monkeypatch.setattr(ff.ort, "InferenceSession", lambda *a, **k: "clf")
    monkeypatch.setattr(ff.os.path, "exists", lambda p: True)

    def broken(*a, **k):
        raise FileNotFoundError("landmarks.json")
    monkeypatch.setattr(ff, "Landmarks", broken)

    loader = ff.ModelLoader()
    got, failed = [], []
    loader.loaded.connect(lambda *args: got.append(args))
    loader.failed.connect(failed.append)
    loader.run()

    assert not failed
    clf, lm, note = got[0]
    assert clf == "clf" and lm is None
    assert "landmarks.json" in note


# ---------- analyse_xray (real models) ----------

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO, RIGHT, LEFT = 0, 1, 2


@pytest.fixture(scope="module")
def models():
    import onnxruntime as ort
    clf = ort.InferenceSession(ff.resource_path("model.onnx"), providers=["CPUExecutionProvider"])
    lm = ff.Landmarks(ff.resource_path("landmarks.onnx"), ff.resource_path("landmarks.json"))
    return clf, lm


def _xray(name):
    return ff.normalize_input(Image.open(os.path.join(HERE, "data", name)))


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


# ---------- heatmap opacity slider ----------

def _count_calls(monkeypatch, name):
    calls = []
    real = getattr(ff, name)
    monkeypatch.setattr(ff, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _show_result(window, size=(4000, 3000)):
    window.img = Image.new("RGB", size, "gray")
    window.cam = np.random.default_rng(0).random((7, 7)).astype(np.float32)
    window.redraw()


def test_moving_opacity_slider_does_not_reprocess_the_xray(window, monkeypatch):
    _show_result(window)
    prepared = _count_calls(monkeypatch, "flatten_alpha")
    upsampled = _count_calls(monkeypatch, "upsample_cam")

    for v in range(0, 101, 10):
        window.heat_alpha.setValue(v)
    window.heat_toggle.setChecked(False)
    window.heat_toggle.setChecked(True)

    assert prepared == [] and upsampled == []


def test_new_result_is_rendered_fresh(window, monkeypatch):
    _show_result(window)
    prepared = _count_calls(monkeypatch, "flatten_alpha")

    window.cam = window.cam[:, ::-1]  # e.g. side changed: new heatmap, same image
    window.redraw()

    assert prepared == [1]


def test_overlay_matches_heatmap_blend(window):
    # opacity a must give base*(1-a) + jet(cam)*a, as before the slider was optimised
    _show_result(window, size=(300, 200))
    window.heat_alpha.setValue(45)

    shown = window.view._base.toImage().convertToFormat(ff.QImage.Format_RGB888)
    got = np.frombuffer(shown.constBits(), np.uint8).reshape(shown.height(), shown.bytesPerLine())
    got = got[:, :shown.width() * 3].reshape(shown.height(), shown.width(), 3).astype(int)

    base = np.asarray(window.img.convert("L"), np.float32)[..., None].repeat(3, -1)
    heat = ff.jet(ff.upsample_cam(window.cam, 300, 200)) * 255
    expected = (base * 0.55 + heat * 0.45).round()

    assert np.abs(got - expected).max() <= 2


# ---------- high-DPI display ----------

def _hidpi_view(monkeypatch, dpr=2.0):
    """An ImageView on a simulated 2x screen; returns (view, list of pixmaps it displayed).

    The offscreen test platform is 1x, and QLabel.pixmap() resamples to the real
    screen, so capture what the view hands to setPixmap instead.
    """
    view = ff.ImageView()
    monkeypatch.setattr(view, "devicePixelRatioF", lambda: dpr)
    shown = []
    monkeypatch.setattr(view, "setPixmap", shown.append)
    view.resize(520, 460)
    return view, shown


def test_xray_is_drawn_at_full_screen_resolution_on_hidpi(app, monkeypatch):
    view, shown = _hidpi_view(monkeypatch)
    view.set_content(ff.QPixmap(2080, 1840), (2080, 1840))

    assert shown[-1].devicePixelRatio() == 2.0
    assert (shown[-1].width(), shown[-1].height()) == (1040, 920)  # physical pixels, not 520x460 upscaled


def test_landmark_overlay_is_sharp_and_aligned_on_hidpi(app, monkeypatch):
    view, shown = _hidpi_view(monkeypatch)
    view.set_content(ff.QPixmap(2080, 1840), (2080, 1840))
    view.points = {"L10": (0, 0), "L11": (2080, 1840)}
    view.refresh()

    shown = shown[-1]
    assert shown.devicePixelRatio() == 2.0 and shown.width() == 1040
    # points are placed in widget (logical) coordinates
    corner = view.to_widget(2080, 1840)
    assert (corner.x(), corner.y()) == pytest.approx((520, 460))


def test_window_renders_more_detail_on_hidpi(window, monkeypatch):
    monkeypatch.setattr(window.view, "devicePixelRatioF", lambda: 2.0)
    _show_result(window, size=(4000, 3000))
    assert window.view._base.width() == 2000  # 1000 logical px at 2x


# ---------- MainWindow failure path ----------

class _Session:
    """Stand-in for the classifier ONNX session.

    Bright images score 0.9 and dark ones 0.1, so a test can tell which image a
    result came from. `slow_bright` delays bright images to simulate a slow model.
    """

    def __init__(self, fail=False, slow_bright=0.0):
        self.fail, self.slow_bright = fail, slow_bright

    def get_outputs(self):
        return [SimpleNamespace(name="prob"), SimpleNamespace(name="cam")]

    def run(self, _, feeds):
        if self.fail:
            raise RuntimeError("boom")
        bright = feeds["image"].mean() > 0
        if bright and self.slow_bright:
            time.sleep(self.slow_bright)
        return [np.array([[0.9 if bright else 0.1]], np.float32), np.ones((1, 1, 7, 7), np.float32)]


def _wait_for_analysis(app, win, timeout=10):
    end = time.monotonic() + timeout
    while win.analysing and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not win.analysing, "analysis did not finish"


def test_window_stays_responsive_while_model_runs(app, window):
    window.on_loaded(_Session(slow_bright=1.0), None, "")

    start = time.monotonic()
    window.set_image(Image.new("RGB", (300, 200), "white"), "slow.png")
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"UI thread blocked for {elapsed:.2f}s"
    assert window.title.text() == "Analysing…"
    assert window.view.img_size == (300, 200)  # image is shown straight away
    _wait_for_analysis(app, window)
    assert window.p == pytest.approx(0.9)


def test_slow_result_for_old_image_is_not_applied_to_new_image(app, window):
    window.on_loaded(_Session(slow_bright=0.5), None, "")

    window.set_image(Image.new("RGB", (300, 200), "white"), "slow.png")
    window.set_image(Image.new("RGB", (120, 400), "black"), "fast.png")
    _wait_for_analysis(app, window)

    assert window.p == pytest.approx(0.1)
    assert window.view.img_size == (120, 400)


def test_result_arriving_after_image_removed_is_ignored(app, window):
    window.on_loaded(_Session(slow_bright=0.5), None, "")

    window.set_image(Image.new("RGB", (300, 200), "white"), "slow.png")
    window.clear_image()
    _wait_for_analysis(app, window)

    assert window.p is None
    assert window.title.text() == "No image yet"


def test_failed_analysis_shows_new_image_without_stale_result(app, window):
    window.on_loaded(_Session(), None, "")
    window.set_image(Image.new("RGB", (300, 200)), "first.png")
    _wait_for_analysis(app, window)
    assert window.p == pytest.approx(0.1)

    window.session = _Session(fail=True)
    window.set_image(Image.new("RGB", (120, 400)), "second.png")
    _wait_for_analysis(app, window)

    assert window.p is None and window.cam is None
    assert window.view.points == {}
    assert window.view.img_size == (120, 400)  # new image is displayed, not the old one
    assert window.title.text() == "Analysis failed"
