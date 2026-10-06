"""Flatfoot X-ray Screener — desktop app.

Two models:
  * model.onnx      — ResNet-18 classifier (pes planus probability + heatmap), same as the website.
  * landmarks.onnx  — ResNet-18 heatmap model that locates landmarks L10 and L11;
                      the app reports the L10→L11 angle against the horizontal.

Both models were trained (Google Colab) on right-foot X-rays only, toes pointing left.
Left feet are handled by mirroring the image before inference and mapping the
heatmap and landmarks back onto the original (see run_model). Keep that contract
when changing preprocessing, side detection or the angle maths.
"""
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter,
                           QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget,
)

SIZE, THRESH, BORDER = 224, 0.5, (0.35, 0.65)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
EMPTY_TEXT = "Drop a foot X-ray here\nor click Open / press Ctrl+V"

VERDICT_STYLE = {
    "idle": ("Awaiting result", "#6b7280"),
    "flat": ("Pes planus likely", "#dc2626"),
    "normal": ("Normal arch likely", "#16a34a"),
    "border": ("Borderline", "#d97706"),
}
POINT_COLORS = {"L10": QColor("#22d3ee"), "L11": QColor("#f472b6")}


def resource_path(name):
    # PyInstaller unpacks bundled files into sys._MEIPASS
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def normalize_input(img):
    """Upright, 8-bit image. Applies EXIF rotation (phone photos) and rescales
    16-bit / float X-rays (TIFF exports), which Pillow would otherwise clip to white."""
    img = ImageOps.exif_transpose(img)
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        a = np.clip((a - lo) / (hi - lo if hi > lo else 1), 0, 1)
        img = Image.fromarray((a * 255).astype(np.uint8), "L")
    return img


def flatten_alpha(img):
    rgba = img.convert("RGBA")
    rgb = Image.new("RGB", img.size, (0, 0, 0))
    rgb.paste(rgba, mask=rgba.split()[3])
    return rgb


# ---------- classifier ----------

def to_tensor(img):
    d = np.asarray(flatten_alpha(img).resize((SIZE, SIZE), Image.Resampling.LANCZOS), np.float32)
    gray = (0.299 * d[..., 0] + 0.587 * d[..., 1] + 0.114 * d[..., 2]) / 255
    t = (gray[None] - MEAN[:, None, None]) / STD[:, None, None]
    return t[None].astype(np.float32)


def analyse(session, img):
    out = session.run(None, {"image": to_tensor(img)})
    res = dict(zip([o.name for o in session.get_outputs()], out))
    p = float(res["prob"].reshape(-1)[0])
    cam = res["cam"].reshape(res["cam"].shape[-2:])
    sign = 1 if p >= THRESH else -1
    cam = np.maximum(0, sign * cam)
    cam = cam / (cam.max() or 1)
    return p, cam


# ---------- landmarks ----------

class Landmarks:
    def __init__(self, onnx_path, meta_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.meta = json.load(open(meta_path, encoding="utf-8"))
        self.w, self.h, self.stride = self.meta["input_w"], self.meta["input_h"], self.meta["stride"]

    def letterbox(self, w, h):
        s = min(self.w / w, self.h / h)
        return s, (self.w - w * s) / 2, (self.h - h * s) / 2

    def predict(self, img):
        """Returns ({'L10': (x, y), 'L11': (x, y)} in image pixels, confidence).

        The model was trained on right feet only (toes pointing left), so pass
        left feet mirrored. Confidence is the weaker of the two heatmap peaks.
        """
        gray = ImageOps.grayscale(flatten_alpha(img))
        s, ox, oy = self.letterbox(*gray.size)
        # same affine letterbox as training
        canvas = gray.transform((self.w, self.h), Image.AFFINE,
                                [1 / s, 0, -ox / s, 0, 1 / s, -oy / s], resample=Image.BILINEAR)
        g = np.asarray(canvas, np.float32) / 255
        x = ((g[None] - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)
        hm = self.session.run(None, {"image": x})[0][0]
        conf = float(hm.reshape(len(hm), -1).max(1).min())
        pts = {}
        for name, h in zip(self.meta["points"], hm):
            yy, xx = np.unravel_index(np.argmax(h), h.shape)
            y0, y1 = max(0, yy - 2), min(h.shape[0], yy + 3)
            x0, x1 = max(0, xx - 2), min(h.shape[1], xx + 3)
            w = np.clip(h[y0:y1, x0:x1], 0, None)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            sw = w.sum() or 1
            px, py = (gx * w).sum() / sw * self.stride, (gy * w).sum() / sw * self.stride
            pts[name] = ((px - ox) / s, (py - oy) / s)
        return pts, conf


def mirror(img):
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def line_angle(p10, p11, left=False):
    """Angle of the L10→L11 line against the horizontal, in degrees (positive = L11 lower).

    Measured in right-foot orientation, so a left foot gives the same value as its mirror image.
    """
    dx = p11[0] - p10[0]
    return math.degrees(math.atan2(p11[1] - p10[1], -dx if left else dx))


AUTO_SIDE, RIGHT_SIDE, LEFT_SIDE = 0, 1, 2  # order of the "Foot side" combo box


@dataclass
class Analysis:
    p: float
    cam: np.ndarray
    points: dict | None  # {'L10': (x, y), 'L11': (x, y)} on the original image, or None
    left: bool
    how: str


def analyse_xray(img, side, session, landmarks):
    """Classifier + landmarks for one X-ray. No UI; safe to run off the main thread.

    Both models were trained on right feet (toes pointing left), so a left foot is
    mirrored first and the heatmap and points are mapped back onto `img`.
    """
    pts = None
    if side == AUTO_SIDE and landmarks is not None:
        pts, conf_r = landmarks.predict(img)
        pts_l, conf_l = landmarks.predict(mirror(img))
        left = conf_l > conf_r
        if left:
            pts = pts_l
        how = "detected automatically"
    else:
        left = side == LEFT_SIDE
        how = "set by you" if side != AUTO_SIDE else "assumed (landmark model not found)"
    work = mirror(img) if left else img
    if landmarks is not None and pts is None:
        pts, _ = landmarks.predict(work)
    if pts is not None:
        # Classify the foot only, framed like the training X-rays, so a calibration ball,
        # the other leg or a long shin can't sway the verdict or pull the heatmap off the foot.
        box = foot_box(pts, work.size)
        p, cam = analyse(session, work.crop(box))
        cam = cam_on_image(cam, box, *work.size)
    else:
        p, cam = analyse(session, work)
    if left:
        cam = cam[:, ::-1]
        pts = {k: (img.width - 1 - x, y) for k, (x, y) in pts.items()} if pts else pts
    return Analysis(p, cam, pts, left, how)


def foot_box(pts, size):
    """Crop (x0, y0, x1, y1) that frames the foot like the Pesplanus training X-rays.

    In those, measured on the 163 hand-labelled ones, the L10–L11 line's midpoint sits 76%
    across and 83% down, and the image is about 8 line-lengths wide and 5.4 tall.
    Parts outside the X-ray come out black, like the training images' borders.
    """
    (x1, y1), (x2, y2) = pts["L10"], pts["L11"]
    d = max(math.hypot(x2 - x1, y2 - y1), 0.02 * max(size))
    mx, my, w, h = (x1 + x2) / 2, (y1 + y2) / 2, 8 * d, 5.4 * d
    return tuple(round(v) for v in (mx - 0.76 * w, my - 0.83 * h, mx + 0.24 * w, my + 0.17 * h))


def cam_on_image(cam, box, w, h, grid=64):
    """Place a heatmap computed on crop `box` onto a coarse grid over the whole w x h image
    (zero outside the crop), so callers can keep stretching it over the full X-ray."""
    s = grid / max(w, h)
    gw, gh = max(2, round(w * s)), max(2, round(h * s))
    ch, cw = cam.shape
    x0, y0, x1, y1 = box
    xs = ((np.arange(gw) + 0.5) / s - x0) / (x1 - x0)  # cell centres, 0..1 inside the crop
    ys = ((np.arange(gh) + 0.5) / s - y0) / (y1 - y0)
    inside = ((ys >= 0) & (ys <= 1))[:, None] & ((xs >= 0) & (xs <= 1))[None, :]
    sx = np.clip(xs * cw - 0.5, 0, cw - 1)
    sy = np.clip(ys * ch - 0.5, 0, ch - 1)
    ix0, iy0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    ix1, iy1 = np.minimum(cw - 1, ix0 + 1), np.minimum(ch - 1, iy0 + 1)
    fx, fy = (sx - ix0)[None, :], (sy - iy0)[:, None]
    out = ((cam[np.ix_(iy0, ix0)] * (1 - fx) + cam[np.ix_(iy0, ix1)] * fx) * (1 - fy)
           + (cam[np.ix_(iy1, ix0)] * (1 - fx) + cam[np.ix_(iy1, ix1)] * fx) * fy)
    return np.where(inside, out, 0).astype(np.float32)


# ---------- drawing ----------

def upsample_cam(cam, w, h):
    """Bilinear resize with half-pixel centres, matching the website."""
    ch, cw = cam.shape
    sx = np.clip((np.arange(w) + 0.5) * cw / w - 0.5, 0, cw - 1)
    sy = np.clip((np.arange(h) + 0.5) * ch / h - 0.5, 0, ch - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(cw - 1, x0 + 1), np.minimum(ch - 1, y0 + 1)
    fx, fy = (sx - x0)[None, :], (sy - y0)[:, None]
    a, b = cam[np.ix_(y0, x0)], cam[np.ix_(y0, x1)]
    c, d = cam[np.ix_(y1, x0)], cam[np.ix_(y1, x1)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def jet(v):
    return np.stack([np.clip(1.5 - np.abs(4 * v - k), 0, 1) for k in (3, 2, 1)], -1)


def render_layers(img, cam, max_w=1000):
    """Grayscale X-ray and jet heatmap as same-sized RGB uint8 arrays (heat is None without a cam).

    The expensive part of drawing; done once per result so the opacity slider only blends.
    """
    scale = min(1, max_w / img.width)
    w, h = round(img.width * scale), round(img.height * scale)
    base = np.asarray(ImageOps.grayscale(flatten_alpha(img)).resize((w, h)), np.uint8)
    base = np.ascontiguousarray(np.repeat(base[..., None], 3, -1))
    heat = None
    if cam is not None:
        heat = np.ascontiguousarray((jet(upsample_cam(cam, w, h)) * 255).round().astype(np.uint8))
    return base, heat


def to_pixmap(arr):
    h, w, _ = arr.shape
    return QPixmap.fromImage(QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy())


def blend(base_pix, heat_pix, alpha):
    """base*(1-alpha) + heat*alpha, composited by Qt."""
    if heat_pix is None or alpha <= 0:
        return base_pix
    out = QPixmap(base_pix)
    p = QPainter(out)
    p.setOpacity(alpha)
    p.drawPixmap(0, 0, heat_pix)
    p.end()
    return out


def verdict(p):
    flat = p >= THRESH
    conf = (p if flat else 1 - p) * 100
    if BORDER[0] <= p <= BORDER[1]:
        return ("border", "Borderline, leaning flatfoot" if flat else "Borderline, leaning normal",
                f"The model is unsure ({conf:.0f}% for its answer). This X-ray needs a clinician's review.")
    if flat:
        return "flat", "Pes planus likely", f"The model is {conf:.1f}% confident this foot has a lowered arch."
    return "normal", "Normal arch likely", f"The model is {conf:.1f}% confident this foot has a normal arch."


class ModelLoader(QThread):
    loaded = Signal(object, object, str)  # classifier, landmarks (or None), landmark note
    failed = Signal(str)

    def run(self):
        try:
            clf = ort.InferenceSession(resource_path("model.onnx"), providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
            return
        # the landmark model is optional: a problem with it must not take down the classifier
        lm, note = None, "Landmark model not found (landmarks.onnx)."
        if os.path.exists(resource_path("landmarks.onnx")):
            try:
                lm, note = Landmarks(resource_path("landmarks.onnx"), resource_path("landmarks.json")), ""
            except Exception as e:  # noqa: BLE001
                note = f"Landmark model failed to load: {e}"
        self.loaded.emit(clf, lm, note)


class AnalysisWorker(QThread):
    """Runs analyse_xray off the UI thread. `job` lets the window drop stale results."""
    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, job, img, side, session, landmarks):
        super().__init__()
        self.job, self.args = job, (img, side, session, landmarks)

    def run(self):
        try:
            self.done.emit(self.job, analyse_xray(*self.args))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self.job, str(e))


# ---------- UI ----------

class ImageView(QLabel):
    """Shows the X-ray; L10/L11 markers can be dragged to correct them."""
    dropped = Signal(str)
    point_moved = Signal(str, float, float)

    def __init__(self):
        super().__init__(EMPTY_TEXT)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 460)
        self.setAcceptDrops(True)
        self.setStyleSheet("QLabel{border-radius:10px;color:#6b7280;"
                           "font-size:15px;background:#111827;}")
        self.points, self.img_size, self.show_points = {}, None, True
        self._base, self._rect, self._drag = None, None, None

    # mapping between widget and original-image coordinates
    def to_widget(self, x, y):
        sx, sy, w, h = self._rect
        return QPointF(sx + x * w / self.img_size[0], sy + y * h / self.img_size[1])

    def to_image(self, pos):
        sx, sy, w, h = self._rect
        x = (pos.x() - sx) * self.img_size[0] / w
        y = (pos.y() - sy) * self.img_size[1] / h
        return min(max(x, 0), self.img_size[0] - 1), min(max(y, 0), self.img_size[1] - 1)

    def set_content(self, pix, img_size):
        self._base, self.img_size = pix, img_size
        self.refresh()

    def clear_content(self):
        self._base, self.img_size, self.points, self._rect = None, None, {}, None
        self.clear()
        self.setText(EMPTY_TEXT)

    def refresh(self):
        if self._base is None:
            return
        # scale to physical pixels so the X-ray stays sharp at 125%/150%/200% display scaling;
        # _rect and all drawing below stay in logical (widget) coordinates
        dpr = self.devicePixelRatioF()
        scaled = self._base.scaled(self.size() * dpr, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled.setDevicePixelRatio(dpr)
        w, h = scaled.width() / dpr, scaled.height() / dpr
        self._rect = ((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        if self.points and self.show_points:
            canvas = QPixmap(self.size() * dpr)
            canvas.setDevicePixelRatio(dpr)
            canvas.fill(Qt.transparent)
            p = QPainter(canvas)
            p.setRenderHint(QPainter.Antialiasing)
            p.drawPixmap(int(self._rect[0]), int(self._rect[1]), scaled)
            a, b = self.to_widget(*self.points["L10"]), self.to_widget(*self.points["L11"])
            # horizontal reference + L10→L11 line
            p.setPen(QPen(QColor(255, 255, 255, 150), 1.5, Qt.DashLine))
            dx = b.x() - a.x()
            reach = max(80.0, abs(dx) + 30)
            p.drawLine(a, QPointF(a.x() + (reach if dx >= 0 else -reach), a.y()))
            p.setPen(QPen(QColor("#facc15"), 2.5))
            p.drawLine(a, b)
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
            for name, pt in (("L10", a), ("L11", b)):
                p.setPen(QPen(Qt.black, 2))
                p.setBrush(POINT_COLORS[name])
                p.drawEllipse(pt, 7, 7)
                p.setPen(Qt.white)
                p.drawText(pt + QPointF(10, -10), name)
            p.end()
            self.setPixmap(canvas)
        else:
            self.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()

    def _hit(self, pos):
        for name, (x, y) in self.points.items():
            q = self.to_widget(x, y)
            if (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2 <= 14 ** 2:
                return name
        return None

    def mousePressEvent(self, e):
        if self._rect and self.show_points and e.button() == Qt.LeftButton:
            self._drag = self._hit(e.position())

    def mouseMoveEvent(self, e):
        if self._drag:
            x, y = self.to_image(e.position())
            self.points[self._drag] = (x, y)
            self.refresh()
            self.point_moved.emit(self._drag, x, y)
        elif self._rect and self.show_points:
            self.setCursor(Qt.OpenHandCursor if self._hit(e.position()) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.dropped.emit(urls[0].toLocalFile())


def muted(text, size=12):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color:#6b7280;font-size:{size}px;")
    return lab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flatfoot X-ray Screener")
        self.session, self.landmarks, self.landmark_note = None, None, ""
        self.img, self.name, self.p, self.cam, self.auto_points = None, "", None, None, None
        self.left = False
        self.job, self.workers = 0, set()
        self._layers = None  # (img, cam, max_w, base pixmap, heat pixmap) cache for redraw()

        self.view = ImageView()
        self.view.setMouseTracking(True)
        self.view.dropped.connect(self.open_path)
        self.view.point_moved.connect(lambda *_: self.update_angle(edited=True))

        open_btn = QPushButton("Open X-ray…")
        open_btn.clicked.connect(self.pick_file)
        open_btn.setMinimumHeight(36)
        self.remove_btn = QPushButton("Remove photo")
        self.remove_btn.clicked.connect(self.clear_image)
        self.remove_btn.setMinimumHeight(36)
        self.remove_btn.setEnabled(False)

        self.side = QComboBox()
        self.side.addItems(["Auto-detect", "Right foot", "Left foot"])
        self.side.currentIndexChanged.connect(lambda _: self.img is not None and self.run_model())
        self.side_note = muted("")

        self.status = QLabel("Loading model…")
        self.status.setStyleSheet("color:#6b7280;")

        self.chip = QLabel()
        self.title = QLabel()
        self.title.setStyleSheet("font-size:20px;font-weight:600;")
        self.title.setWordWrap(True)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.prob = QLabel()
        self.prob.setStyleSheet("font-family:Consolas,monospace;font-size:14px;")
        self.info = muted("")

        # L10–L11 angle section
        self.angle_title = QLabel("L10 → L11 angle")
        self.angle_title.setStyleSheet("font-weight:600;")
        self.angle_value = QLabel("–")
        self.angle_value.setStyleSheet("font-size:26px;font-weight:600;font-family:Consolas,monospace;")
        self.angle_note = muted("")
        self.points_toggle = QCheckBox("Show L10 / L11 points")
        self.points_toggle.setChecked(True)
        self.points_toggle.toggled.connect(self.toggle_points)
        self.reset_pts_btn = QPushButton("Reset points")
        self.reset_pts_btn.clicked.connect(self.reset_points)
        self.reset_pts_btn.setEnabled(False)

        self.heat_toggle = QCheckBox("Show heatmap")
        self.heat_toggle.setChecked(True)
        self.heat_toggle.toggled.connect(self.redraw)
        self.heat_alpha = QSlider(Qt.Horizontal)
        self.heat_alpha.setRange(0, 100)
        self.heat_alpha.setValue(45)
        self.heat_alpha.valueChanged.connect(self.redraw)

        explain = muted("Check the heatmap: the red should sit on the heel and midfoot bones. "
                        "If it lands on a text marker or empty background, don't trust this result.")
        disclaimer = muted("Screening AI Model, not a medical diagnosis.", 11)

        def hline():
            f = QFrame()
            f.setFrameShape(QFrame.HLine)
            return f

        side = QVBoxLayout()
        buttons = QHBoxLayout()
        buttons.addWidget(open_btn)
        buttons.addWidget(self.remove_btn)
        side.addLayout(buttons)
        side.addWidget(self.status)
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Foot side"))
        side_row.addWidget(self.side, 1)
        side.addLayout(side_row)
        side.addWidget(self.side_note)
        side.addWidget(hline())
        side.addWidget(self.chip, alignment=Qt.AlignLeft)
        for w in (self.title, self.sub, self.prob, self.info, hline(),
                  self.angle_title, self.angle_value, self.angle_note):
            side.addWidget(w)
        pts_row = QHBoxLayout()
        pts_row.addWidget(self.points_toggle)
        pts_row.addWidget(self.reset_pts_btn)
        side.addLayout(pts_row)
        side.addWidget(hline())
        for w in (self.heat_toggle, QLabel("Heatmap opacity"), self.heat_alpha, explain):
            side.addWidget(w)
        side.addStretch()
        side.addWidget(disclaimer)
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(320)

        root = QHBoxLayout()
        root.addWidget(self.view, 1)
        root.addWidget(panel)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1100, 720)

        QShortcut(QKeySequence.Paste, self, activated=self.paste)
        QShortcut(QKeySequence.Open, self, activated=self.pick_file)
        QShortcut(QKeySequence.Delete, self, activated=self.clear_image)

        self.set_verdict("idle", "No image yet", "Open a JPG or PNG foot X-ray to begin.")
        self.update_angle()
        self.loader = ModelLoader()
        self.loader.loaded.connect(self.on_loaded)
        self.loader.failed.connect(self.on_failed)
        self.loader.start()

    # --- model lifecycle ---
    def on_loaded(self, session, landmarks, landmark_note=""):
        self.session, self.landmarks, self.landmark_note = session, landmarks, landmark_note
        self.status.setText("● Models ready · runs on this PC" if landmarks else "● Model ready · runs on this PC")
        self.status.setStyleSheet("color:#16a34a;")
        if self.img is not None:
            self.run_model()
        else:
            self.update_angle()

    def on_failed(self, msg):
        self.status.setText("Model failed to load")
        self.status.setStyleSheet("color:#dc2626;")
        QMessageBox.critical(self, "Model error", f"Could not load the model files:\n{msg}")

    # --- input ---
    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open foot X-ray", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self.open_path(path)

    def paste(self):
        qimg = QGuiApplication.clipboard().image()
        if not qimg.isNull():
            qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
            arr = np.frombuffer(qimg.constBits(), np.uint8).reshape(qimg.height(), qimg.bytesPerLine())
            arr = arr[:, :qimg.width() * 4].reshape(qimg.height(), qimg.width(), 4)
            self.set_image(Image.fromarray(arr.copy(), "RGBA"), "pasted image")
        else:
            urls = QGuiApplication.clipboard().mimeData().urls()
            if urls:
                self.open_path(urls[0].toLocalFile())

    def open_path(self, path):
        if not path.lower().endswith(IMAGE_EXTS):
            self.set_verdict("idle", "That file isn't an image", "Choose a JPG or PNG X-ray.")
            return
        try:
            img = Image.open(path)
            img.load()
            img = normalize_input(img)
        except Exception:  # noqa: BLE001
            self.set_verdict("idle", "This image couldn't be opened",
                             "Try saving it as JPG or PNG and open it again.")
            return
        self.set_image(img, os.path.basename(path))

    def set_image(self, img, name):
        self.img, self.name, self.p, self.cam, self.auto_points = img, name, None, None, None
        self.view.points = {}
        self.info.setText(f"{name}\n{img.width} × {img.height}px")
        self.remove_btn.setEnabled(True)
        self.run_model()

    def clear_image(self):
        self.job += 1  # discard any analysis still running for the removed image
        self.img, self.name, self.p, self.cam, self.auto_points = None, "", None, None, None
        self._layers = None
        self.view.clear_content()
        self.info.setText("")
        self.side_note.setText("")
        self.remove_btn.setEnabled(False)
        self.set_verdict("idle", "No image yet", "Open a JPG or PNG foot X-ray to begin.")
        self.update_angle()

    # --- inference ---
    def run_model(self):
        if self.session is None:
            self.set_verdict("idle", "Waiting for the model",
                             "Your image is ready. The result appears as soon as the model finishes loading.")
            self.redraw()
            return
        # show the plain image now; the result is filled in when the worker reports back
        self.job += 1
        self.p, self.cam, self.auto_points = None, None, None
        self.view.points = {}
        self.side_note.setText("")
        self.set_verdict("idle", "Analysing…", "Running the models on this PC.")
        self.update_angle()
        self.redraw()
        worker = AnalysisWorker(self.job, self.img, self.side.currentIndex(), self.session, self.landmarks)
        worker.done.connect(self.on_analysed)
        worker.failed.connect(self.on_analysis_failed)
        worker.finished.connect(lambda: self.workers.discard(worker))
        self.workers.add(worker)
        worker.start()

    @property
    def analysing(self):
        return bool(self.workers)

    def on_analysis_failed(self, job, msg):
        if job != self.job:  # a newer image or side choice has replaced this one
            return
        self.set_verdict("idle", "Analysis failed", msg)

    def on_analysed(self, job, res):
        if job != self.job:
            return
        self.p, self.cam, self.left = res.p, res.cam, res.left
        self.side_note.setText(f"{'Left' if res.left else 'Right'} foot, {res.how}."
                               + (" Image mirrored for the models." if res.left else ""))
        if res.points is not None:
            self.auto_points = res.points
            self.view.points = dict(res.points)
        self.set_verdict(*verdict(self.p))
        self.prob.setText(f"p(pes planus)     = {self.p:.3f}\np(not pes planus) = {1 - self.p:.3f}")
        self.update_angle()
        self.redraw()

    # --- angle ---
    def update_angle(self, edited=False):
        pts = self.view.points
        self.reset_pts_btn.setEnabled(bool(pts) and pts != self.auto_points)
        if not pts:
            self.angle_value.setText("–")
            self.angle_note.setText("" if self.landmarks or self.session is None else self.landmark_note)
            return
        a = line_angle(pts["L10"], pts["L11"], self.left)
        m = self.landmarks.meta
        t = m["threshold_deg"]
        self.angle_value.setText(f"{a:.1f}°")
        lean = "pes planus" if a < t else "a normal arch"
        color = "#dc2626" if a < t else "#16a34a"
        self.angle_value.setStyleSheet(f"font-size:26px;font-weight:600;"
                                       f"font-family:Consolas,monospace;color:{color};")
        source = "Adjusted by you." if edited or pts != self.auto_points else "Points found automatically."
        self.angle_note.setText(
            f"Angle of the L10–L11 line against the horizontal. {a:.1f}° is "
            f"{'below' if a < t else 'above'} the {t:.1f}° cut-off, consistent with {lean}. "
            f"Training averages: pes planus {m['pes_planus_mean_deg']}°, normal {m['normal_mean_deg']}°.\n"
            f"{source} Drag a point to correct it.")

    def toggle_points(self, on):
        self.view.show_points = on
        self.view.refresh()

    def reset_points(self):
        if self.auto_points:
            self.view.points = dict(self.auto_points)
            self.view.refresh()
            self.update_angle()

    # --- display ---
    def set_verdict(self, kind, title, sub):
        text, color = VERDICT_STYLE[kind]
        self.chip.setText(text)
        self.chip.setStyleSheet(f"background:{color};color:white;border-radius:10px;"
                                "padding:4px 10px;font-weight:600;")
        self.title.setText(title)
        self.sub.setText(sub)
        if kind == "idle":
            self.prob.setText("p(pes planus)     = –\np(not pes planus) = –")

    def redraw(self):
        if self.img is None:
            return
        # render at 1000 logical px wide, i.e. more physical pixels on high-DPI screens
        max_w = round(1000 * self.view.devicePixelRatioF())
        c = self._layers
        if c is None or c[0] is not self.img or c[1] is not self.cam or c[2] != max_w:
            base, heat = render_layers(self.img, self.cam, max_w)
            c = self._layers = (self.img, self.cam, max_w, to_pixmap(base),
                                None if heat is None else to_pixmap(heat))
        alpha = self.heat_alpha.value() / 100 if self.heat_toggle.isChecked() else 0
        self.view.set_content(blend(c[3], c[4], alpha), self.img.size)

    def closeEvent(self, e):
        # Qt aborts the process if a QThread is destroyed while still running
        for t in (self.loader, *self.workers):
            t.wait()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_path(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
