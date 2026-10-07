"""The black X-ray lightbox on each patient page.

XrayCanvas paints the X-ray with its heatmaps, the model's points (draggable), rulers and
HUD text. Drawer wraps it with the toolbar and the strip of the patient's X-rays.
All points are kept in original image pixels, so they match what the models return.
"""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSlider, QVBoxLayout, QWidget)

import flatfoot_app as ff

from . import theme
from .engine import resource
from .widgets import set_on

BUNION_LINES = ((0, 1, "TOE", QColor("#5fd4ff")), (2, 3, "MT1", QColor("#ffb547")), (4, 5, "MT2", QColor("#f472b6")))
ARCH_COLOURS = {"L10": QColor("#5fd4ff"), "L11": QColor("#f472b6")}
RULER = 22
HIT = 13


def to_pixmap_gray(gray):
    h, w = gray.shape
    gray = np.ascontiguousarray(gray)
    return QPixmap.fromImage(QImage(gray.data, w, h, w, QImage.Format_Grayscale8).copy())


def heat_pixmap(heat):
    """Jet colours with transparency that follows the heat, so cold areas stay clear."""
    heat = np.clip(np.asarray(heat, np.float32), 0, 1)
    rgb = (ff.jet(heat) * 255).round().astype(np.uint8)
    a = (np.clip(heat * 1.4, 0, 1) * 255).astype(np.uint8)
    rgba = np.ascontiguousarray(np.dstack([rgb, a]))
    h, w = heat.shape
    return QPixmap.fromImage(QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy())


class XrayCanvas(QWidget):
    points_edited = Signal(str, int, object)   # kind ('arch' / 'bunion'), foot index, new points

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 380)
        self.setMouseTracking(True)
        self.pix = None
        self.size_ = (1, 1)
        self.heats = {}          # name -> (QPixmap, (x0, y0, x1, y1) in image pixels)
        self.heat_on = None
        self.heat_alpha = 0.45
        self.arch = None         # {'L10': [x, y], 'L11': [x, y], 'left': bool}
        self.feet = []           # [[6 x [x, y]], ...]
        self.foot_labels = []
        self.heel_box = None
        self.show_points = True
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.hud = {"tl": [], "tr": [], "bl": []}
        self.empty = QPixmap(resource("assistant", "assets", "lightbox.png"))
        self.empty_text = "Add an X-ray to this patient to start"
        self._drag = None
        self._pan_from = None
        self._scan = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ---------- content ----------

    def set_image(self, gray):
        """gray: 2-D uint8 array of the original X-ray, or None to clear."""
        self.pix = to_pixmap_gray(gray) if gray is not None else None
        self.size_ = (gray.shape[1], gray.shape[0]) if gray is not None else (1, 1)
        self.heats, self.arch, self.feet, self.foot_labels, self.heel_box = {}, None, [], [], None
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.update()

    def set_heat(self, name, heat, box=None):
        if heat is None:
            self.heats.pop(name, None)
        else:
            self.heats[name] = (heat_pixmap(heat), box or (0, 0, *self.size_))
        self.update()

    def set_overlays(self, arch=None, feet=(), foot_labels=(), heel_box=None):
        self.arch = {k: list(v) if k != "left" else v for k, v in arch.items()} if arch else None
        self.feet = [[list(p) for p in f] for f in feet]
        self.foot_labels = list(foot_labels)
        self.heel_box = heel_box
        self.update()

    def set_zoom(self, z):
        self.zoom = max(1.0, min(6.0, z))
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        self.update()

    def scanning(self, on):
        if on:
            self._scan = 0.0
            self._timer.start(30)
        else:
            self._scan = None
            self._timer.stop()
        self.update()

    def _tick(self):
        self._scan = (self._scan + 0.012) % 1.0
        self.update()

    # ---------- geometry ----------

    def area(self):
        return QRectF(RULER, RULER, self.width() - RULER - 6, self.height() - RULER - 6)

    def image_rect(self):
        a = self.area()
        w, h = self.size_
        s = min(a.width() / w, a.height() / h) * self.zoom
        c = a.center() + self.pan
        return QRectF(c.x() - w * s / 2, c.y() - h * s / 2, w * s, h * s)

    def to_widget(self, x, y):
        r = self.image_rect()
        return QPointF(r.x() + x * r.width() / self.size_[0], r.y() + y * r.height() / self.size_[1])

    def to_image(self, pos):
        r = self.image_rect()
        x = (pos.x() - r.x()) * self.size_[0] / r.width()
        y = (pos.y() - r.y()) * self.size_[1] / r.height()
        return [min(max(x, 0), self.size_[0] - 1), min(max(y, 0), self.size_[1] - 1)]

    def _handles(self):
        if not self.show_points:
            return
        if self.arch:
            for k in ("L10", "L11"):
                yield ("arch", 0, k), self.arch[k]
        for i, foot in enumerate(self.feet):
            for j, p in enumerate(foot):
                yield ("bunion", i, j), p

    def _hit(self, pos):
        for key, (x, y) in self._handles():
            q = self.to_widget(x, y)
            if (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2 <= HIT ** 2:
                return key
        return None

    # ---------- mouse ----------

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or self.pix is None:
            return
        self._drag = self._hit(e.position())
        if self._drag is None and self.zoom > 1:
            self._pan_from = (e.position(), QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag:
            kind, i, k = self._drag
            p = self.to_image(e.position())
            if kind == "arch":
                self.arch[k] = p
            else:
                self.feet[i][k] = p
            self.update()
        elif self._pan_from:
            start, pan = self._pan_from
            self.pan = pan + (e.position() - start)
            self.update()
        elif self.pix is not None:
            self.setCursor(Qt.OpenHandCursor if self._hit(e.position()) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self._drag:
            kind, i, _ = self._drag
            pts = {"L10": self.arch["L10"], "L11": self.arch["L11"]} if kind == "arch" else self.feet[i]
            self.points_edited.emit(kind, i, pts)
        self._drag = self._pan_from = None
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, e):
        if self.pix is not None:
            self.set_zoom(self.zoom * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15))
            self.zoom_changed()

    def zoom_changed(self):
        """Overridden by Drawer to keep its slider in step."""

    # ---------- painting ----------

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), theme.X_BG)
        a = self.area()
        p.setPen(QPen(theme.X_LINE, 1))
        step = 40
        x = a.left()
        while x < a.right():
            p.drawLine(QPointF(x, a.top()), QPointF(x, a.bottom()))
            x += step
        y = a.top()
        while y < a.bottom():
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
            y += step
        if self.pix is None:
            self._paint_empty(p, a)
        else:
            p.save()
            p.setClipRect(a)
            r = self.image_rect()
            p.drawPixmap(r, self.pix, QRectF(self.pix.rect()))
            if self.heat_on in self.heats:
                pix, (x0, y0, x1, y1) = self.heats[self.heat_on]
                p.setOpacity(self.heat_alpha)
                tl, br = self.to_widget(x0, y0), self.to_widget(x1, y1)
                p.drawPixmap(QRectF(tl, br), pix, QRectF(pix.rect()))
                p.setOpacity(1)
            if self.show_points:
                self._paint_heel(p)
                self._paint_arch(p)
                self._paint_feet(p)
            if self._scan is not None:
                yy = r.top() + self._scan * r.height()
                p.setPen(QPen(theme.X_CYAN, 2))
                p.drawLine(QPointF(r.left(), yy), QPointF(r.right(), yy))
                glow = QColor(theme.X_CYAN)
                glow.setAlpha(40)
                p.fillRect(QRectF(r.left(), yy - 18, r.width(), 18), glow)
            p.restore()
        self._paint_rulers(p, a)
        self._paint_hud(p, a)
        p.end()

    def _paint_empty(self, p, a):
        if not self.empty.isNull():
            s = min(a.width() / self.empty.width(), a.height() / self.empty.height())
            w, h = self.empty.width() * s, self.empty.height() * s
            p.setOpacity(0.55)
            p.drawPixmap(QRectF(a.center().x() - w / 2, a.center().y() - h / 2, w, h), self.empty,
                         QRectF(self.empty.rect()))
            p.setOpacity(1)
        p.setPen(theme.X_INK)
        p.setFont(QFont(theme.MONO, 11))
        p.drawText(a.adjusted(0, 0, 0, -24), Qt.AlignHCenter | Qt.AlignBottom, self.empty_text)

    def _label(self, p, at, text, colour):
        p.setFont(QFont(theme.MONO, 9, QFont.Bold))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 10
        box = QRectF(at.x(), at.y() - fm.height() - 2, w, fm.height() + 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(2, 4, 7, 200))
        p.drawRoundedRect(box, 4, 4)
        p.setPen(colour)
        p.drawText(box, Qt.AlignCenter, text)

    def _dot(self, p, pt, colour):
        p.setPen(QPen(QColor(2, 4, 7), 2))
        p.setBrush(colour)
        p.drawEllipse(pt, 6.5, 6.5)
        glow = QColor(colour)
        glow.setAlpha(70)
        p.setPen(QPen(glow, 5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(pt, 10, 10)

    def _paint_heel(self, p):
        if not self.heel_box:
            return
        x0, y0, x1, y1 = self.heel_box
        tl, br = self.to_widget(x0, y0), self.to_widget(x1, y1)
        p.setPen(QPen(theme.X_AMBER, 1.5, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(tl, br))
        self._label(p, QPointF(tl.x(), tl.y() - 2), "HEEL", theme.X_AMBER)

    def _paint_arch(self, p):
        if not self.arch:
            return
        a, b = self.to_widget(*self.arch["L10"]), self.to_widget(*self.arch["L11"])
        dx = b.x() - a.x()
        reach = max(80.0, abs(dx) + 30)
        p.setPen(QPen(QColor(220, 243, 255, 150), 1.2, Qt.DashLine))
        p.drawLine(a, QPointF(a.x() + (reach if dx >= 0 else -reach), a.y()))
        p.setPen(QPen(theme.X_CYAN, 2.4))
        p.drawLine(a, b)
        ang = ff.line_angle(self.arch["L10"], self.arch["L11"], self.arch.get("left", False))
        for k, pt in (("L10", a), ("L11", b)):
            self._dot(p, pt, ARCH_COLOURS[k])
        self._label(p, QPointF(a.x() + 12, a.y() - 12), f"PITCH {ang:.1f}°", theme.X_INK)

    def _paint_feet(self, p):
        for i, foot in enumerate(self.feet):
            pts = [self.to_widget(*q) for q in foot]
            for j0, j1, name, colour in BUNION_LINES:
                a, b = pts[j0], pts[j1]
                d = b - a
                n = math.hypot(d.x(), d.y()) or 1
                ext = d * (25 / n)
                p.setPen(QPen(colour, 2.2))
                p.drawLine(a - ext, b + ext)
            for j0, j1, name, colour in BUNION_LINES:
                self._dot(p, pts[j0], colour)
                self._dot(p, pts[j1], colour)
            if i < len(self.foot_labels):
                top = min(pts, key=lambda q: q.y())
                self._label(p, QPointF(top.x() + 14, top.y() - 6), self.foot_labels[i], theme.X_INK)

    def _paint_rulers(self, p, a):
        p.fillRect(QRectF(0, 0, self.width(), RULER), theme.X_BG)
        p.fillRect(QRectF(0, 0, RULER, self.height()), theme.X_BG)
        p.setPen(QPen(theme.X_MUTED, 1))
        p.setFont(QFont(theme.MONO, 7))
        if self.pix is None:
            unit, origin = 40.0, QPointF(a.left(), a.top())
        else:
            r = self.image_rect()
            px = r.width() / self.size_[0]            # widget px per image px
            raw = 50 / px
            mag = 10 ** math.floor(math.log10(raw))
            nice = min((m * mag for m in (1, 2, 5, 10)), key=lambda v: abs(v - raw))
            unit, origin = nice * px, r.topLeft()
        i = math.floor((a.left() - origin.x()) / unit)
        while True:
            x = origin.x() + i * unit
            if x > a.right():
                break
            if x >= a.left():
                p.drawLine(QPointF(x, RULER - 7), QPointF(x, RULER))
                for k in range(1, 5):
                    xx = x + k * unit / 5
                    if xx <= a.right():
                        p.drawLine(QPointF(xx, RULER - 3), QPointF(xx, RULER))
            i += 1
        i = math.floor((a.top() - origin.y()) / unit)
        while True:
            y = origin.y() + i * unit
            if y > a.bottom():
                break
            if y >= a.top():
                p.drawLine(QPointF(RULER - 7, y), QPointF(RULER, y))
                for k in range(1, 5):
                    yy = y + k * unit / 5
                    if yy <= a.bottom():
                        p.drawLine(QPointF(RULER - 3, yy), QPointF(RULER, yy))
            i += 1

    def _paint_hud(self, p, a):
        p.setFont(QFont(theme.MONO, 9))
        fm = p.fontMetrics()
        lh = fm.height() + 2
        pad = 12
        colours = {"red": theme.X_RED, "amber": theme.X_AMBER, "green": QColor("#6ee7a8")}
        z = f"MAG {self.zoom:.1f}x"
        blocks = (("tl", [(t, theme.X_INK if i == 0 else theme.X_MUTED) for i, t in enumerate(self.hud["tl"])]),
                  ("tr", [(t, theme.X_MUTED) for t in self.hud["tr"]]),
                  ("bl", [(t, colours.get(c, theme.X_INK)) for t, c in self.hud["bl"]]),
                  ("br", [(z, theme.X_MUTED)]))
        for corner, rows in blocks:
            if not rows:
                continue
            w = max(fm.horizontalAdvance(t) for t, _ in rows) + 16
            h = lh * len(rows) + 8
            x = a.left() + pad if corner[1] == "l" else a.right() - pad - w
            y = a.top() + pad if corner[0] == "t" else a.bottom() - pad - h
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(2, 4, 7, 185))
            p.drawRoundedRect(QRectF(x, y, w, h), 6, 6)
            for i, (t, colour) in enumerate(rows):
                p.setPen(colour)
                p.drawText(QPointF(x + 8, y + 4 + lh * i + fm.ascent() + 1), t)


class Thumb(QPushButton):
    def __init__(self, xray_id, pix, caption):
        super().__init__()
        self.xray_id = xray_id
        self.setCheckable(False)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(132, 104)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 4)
        lay.setSpacing(2)
        img = QLabel()
        img.setAlignment(Qt.AlignCenter)
        img.setPixmap(pix.scaled(118, 68, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        img.setStyleSheet("background: transparent; border: none;")
        cap = QLabel(caption)
        cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet("background: transparent; border: none; font-size: 10px;")
        lay.addWidget(img)
        lay.addWidget(cap)


class Drawer(QFrame):
    """Black lightbox: toolbar, canvas and the X-ray strip."""
    picked = Signal(int)
    add_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("drawer")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        title = QLabel("LIGHTBOX")
        title.setStyleSheet("color: #dcf3ff; font-weight: 700; letter-spacing: 2px;")
        bar.addWidget(title)
        bar.addSpacing(10)
        self.heat_buttons = {}
        for key, text in (("off", "NO HEATMAP"), ("arch", "ARCH HEATMAP"), ("heel", "HEEL HEATMAP"),
                          ("break", "FRACTURE HEATMAP")):
            b = QPushButton(text)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.set_heat(k))
            self.heat_buttons[key] = b
            bar.addWidget(b)
        self.points = QCheckBox("POINTS")
        self.points.setChecked(True)
        self.points.toggled.connect(self._toggle_points)
        bar.addSpacing(6)
        bar.addWidget(self.points)
        self.reset = QPushButton("RESET POINTS")
        self.reset.setCursor(Qt.PointingHandCursor)
        self.reset.clicked.connect(self.reset_clicked)
        bar.addWidget(self.reset)
        bar.addStretch(1)
        bar.addWidget(QLabel("MAG"))
        self.zoom = QSlider(Qt.Horizontal)
        self.zoom.setRange(10, 60)
        self.zoom.setFixedWidth(130)
        self.zoom.valueChanged.connect(lambda v: self.canvas.set_zoom(v / 10))
        bar.addWidget(self.zoom)
        lay.addLayout(bar)

        self.canvas = XrayCanvas()
        self.canvas.zoom_changed = lambda: self._sync_zoom()
        lay.addWidget(self.canvas, 1)

        strip = QHBoxLayout()
        strip.setSpacing(8)
        self.thumbs_box = QWidget()
        self.thumbs_box.setStyleSheet("background: transparent;")
        self.thumbs = QHBoxLayout(self.thumbs_box)
        self.thumbs.setContentsMargins(0, 0, 0, 0)
        self.thumbs.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidget(self.thumbs_box)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(120)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        strip.addWidget(scroll, 1)
        self.add = QPushButton("+  ADD X-RAY")
        self.add.setCursor(Qt.PointingHandCursor)
        self.add.setFixedSize(132, 104)
        self.add.clicked.connect(self.add_clicked)
        strip.addWidget(self.add)
        lay.addLayout(strip)
        self.set_heat("off")
        self._thumb_buttons = []

    def _sync_zoom(self):
        self.zoom.blockSignals(True)
        self.zoom.setValue(round(self.canvas.zoom * 10))
        self.zoom.blockSignals(False)

    def _toggle_points(self, on):
        self.canvas.show_points = on
        self.canvas.update()

    def set_heat(self, key):
        for k, b in self.heat_buttons.items():
            set_on(b, k == key)
        self.canvas.heat_on = None if key == "off" else key
        self.canvas.update()

    def available_heats(self, names):
        for k, b in self.heat_buttons.items():
            b.setVisible(k == "off" or k in names)
        if self.canvas.heat_on not in names:
            self.set_heat("off")

    def set_thumbs(self, items, current=None):
        """items: [(xray_id, QPixmap, caption)]"""
        while self.thumbs.count():
            w = self.thumbs.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._thumb_buttons = []
        for xid, pix, cap in items:
            t = Thumb(xid, pix, cap)
            t.clicked.connect(lambda _=False, i=xid: self.picked.emit(i))
            set_on(t, xid == current)
            self.thumbs.addWidget(t)
            self._thumb_buttons.append(t)
        self.thumbs.addStretch(1)

    def mark_current(self, xray_id):
        for t in self._thumb_buttons:
            set_on(t, t.xray_id == xray_id)
