"""The black X-ray lightbox on each patient page.

XrayCanvas paints the X-ray with its heatmaps, the model's points (draggable), rulers and
HUD text. Drawer wraps it with the toolbar and the strip of the patient's X-rays.
All points are kept in original image pixels, so they match what the models return.
"""
import math
import textwrap
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
                               QSlider, QVBoxLayout, QWidget)

import flatfoot_app as ff

from . import theme
from .widgets import set_on

BUNION_LINES = ((0, 1, "TOE", QColor("#5fd4ff")), (2, 3, "MT1", QColor("#ffb547")), (4, 5, "MT2", QColor("#f472b6")))
ARCH_COLOURS = {"L10": QColor("#5fd4ff"), "L11": QColor("#f472b6")}
RULER = 22
SCAN_SECONDS = 5.0
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


PEN_COLOURS = ("#ffd60a", "#ff4d5e", "#5fd4ff", "#ffffff")
PIN = 11


class XrayCanvas(QWidget):
    points_edited = Signal(str, int, object)   # kind ('arch' / 'bunion'), foot index, new points
    marks_changed = Signal(object)             # the doctor's drawings and notes: {"strokes": [...], "notes": [...]}
    note_clicked = Signal(int, float, float)   # note index (-1 for a new one), position in image pixels

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
        self.empty_text = "Add an X-ray to this patient to start"
        self._drag = None
        self._pan_from = None
        self._fixed = None       # image rect while rendering a marked copy
        self.marks = {"strokes": [], "notes": []}
        self.show_marks = True
        self.tool = None         # None (move points), "pen" or "note"
        self.pen_colour = PEN_COLOURS[0]
        self._stroke = None
        self._note_drag = None   # [index, press position, moved yet]
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

    def set_marks(self, marks):
        m = marks or {}
        self.marks = {"strokes": [dict(st) for st in m.get("strokes", [])], "notes": [dict(n) for n in m.get("notes", [])]}
        self.update()

    def set_tool(self, tool):
        self.tool = tool
        self.setCursor(Qt.CrossCursor if tool else Qt.ArrowCursor)

    def undo_stroke(self):
        if self.marks["strokes"]:
            self.marks["strokes"].pop()
            self.update()
            self.marks_changed.emit(self.marks)

    def clear_marks(self):
        self.marks = {"strokes": [], "notes": []}
        self.update()
        self.marks_changed.emit(self.marks)

    def _scale(self):
        return self.image_rect().width() / self.size_[0]

    def _hit_note(self, pos):
        for i, n in enumerate(self.marks["notes"]):
            q = self.to_widget(n["x"], n["y"])
            if (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2 <= (PIN + 3) ** 2:
                return i
        return None

    def set_zoom(self, z):
        self.zoom = max(1.0, min(6.0, z))
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        self.update()

    def scanning(self, on, start=None):
        """Sweep a line down the whole X-ray, top to bottom, every SCAN_SECONDS while the models run.
        start: time.monotonic() when this X-ray's check began, so switching X-rays keeps the sweep in step."""
        if on and self._scan is None:
            self._scan_start = start or time.monotonic()
            self._scan = 0.0
            self._timer.start(16)
        elif not on:
            self._scan = None
            self._timer.stop()
        self.update()

    def scan_elapsed(self):
        return 0.0 if self._scan is None else time.monotonic() - self._scan_start

    def _tick(self):
        self._scan = (self.scan_elapsed() % SCAN_SECONDS) / SCAN_SECONDS
        self.update()

    # ---------- geometry ----------

    def area(self):
        return QRectF(RULER, RULER, self.width() - RULER - 6, self.height() - RULER - 6)

    def image_rect(self):
        if self._fixed is not None:
            return self._fixed
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
        i = self._hit_note(e.position()) if self.show_marks else None
        if i is not None:
            # a note pin: drag to move it, or a plain click to edit its text (on release)
            self._note_drag = [i, e.position(), False]
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self.tool == "pen":
            self.show_marks = True
            self._stroke = {"c": self.pen_colour, "w": 3.0 / self._scale(), "pts": [self.to_image(e.position())]}
            self.marks["strokes"].append(self._stroke)
            self.update()
            return
        if self.tool == "note":
            self.show_marks = True
            x, y = self.to_image(e.position())
            self.note_clicked.emit(-1, x, y)
            return
        self._drag = self._hit(e.position())
        if self._drag is None and self.zoom > 1:
            self._pan_from = (e.position(), QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._note_drag is not None:
            i, start, moved = self._note_drag
            d = e.position() - start
            if moved or d.x() ** 2 + d.y() ** 2 > 16:
                self._note_drag[2] = True
                x, y = self.to_image(e.position())
                self.marks["notes"][i].update(x=round(x, 1), y=round(y, 1))
                self.update()
        elif self._stroke is not None:
            self._stroke["pts"].append(self.to_image(e.position()))
            self.update()
        elif self._drag:
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
            over = (self.show_marks and self._hit_note(e.position()) is not None) or \
                (not self.tool and self._hit(e.position()))
            self.setCursor(Qt.OpenHandCursor if over else Qt.CrossCursor if self.tool else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self._note_drag is not None:
            i, _, moved = self._note_drag
            self._note_drag = None
            self.setCursor(Qt.OpenHandCursor)
            if moved:
                self.marks_changed.emit(self.marks)
            else:
                n = self.marks["notes"][i]
                self.note_clicked.emit(i, n["x"], n["y"])
            return
        if self._stroke is not None:
            st = self._stroke
            self._stroke = None
            st["pts"] = [[round(x, 1), round(y, 1)] for x, y in st["pts"]]
            self.marks_changed.emit(self.marks)
            return
        if self._drag:
            kind, i, _ = self._drag
            pts = {"L10": self.arch["L10"], "L11": self.arch["L11"]} if kind == "arch" else self.feet[i]
            self.points_edited.emit(kind, i, pts)
        self._drag = self._pan_from = None
        self.setCursor(Qt.CrossCursor if self.tool else Qt.ArrowCursor)

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
            self._paint_content(p, r)
            if self._scan is not None:
                yy = r.top() + self._scan * r.height()
                glow = QColor(theme.X_CYAN)
                glow.setAlpha(45)
                p.fillRect(QRectF(r.left(), yy - 26, r.width(), 26), glow)
                p.setPen(QPen(theme.X_CYAN, 2))
                p.drawLine(QPointF(r.left(), yy), QPointF(r.right(), yy))
            p.restore()
        self._paint_rulers(p, a)
        self._paint_hud(p, a)
        p.end()

    def _paint_content(self, p, r):
        p.drawPixmap(r, self.pix, QRectF(self.pix.rect()))
        if self.heat_on in self.heats:
            pix, (x0, y0, x1, y1) = self.heats[self.heat_on]
            p.setOpacity(self.heat_alpha)
            p.drawPixmap(QRectF(self.to_widget(x0, y0), self.to_widget(x1, y1)), pix, QRectF(pix.rect()))
            p.setOpacity(1)
        if self.show_points:
            self._paint_heel(p)
            self._paint_arch(p)
            self._paint_feet(p)
        if self.show_marks:
            self._paint_marks(p)

    def _paint_marks(self, p):
        s = self._scale()
        for st in self.marks["strokes"]:
            pts = [self.to_widget(x, y) for x, y in st["pts"]]
            pen = QPen(QColor(st["c"]), max(1.5, st["w"] * s), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            if len(pts) == 1:
                p.drawPoint(pts[0])
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        p.setFont(QFont(theme.MONO, 9, QFont.Bold))
        fm = p.fontMetrics()
        for i, n in enumerate(self.marks["notes"]):
            q = self.to_widget(n["x"], n["y"])
            p.setPen(QPen(QColor(2, 4, 7), 2))
            p.setBrush(QColor("#ffd60a"))
            p.drawEllipse(q, PIN, PIN)
            p.setPen(QColor(2, 4, 7))
            p.drawText(QRectF(q.x() - PIN, q.y() - PIN, 2 * PIN, 2 * PIN), Qt.AlignCenter, str(i + 1))
            text = n.get("text", "")
            if text:
                lines = textwrap.wrap(text, 30)[:5]
                w = max(fm.horizontalAdvance(t) for t in lines) + 14
                h = fm.height() * len(lines) + 8
                right = self._fixed.right() if self._fixed is not None else self.area().right()
                x0 = q.x() + PIN + 6 if q.x() + PIN + 6 + w <= right else q.x() - PIN - 6 - w
                box = QRectF(x0, q.y() - h / 2, w, h)
                p.setPen(QPen(QColor("#ffd60a"), 1))
                p.setBrush(QColor(2, 4, 7, 215))
                p.drawRoundedRect(box, 5, 5)
                p.setPen(QColor("#fff6cc"))
                for k, t in enumerate(lines):
                    p.drawText(QPointF(box.x() + 7, box.y() + 4 + fm.ascent() + k * fm.height()), t)

    def render_marked(self, long_side=1600):
        """The X-ray with the heatmap shown now, the points, lines and findings drawn on, as a QImage."""
        if self.pix is None:
            return None
        w, h = self.size_
        s = long_side / max(w, h)
        out = QImage(round(w * s), round(h * s), QImage.Format_RGB32)
        out.fill(theme.X_BG)
        self._fixed = QRectF(0, 0, out.width(), out.height())
        try:
            p = QPainter(out)
            p.setRenderHint(QPainter.Antialiasing)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            self._paint_content(p, self._fixed)
            hud, self.hud = self.hud, {**self.hud, "br": []}
            zoom, self.zoom = self.zoom, None
            self._paint_hud(p, QRectF(0, 0, out.width(), out.height()))
            self.hud, self.zoom = hud, zoom
            p.end()
        finally:
            self._fixed = None
        return out

    def _paint_empty(self, p, a):
        p.setPen(theme.X_MUTED)
        p.setFont(QFont(theme.MONO, 11))
        p.drawText(a, Qt.AlignCenter, self.empty_text)

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
        z = f"MAG {self.zoom:.1f}x" if self.zoom and self.pix is not None else None
        blocks = (("tl", [(t, theme.X_INK if i == 0 else theme.X_MUTED) for i, t in enumerate(self.hud["tl"])]),
                  ("tr", [(t, theme.X_MUTED) for t in self.hud["tr"]]),
                  ("bl", [(t, colours.get(c, theme.X_INK)) for t, c in self.hud["bl"]]),
                  ("br", [(z, theme.X_MUTED)] if z else []))
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
    removed = Signal(int)

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
        self.close_btn = QPushButton("×", self)
        self.close_btn.setObjectName("thumbx")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setToolTip("Delete this X-ray")
        self.close_btn.setFixedSize(22, 22)
        self.close_btn.move(self.width() - 26, 4)
        self.close_btn.setStyleSheet(
            "QPushButton#thumbx { min-height: 22px; max-height: 22px; padding: 0; border-radius: 11px;"
            " background: rgba(2,4,7,0.85); color: #dcf3ff; border: 1px solid rgba(110,205,255,0.45);"
            " font-size: 14px; font-weight: 700; }"
            " QPushButton#thumbx:hover { background: #ff6b78; color: #020407; border-color: #ff6b78; }")
        self.close_btn.clicked.connect(lambda: self.removed.emit(self.xray_id))


class Drawer(QFrame):
    """Black lightbox: toolbar, canvas and the X-ray strip."""
    picked = Signal(int)
    remove_clicked = Signal(int)
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

        tools = QHBoxLayout()
        tools.setSpacing(8)
        tools.addWidget(QLabel("MARK UP"))
        self.tool_buttons = {}
        for key, text in (("pen", "✎  PEN"), ("note", "✚  NOTE")):
            b = QPushButton(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip("Draw on the X-ray" if key == "pen" else "Click the X-ray to pin a typed remark; click a pin to edit it")
            b.clicked.connect(lambda _=False, k=key: self.pick_tool(k))
            self.tool_buttons[key] = b
            tools.addWidget(b)
        self.colour_buttons = {}
        for c in PEN_COLOURS:
            b = QPushButton("")
            b.setObjectName("swatch")
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedSize(24, 24)
            b.clicked.connect(lambda _=False, col=c: self.pick_colour(col))
            self.colour_buttons[c] = b
            tools.addWidget(b)
        self.undo = QPushButton("UNDO LINE")
        self.undo.setCursor(Qt.PointingHandCursor)
        self.clear = QPushButton("CLEAR MARKS")
        self.clear.setCursor(Qt.PointingHandCursor)
        tools.addWidget(self.undo)
        tools.addWidget(self.clear)
        self.show_marks = QCheckBox("SHOW MARKS")
        self.show_marks.setChecked(True)
        self.show_marks.toggled.connect(self._toggle_marks)
        tools.addWidget(self.show_marks)
        tools.addStretch(1)
        lay.addLayout(tools)

        self.canvas = XrayCanvas()
        self.undo.clicked.connect(lambda: self.canvas.undo_stroke())
        self.clear.clicked.connect(self._clear_marks)
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
        self.pick_colour(PEN_COLOURS[0], activate=False)
        self._thumb_buttons = []

    def pick_tool(self, key):
        """Click a tool to use it, click it again to go back to moving the model's points."""
        tool = None if self.canvas.tool == key else key
        self.canvas.set_tool(tool)
        for k, b in self.tool_buttons.items():
            set_on(b, k == tool)
        if tool and not self.show_marks.isChecked():
            self.show_marks.setChecked(True)

    def pick_colour(self, colour, activate=True):
        self.canvas.pen_colour = colour
        for c, b in self.colour_buttons.items():
            ring = "#ffffff" if c == colour else "rgba(110,205,255,0.35)"
            width = 3 if c == colour else 1
            b.setStyleSheet(f"QPushButton#swatch {{ background: {c}; border: {width}px solid {ring}; border-radius: 12px;"
                            " min-height: 24px; max-height: 24px; min-width: 24px; max-width: 24px; padding: 0; }")
        if activate and self.canvas.tool != "pen":
            self.pick_tool("pen")

    def _toggle_marks(self, on):
        self.canvas.show_marks = on
        self.canvas.update()

    def _clear_marks(self):
        if self.canvas.marks["strokes"] or self.canvas.marks["notes"]:
            if QMessageBox.question(self, "Clear marks", "Remove all drawings and notes from this X-ray?") == QMessageBox.Yes:
                self.canvas.clear_marks()

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
            t.removed.connect(self.remove_clicked)
            set_on(t, xid == current)
            self.thumbs.addWidget(t)
            self._thumb_buttons.append(t)
        self.thumbs.addStretch(1)

    def mark_current(self, xray_id):
        for t in self._thumb_buttons:
            set_on(t, t.xray_id == xray_id)
