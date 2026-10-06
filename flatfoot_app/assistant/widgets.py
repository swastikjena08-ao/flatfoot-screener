"""Small building blocks shared by the pages: cards, chips, buttons, labelled fields."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from . import theme


def restyle(w):
    w.style().unpolish(w)
    w.style().polish(w)


def set_on(button, on):
    button.setProperty("on", "true" if on else "false")
    restyle(button)


def label(text="", *, muted=False, h1=False, h2=False, big=False, err=False, size=None, wrap=False):
    w = QLabel(text)
    for name, on in (("muted", muted), ("h1", h1), ("h2", h2), ("big", big), ("err", err)):
        if on:
            w.setProperty(name, "true")
    if size:
        w.setStyleSheet(f"font-size: {size}px;")
    w.setWordWrap(wrap)
    return w


def button(text, kind=None, on_click=None, tip=""):
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if kind:
        b.setProperty("kind", kind)
    if on_click:
        b.clicked.connect(on_click)
    if tip:
        b.setToolTip(tip)
    return b


def card(row=False, margins=(20, 18, 20, 18), spacing=10):
    f = QFrame()
    f.setProperty("row" if row else "card", "true")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return f, lay


def chip(text, colour="grey"):
    bg, fg = theme.CHIPS.get(colour, theme.CHIPS["grey"])
    w = QLabel(text)
    w.setStyleSheet(f"background: {bg}; color: {fg}; border-radius: 12px; padding: 3px 10px;"
                    " font-size: 12px; font-weight: 600;")
    w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return w


def chips(items):
    """A row of chips from [(text, colour)]."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for text, colour in items:
        lay.addWidget(chip(text, colour))
    lay.addStretch(1)
    return box


def field(title, widget):
    """A small grey caption over an input."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    cap = label(title, muted=True, size=12)
    lay.addWidget(cap)
    lay.addWidget(widget)
    return box


def line_edit(placeholder="", password=False):
    e = QLineEdit()
    e.setPlaceholderText(placeholder)
    if password:
        e.setEchoMode(QLineEdit.Password)
    return e


def clear_layout(lay):
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def hrow(*widgets, spacing=10, stretch_at=None):
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for i, w in enumerate(widgets):
        if i == stretch_at:
            lay.addStretch(1)
        if w is None:
            lay.addStretch(1)
        else:
            lay.addWidget(w)
    return box


def pct(v):
    return f"{v * 100:.1f}%"
