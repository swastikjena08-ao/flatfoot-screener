"""Colours, fonts and the Qt style sheet: white window, soft grey rounded cards, one blue.
The X-ray viewer is the one dark element (see viewer.py)."""
from PySide6.QtGui import QColor, QFont

INK = "#101114"
MUTED = "#5f6672"
CARD = "#f3f4f6"
CARD_2 = "#e9ebef"
BLUE = "#0f5cf5"
BLUE_SOFT = "#e3ecff"
CHIPS = {  # background, text
    "blue": ("#e3ecff", "#0b45b8"),
    "red": ("#fde7e7", "#a3242a"),
    "amber": ("#fff1dc", "#9a5205"),
    "green": ("#e1f5e8", "#15803d"),
    "grey": ("#e9ebef", "#5f6672"),
}
# X-ray lightbox
X_BG = QColor("#020407")
X_LINE = QColor(110, 205, 255, 60)
X_INK = QColor("#dcf3ff")
X_MUTED = QColor("#8eaebf")
X_CYAN = QColor("#5fd4ff")
X_AMBER = QColor("#ffb547")
X_RED = QColor("#ff6b78")

FAMILY = "Segoe UI"
MONO = "Consolas"
HAND = "Segoe Print"


def font(size=10, weight=QFont.Normal, family=FAMILY):
    f = QFont(family, size)
    f.setWeight(weight)
    return f


STYLE = f"""
* {{ font-family: "{FAMILY}", "Segoe UI Variable", "Helvetica Neue", Arial; color: {INK}; }}
QMainWindow, QWidget#page, QWidget#window {{ background: #ffffff; }}
QDialog {{ background: #ffffff; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QFrame[card="true"] {{ background: {CARD}; border-radius: 22px; }}
QFrame[row="true"] {{ background: #ffffff; border-radius: 16px; }}
QFrame[row="true"]:hover {{ background: #f7f9ff; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[h1="true"] {{ font-size: 30px; font-weight: 500; }}
QLabel[h2="true"] {{ font-size: 16px; font-weight: 600; }}
QLabel[big="true"] {{ font-size: 36px; font-weight: 500; }}
QLabel[err="true"] {{ color: #a3242a; font-weight: 600; }}
QPushButton {{ border: none; border-radius: 22px; min-height: 44px; padding: 0 18px;
              background: {CARD}; font-weight: 600; }}
QPushButton:hover {{ background: {CARD_2}; }}
QPushButton:disabled {{ color: #9aa0aa; }}
QPushButton[kind="primary"] {{ background: {BLUE}; color: #ffffff; }}
QPushButton[kind="primary"]:hover {{ background: #0b4fd6; }}
QPushButton[kind="primary"]:disabled {{ background: #9db8f5; color: #eef3ff; }}
QPushButton[kind="ghost"] {{ background: #ffffff; border: 1.5px solid {CARD_2}; }}
QPushButton[kind="nav"] {{ background: {CARD}; }}
QPushButton[kind="nav"][on="true"] {{ background: #ffffff; border: 1.5px solid {CARD_2}; }}
QPushButton[kind="filter"] {{ background: #ffffff; min-height: 38px; border-radius: 19px; }}
QPushButton[kind="filter"][on="true"] {{ background: {INK}; color: #ffffff; }}
QPushButton[kind="round"] {{ min-width: 44px; max-width: 44px; min-height: 44px; max-height: 44px;
                             padding: 0; border-radius: 22px; }}
QPushButton[kind="round"][on="true"] {{ background: {BLUE}; color: #ffffff; }}
QPushButton[kind="flag"] {{ min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px;
                            padding: 0; border-radius: 20px; color: {MUTED}; }}
QPushButton[kind="flag"][on="true"] {{ background: #fde7e7; color: #d93a3f; }}
QPushButton[kind="link"] {{ background: transparent; color: {BLUE}; padding: 0; min-height: 28px; }}
QPushButton[kind="link"]:hover {{ text-decoration: underline; }}
QLineEdit, QComboBox, QDateEdit {{ background: #ffffff; border: 1.5px solid {CARD_2}; border-radius: 22px;
                                  min-height: 44px; padding: 0 16px; font-size: 14px; }}
QLineEdit:focus, QComboBox:focus, QDateEdit:focus {{ border: 2px solid {BLUE}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QTextEdit, QPlainTextEdit {{ background: #ffffff; border: 1.5px solid {CARD_2}; border-radius: 16px;
                             padding: 8px; font-size: 14px; }}
QTextEdit:focus, QPlainTextEdit:focus {{ border: 2px solid {BLUE}; }}
QCheckBox {{ spacing: 8px; }}
QRadioButton {{ spacing: 8px; }}
QToolTip {{ background: {INK}; color: #ffffff; border: none; padding: 6px; }}
QFrame#drawer {{ background: #020407; border-radius: 22px; }}
QFrame#drawer QLabel {{ color: #8eaebf; font-family: "{MONO}"; font-size: 11px; }}
QFrame#drawer QCheckBox {{ color: #dcf3ff; font-family: "{MONO}"; font-size: 11px; }}
QFrame#drawer QPushButton {{ background: transparent; color: #8eaebf; border: 1px solid rgba(110,205,255,0.35);
                            border-radius: 16px; min-height: 32px; font-family: "{MONO}"; font-size: 11px; }}
QFrame#drawer QPushButton:hover {{ color: #dcf3ff; border-color: #5fd4ff; }}
QFrame#drawer QPushButton[on="true"] {{ background: #5fd4ff; color: #020407; border-color: #5fd4ff; }}
QFrame#drawer QSlider::groove:horizontal {{ height: 4px; background: rgba(110,205,255,0.3); border-radius: 2px; }}
QFrame#drawer QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: #5fd4ff; }}
QScrollBar:vertical {{ width: 10px; background: transparent; }}
QScrollBar::handle:vertical {{ background: {CARD_2}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: rgba(110,205,255,0.35); border-radius: 5px; min-width: 30px; }}
"""
