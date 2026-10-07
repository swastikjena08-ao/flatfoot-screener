"""Colours, fonts and the Qt style sheet: white window, soft grey rounded cards, one blue.
The X-ray viewer is the one dark element (see viewer.py)."""
from PySide6.QtGui import QColor, QFont

PALETTES = {
    "light": {"BG": "#ffffff", "INK": "#101114", "MUTED": "#5f6672", "CARD": "#f3f4f6", "CARD_2": "#e9ebef",
              "ROW": "#ffffff", "ROW_HOVER": "#f7f9ff", "BLUE": "#0f5cf5", "BLUE_HOVER": "#0b4fd6",
              "BLUE_SOFT": "#e3ecff", "BLUE_OFF": "#9db8f5", "ON_BLUE": "#ffffff", "ERR": "#a3242a",
              "DISABLED": "#9aa0aa",
              "CHIPS": {"blue": ("#e3ecff", "#0b45b8"), "red": ("#fde7e7", "#a3242a"),
                        "amber": ("#fff1dc", "#9a5205"), "green": ("#e1f5e8", "#15803d"),
                        "grey": ("#e9ebef", "#5f6672")}},
    "dark": {"BG": "#0e1116", "INK": "#e8eaee", "MUTED": "#9aa3b0", "CARD": "#181c23", "CARD_2": "#262b35",
             "ROW": "#1f242d", "ROW_HOVER": "#252c38", "BLUE": "#3d7bff", "BLUE_HOVER": "#5a8fff",
             "BLUE_SOFT": "#1c2a4a", "BLUE_OFF": "#2a3b63", "ON_BLUE": "#ffffff", "ERR": "#ff8a92",
             "DISABLED": "#5d6573",
             "CHIPS": {"blue": ("#1c2a4a", "#9cc0ff"), "red": ("#3a1c1f", "#ff9aa2"),
                       "amber": ("#3a2a14", "#ffc46b"), "green": ("#15301f", "#7ee2a3"),
                       "grey": ("#2a2f38", "#b4bcc8")}},
}
MODE = "light"
globals().update(PALETTES[MODE])
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


def set_mode(mode):
    """Switch the colours used by the style sheet, chips and avatars. Returns the new style sheet."""
    global MODE
    MODE = mode if mode in PALETTES else "light"
    globals().update(PALETTES[MODE])
    return style()


def style():
    P = PALETTES[MODE]
    red_bg, red = P["CHIPS"]["red"]
    return f"""
* {{ font-family: "{FAMILY}", "Segoe UI Variable", "Helvetica Neue", Arial; color: {P['INK']}; }}
QMainWindow, QWidget#page, QWidget#window, QStackedWidget {{ background: {P['BG']}; }}
QDialog, QMessageBox {{ background: {P['BG']}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QFrame[card="true"] {{ background: {P['CARD']}; border-radius: 22px; }}
QFrame[row="true"] {{ background: {P['ROW']}; border-radius: 16px; }}
QFrame[row="true"]:hover {{ background: {P['ROW_HOVER']}; }}
QLabel[muted="true"] {{ color: {P['MUTED']}; }}
QLabel[h1="true"] {{ font-size: 30px; font-weight: 500; }}
QLabel[h2="true"] {{ font-size: 16px; font-weight: 600; }}
QLabel[big="true"] {{ font-size: 36px; font-weight: 500; }}
QLabel[err="true"] {{ color: {P['ERR']}; font-weight: 600; }}
QPushButton {{ border: none; border-radius: 22px; min-height: 44px; padding: 0 18px;
              background: {P['CARD']}; font-weight: 600; }}
QPushButton:hover {{ background: {P['CARD_2']}; }}
QPushButton:disabled {{ color: {P['DISABLED']}; }}
QPushButton[kind="primary"] {{ background: {P['BLUE']}; color: {P['ON_BLUE']}; }}
QPushButton[kind="primary"]:hover {{ background: {P['BLUE_HOVER']}; }}
QPushButton[kind="primary"]:disabled {{ background: {P['BLUE_OFF']}; color: {P['ON_BLUE']}; }}
QPushButton[kind="ghost"] {{ background: {P['BG']}; border: 1.5px solid {P['CARD_2']}; }}
QPushButton[kind="ghost"]:hover {{ background: {P['CARD_2']}; }}
QPushButton[kind="danger"] {{ background: {P['BG']}; border: 1.5px solid {red_bg}; color: {red}; }}
QPushButton[kind="danger"]:hover {{ background: {red_bg}; }}
QPushButton[kind="danger"]:disabled {{ color: {P['DISABLED']}; }}
QPushButton[kind="filter"] {{ background: {P['ROW']}; min-height: 38px; border-radius: 19px; }}
QPushButton[kind="filter"][on="true"] {{ background: {P['INK']}; color: {P['BG']}; }}
QPushButton[kind="round"] {{ min-width: 44px; max-width: 44px; min-height: 44px; max-height: 44px;
                             padding: 0; border-radius: 22px; }}
QPushButton[kind="round"][on="true"] {{ background: {P['BLUE']}; color: {P['ON_BLUE']}; }}
QPushButton[kind="flag"] {{ min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px;
                            padding: 0; border-radius: 20px; color: {P['MUTED']}; }}
QPushButton[kind="flag"][on="true"] {{ background: {red_bg}; color: {red}; }}
QPushButton[kind="link"] {{ background: transparent; color: {P['BLUE']}; padding: 0; min-height: 28px; }}
QPushButton[kind="link"]:hover {{ text-decoration: underline; }}
QLineEdit, QComboBox, QDateEdit {{ background: {P['ROW']}; border: 1.5px solid {P['CARD_2']}; border-radius: 22px;
                                  min-height: 44px; padding: 0 16px; font-size: 14px; }}
QLineEdit:focus, QComboBox:focus, QDateEdit:focus {{ border: 2px solid {P['BLUE']}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox QAbstractItemView {{ background: {P['ROW']}; selection-background-color: {P['BLUE_SOFT']}; }}
QTextEdit, QPlainTextEdit {{ background: {P['ROW']}; border: 1.5px solid {P['CARD_2']}; border-radius: 16px;
                             padding: 8px; font-size: 14px; }}
QTextEdit:focus, QPlainTextEdit:focus {{ border: 2px solid {P['BLUE']}; }}
QCheckBox, QRadioButton {{ spacing: 8px; }}
QToolTip {{ background: {P['INK']}; color: {P['BG']}; border: none; padding: 6px; }}
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
QScrollBar::handle:vertical {{ background: {P['CARD_2']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: rgba(110,205,255,0.35); border-radius: 5px; min-width: 30px; }}
"""


STYLE = style()
