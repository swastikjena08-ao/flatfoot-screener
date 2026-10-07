"""Sign-in, account set-up with an activation key, and the home page with the patient list."""
import datetime as dt

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from . import theme
from .store import StoreError
from .viewer import XrayCanvas
from .widgets import button, card, chip, clear_layout, field, label, line_edit, set_on


def age_of(birth_date, today=None):
    try:
        b = dt.date.fromisoformat(birth_date)
    except (TypeError, ValueError):
        return None
    today = today or dt.date.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


def when(ts):
    if not ts:
        return ""
    d = dt.datetime.fromtimestamp(ts)
    days = (dt.date.today() - d.date()).days
    if days == 0:
        return "Today " + d.strftime("%H:%M")
    if days == 1:
        return "Yesterday"
    return d.strftime("%d %b %Y")


def patient_facts(p):
    bits = []
    a = age_of(p["birth_date"])
    if a is not None:
        bits.append(f"{a} y")
    if p["sex"]:
        bits.append(p["sex"])
    return " · ".join(bits)


class Hero(QFrame):
    """The black lightbox panel next to the sign-in form: an empty viewer grid with the three checks."""

    def __init__(self):
        super().__init__()
        self.setObjectName("drawer")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        top = QLabel("LIGHTBOX")
        top.setStyleSheet("color: #dcf3ff; letter-spacing: 2px; font-weight: 700;")
        lay.addWidget(top)
        grid = XrayCanvas()
        grid.setMinimumSize(200, 200)
        grid.empty_text = "Sign in to open your patients' X-rays"
        lay.addWidget(grid, 1)
        for t in ("Flat foot · heel bone angle", "Heel spur · zoomed heel check", "Bunion · big toe and metatarsal angles"):
            lay.addWidget(QLabel("▸  " + t))


def split_page(form):
    page = QWidget()
    page.setObjectName("page")
    lay = QHBoxLayout(page)
    lay.setContentsMargins(28, 28, 28, 28)
    lay.setSpacing(28)
    left = QWidget()
    ll = QVBoxLayout(left)
    ll.setContentsMargins(40, 0, 40, 0)
    ll.addStretch(1)
    ll.addWidget(form)
    ll.addStretch(1)
    left.setMaximumWidth(560)
    lay.addWidget(left, 5)
    lay.addWidget(Hero(), 6)
    return page


class LoginPage(QWidget):
    signed_in = Signal(object)
    want_activate = Signal()

    def __init__(self, store):
        super().__init__()
        self.store = store
        form = QWidget()
        f = QVBoxLayout(form)
        f.setSpacing(14)
        f.addWidget(label("Welcome back", h1=True))
        f.addWidget(label("Sign in with your doctor or assistant ID.", muted=True))
        f.addSpacing(8)
        self.login = line_edit("e.g. dr.mehta")
        self.password = line_edit("Password", password=True)
        f.addWidget(field("ID", self.login))
        f.addWidget(field("Password", self.password))
        self.err = label("", err=True, wrap=True)
        self.err.hide()
        f.addWidget(self.err)
        self.go = button("Sign in", "primary", self.submit)
        f.addWidget(self.go)
        f.addSpacing(6)
        f.addWidget(button("New doctor? Set up an account with an activation key", "link", self.want_activate.emit))
        self.login.returnPressed.connect(self.password.setFocus)
        self.password.returnPressed.connect(self.submit)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split_page(form))

    def reset(self, message=""):
        self.password.clear()
        self.err.setText(message)
        self.err.setVisible(bool(message))
        (self.password if self.login.text() else self.login).setFocus()

    def submit(self):
        try:
            user = self.store.sign_in(self.login.text(), self.password.text())
        except StoreError as e:
            self.err.setText(str(e))
            self.err.show()
            self.password.clear()
            self.password.setFocus()
            return
        self.err.hide()
        self.password.clear()
        self.signed_in.emit(user)


class ActivatePage(QWidget):
    activated = Signal(object)
    back = Signal()

    def __init__(self, store):
        super().__init__()
        self.store = store
        form = QWidget()
        f = QVBoxLayout(form)
        f.setSpacing(12)
        f.addWidget(label("Set up your account", h1=True))
        f.addWidget(label("Each activation key creates one doctor account on this PC and then stops working.",
                          muted=True, wrap=True))
        self.key = line_edit("FXA-XXXX-XXXX-XXXX-XXXX")
        self.name = line_edit("Dr. Asha Mehta")
        self.login = line_edit("dr.mehta (no spaces)")
        self.pw = line_edit("At least 8 characters, letters and numbers", password=True)
        self.pw2 = line_edit("Type it again", password=True)
        f.addWidget(field("Activation key", self.key))
        f.addWidget(field("Your name", self.name))
        f.addWidget(field("Doctor ID you'll sign in with", self.login))
        f.addWidget(field("Password", self.pw))
        f.addWidget(field("Confirm password", self.pw2))
        self.err = label("", err=True, wrap=True)
        self.err.hide()
        f.addWidget(self.err)
        f.addWidget(button("Create account", "primary", self.submit))
        self.back_link = button("Back to sign in", "link", self.back.emit)
        f.addWidget(self.back_link)
        self.pw2.returnPressed.connect(self.submit)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split_page(form))

    def reset(self):
        for e in (self.key, self.name, self.login, self.pw, self.pw2):
            e.clear()
        self.err.hide()
        self.back_link.setVisible(self.store.has_users())
        self.key.setFocus()

    def submit(self):
        if self.pw.text() != self.pw2.text():
            self.err.setText("The two passwords don't match.")
            self.err.show()
            return
        try:
            user = self.store.activate(self.key.text(), self.login.text(), self.name.text(), self.pw.text())
        except StoreError as e:
            self.err.setText(str(e))
            self.err.show()
            return
        self.reset()
        self.activated.emit(user)


class Stat(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setProperty("card", "true")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.addWidget(label(title, muted=True))
        self.value = label("0", big=True)
        lay.addWidget(self.value)


class PatientRow(QFrame):
    opened = Signal(int)
    flag_toggled = Signal(int, bool)

    def __init__(self, p):
        super().__init__()
        self.pid = p["id"]
        self.setProperty("row", "true")
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 12, 10)
        lay.setSpacing(14)
        initials = "".join(w[0] for w in p["name"].split()[:2]).upper() or "?"
        av = QLabel(initials)
        av.setFixedSize(40, 40)
        av.setAlignment(Qt.AlignCenter)
        av.setStyleSheet(f"background: {theme.BLUE_SOFT}; color: {theme.BLUE}; border-radius: 20px; font-weight: 700;")
        lay.addWidget(av)
        who = QVBoxLayout()
        who.setSpacing(0)
        who.addWidget(label(p["name"], h2=True))
        facts = " · ".join(x for x in (p["code"], patient_facts(p)) if x)
        who.addWidget(label(facts, muted=True))
        lay.addLayout(who, 3)
        last = label(("Last X-ray " + when(p["last_xray"])) if p["last_xray"] else "No X-rays yet", muted=True)
        lay.addWidget(last, 2)
        state = QHBoxLayout()
        state.setSpacing(6)
        todo = (p["n_new"] or 0) + (p["n_review"] or 0)
        if todo:
            state.addWidget(chip(f"{todo} to review", "blue"))
        elif p["n_xrays"]:
            state.addWidget(chip("Signed off", "green"))
        state.addStretch(1)
        lay.addLayout(state, 2)
        self.flag = button("⚑", "flag", tip="Flag this patient")
        set_on(self.flag, bool(p["flagged"]))
        self.flag.clicked.connect(lambda: self.flag_toggled.emit(self.pid, self.flag.property("on") != "true"))
        lay.addWidget(self.flag)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.opened.emit(self.pid)


class HomePage(QWidget):
    open_patient = Signal(int)
    new_patient = Signal()
    settings = Signal()
    sign_out = Signal()
    toggle_theme = Signal()

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.user = None
        self.filter = "all"
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(18)

        top = QHBoxLayout()
        hello = QVBoxLayout()
        hello.setSpacing(2)
        self.hello = label("Welcome back", h1=True)
        self.today = label("", muted=True)
        hello.addWidget(self.hello)
        hello.addWidget(self.today)
        top.addLayout(hello, 1)
        self.theme_btn = button("Dark mode", "ghost", self.toggle_theme.emit)
        top.addWidget(self.theme_btn)
        top.addWidget(button("Settings", "ghost", self.settings.emit))
        top.addWidget(button("Sign out", "ghost", self.sign_out.emit))
        outer.addLayout(top)

        stats = QHBoxLayout()
        stats.setSpacing(14)
        self.stats = {k: Stat(t) for k, t in (("patients", "Patients"), ("flagged", "Flagged"),
                                               ("to_review", "X-rays to review"), ("checks", "X-rays checked"))}
        for s in self.stats.values():
            stats.addWidget(s)
        outer.addLayout(stats)

        box, lay = card(margins=(18, 16, 18, 16), spacing=12)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(label("Patients", h2=True))
        bar.addSpacing(10)
        self.filters = {}
        for key, text in (("all", "All"), ("flagged", "Flagged"), ("review", "To review")):
            b = button(text, "filter", lambda _=False, k=key: self.set_filter(k))
            self.filters[key] = b
            bar.addWidget(b)
        bar.addStretch(1)
        self.search = line_edit("Search name, Patient ID or phone")
        self.search.setMinimumWidth(300)
        self.search.textChanged.connect(self.refresh)
        bar.addWidget(self.search)
        bar.addWidget(button("+  New patient", "primary", self.new_patient.emit))
        lay.addLayout(bar)
        self.list_box = QWidget()
        self.rows = QVBoxLayout(self.list_box)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidget(self.list_box)
        scroll.setWidgetResizable(True)
        lay.addWidget(scroll, 1)
        outer.addWidget(box, 1)
        self.set_filter("all")

    def show_theme(self, mode):
        self.theme_btn.setText("Light mode" if mode == "dark" else "Dark mode")

    def show_user(self, user):
        self.user = user
        title = user["name"] if user["role"] == "assistant" or user["name"].lower().startswith("dr") else "Dr. " + user["name"]
        self.hello.setText(f"Welcome back, {title}")
        role = "Doctor" if user["role"] == "doctor" else "Assistant"
        self.today.setText(f"{dt.date.today():%A %d %B %Y}  ·  Signed in as {user['login']} ({role})")
        self.search.clear()
        self.refresh()

    def set_filter(self, key):
        self.filter = key
        for k, b in self.filters.items():
            set_on(b, k == key)
        self.refresh()

    def refresh(self):
        for k, v in self.store.counts().items():
            self.stats[k].value.setText(str(v))
        clear_layout(self.rows)
        rows = self.store.patients(self.search.text(), only_flagged=self.filter == "flagged",
                                   status="review" if self.filter == "review" else None)
        for p in rows:
            r = PatientRow(p)
            r.opened.connect(self.open_patient)
            r.flag_toggled.connect(self.toggle_flag)
            self.rows.addWidget(r)
        if not rows:
            msg = ("No patients yet. Add the first one with New patient." if not self.store.counts()["patients"]
                   else "No patients match.")
            e = label(msg, muted=True)
            e.setAlignment(Qt.AlignCenter)
            e.setMinimumHeight(120)
            self.rows.addWidget(e)
        self.rows.addStretch(1)

    def toggle_flag(self, pid, on):
        self.store.update_patient(self.user["id"], pid, flagged=int(on))
        self.refresh()
