"""Main window: sign-in, home and patient pages, the model threads and the idle sign-out."""
import traceback

from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from . import engine, theme
from .dialogs import PatientDialog, SettingsDialog
from .pages import ActivatePage, HomePage, LoginPage
from .patient import PatientPage
from .store import Store

IDLE_MINUTES = 20


class EngineLoader(QThread):
    loaded = Signal(object)

    def __init__(self, bunion_dir=None):
        super().__init__()
        self.bunion_dir = bunion_dir

    def run(self):
        self.loaded.emit(engine.Engine(self.bunion_dir))


class AnalysisWorker(QThread):
    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, eng, xray_id, path, view, side):
        super().__init__()
        self.eng, self.xray_id, self.path, self.view, self.side = eng, xray_id, path, view, side

    def run(self):
        try:
            img = engine.open_xray(self.path)
            self.done.emit(self.xray_id, self.eng.analyse(img, self.view, self.side))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(self.xray_id, str(e))


class IdleWatch(QObject):
    """Restarts the sign-out timer on any key press or mouse use in the app."""
    EVENTS = {QEvent.KeyPress, QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.Wheel}

    def __init__(self, timer):
        super().__init__()
        self.timer = timer

    def eventFilter(self, obj, e):
        if e.type() in self.EVENTS and self.timer.isActive():
            self.timer.start()
        return False


class MainWindow(QMainWindow):
    def __init__(self, data_dir=None, key_file=None, bunion_dir=None, load_models=True):
        super().__init__()
        self.setWindowTitle("Foot X-ray")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
        self.store = Store(data_dir, key_file or engine.resource("assistant", "activation_keys.json"))
        try:
            self.scores = engine.load_scores()
        except (OSError, ValueError):
            self.scores = {}
        self.engine = None
        self.user = None
        self.apply_theme(self.store.prefs().get("theme", "light"))
        self.queue, self.worker = [], None

        self.stack = QStackedWidget()
        self.stack.setObjectName("window")
        self.setCentralWidget(self.stack)
        self.login = LoginPage(self.store)
        self.activate = ActivatePage(self.store)
        self.home = HomePage(self.store)
        self.patient = PatientPage(self.store, self.scores)
        for w in (self.login, self.activate, self.home, self.patient):
            w.setObjectName("page")
            self.stack.addWidget(w)
        self.login.signed_in.connect(self.enter)
        self.login.want_activate.connect(self.show_activate)
        self.activate.activated.connect(self.enter)
        self.activate.back.connect(lambda: self.show_login())
        self.home.open_patient.connect(self.open_patient)
        self.home.new_patient.connect(self.new_patient)
        self.home.settings.connect(self.open_settings)
        self.home.sign_out.connect(lambda: self.sign_out())
        self.home.toggle_theme.connect(self.toggle_theme)
        self.patient.back.connect(self.go_home)
        self.patient.analyse.connect(self.enqueue)
        self.patient.changed.connect(self.home.refresh)

        self.idle = QTimer(self)
        self.idle.setInterval(IDLE_MINUTES * 60 * 1000)
        self.idle.setSingleShot(True)
        self.idle.timeout.connect(lambda: self.sign_out(f"Signed out after {IDLE_MINUTES} minutes without use."))
        self.watch = IdleWatch(self.idle)
        QApplication.instance().installEventFilter(self.watch)

        self.home.show_theme(theme.MODE)
        if self.store.has_users():
            self.show_login()
        else:
            self.show_activate()
        if load_models:
            self.loader = EngineLoader(bunion_dir)
            self.loader.loaded.connect(self.on_engine)
            self.loader.start()

    # ---------- pages ----------

    def show_login(self, message=""):
        self.login.reset(message)
        self.stack.setCurrentWidget(self.login)

    def show_activate(self):
        self.activate.reset()
        self.stack.setCurrentWidget(self.activate)

    def enter(self, user):
        self.user = user
        self.home.show_user(user)
        self.stack.setCurrentWidget(self.home)
        self.idle.start()

    def go_home(self):
        self.home.refresh()
        self.stack.setCurrentWidget(self.home)

    def open_patient(self, pid):
        self.patient.open(self.user, pid, self.engine)
        self.stack.setCurrentWidget(self.patient)

    def new_patient(self):
        d = PatientDialog(self, self.store, self.user)
        if d.exec() and d.result_row:
            self.home.refresh()
            self.open_patient(d.result_row["id"])

    def open_settings(self):
        d = SettingsDialog(self, self.store, self.user, self.engine, self.scores)
        d.toggle_theme.connect(self.toggle_theme)
        d.exec()

    # ---------- light / dark ----------

    def apply_theme(self, mode):
        app = QApplication.instance()
        app.setStyleSheet(theme.set_mode(mode))
        if hasattr(self, "home"):
            self.home.show_theme(theme.MODE)
            if self.user is not None:
                self.home.refresh()
                if self.patient.patient is not None:
                    self.patient.show_header()
                    self.patient.show_cards()

    def toggle_theme(self):
        mode = "light" if theme.MODE == "dark" else "dark"
        self.store.set_pref("theme", mode)
        self.apply_theme(mode)

    def sign_out(self, message=""):
        self.idle.stop()
        for w in QApplication.topLevelWidgets():
            if w is not self and w.isVisible() and w.isWindow() and hasattr(w, "reject"):
                w.reject()
        self.user = None
        self.patient.user = None
        self.show_login(message)

    # ---------- models ----------

    def on_engine(self, eng):
        self.engine = eng
        self.patient.engine = eng
        if eng.flat_lm:
            self.patient.threshold = eng.lm_meta.get("threshold_deg", 17.7)
        # X-rays added before the models were ready, or left unchecked when the app closed
        for x in self.store.db.execute("SELECT id FROM xrays WHERE results IS NULL").fetchall():
            self.enqueue(x["id"])

    def enqueue(self, xray_id):
        if xray_id not in self.queue:
            self.queue.append(xray_id)
            self.patient.mark_busy(xray_id)
        self.next_job()

    def next_job(self):
        if self.worker is not None or self.engine is None or not self.queue:
            return
        xid = self.queue.pop(0)
        x = self.store.xray(xid)
        if x is None:
            return self.next_job()
        self.worker = AnalysisWorker(self.engine, xid, self.store.xray_path(x), x["view"], x["side"] or None)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_done(self, xid, results):
        side = results.get("side") or None
        if side == "both":
            side = ""
        self.store.save_results(xid, results, side)
        self.patient.analysis_done(xid)

    def on_failed(self, xid, msg):
        self.patient.analysis_failed(xid, msg)

    def on_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.next_job()

    def closeEvent(self, e):
        for t in (getattr(self, "loader", None), self.worker):
            if t is not None and t.isRunning():
                t.wait(5000)
        self.store.close()
        super().closeEvent(e)


def run(argv=None, data_dir=None, key_file=None, bunion_dir=None):
    app = QApplication.instance() or QApplication(argv or [])
    app.setFont(theme.font(10))
    w = MainWindow(data_dir, key_file, bunion_dir)
    w.show()
    return app.exec()
