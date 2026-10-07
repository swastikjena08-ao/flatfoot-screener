"""Tests for the assistant app: accounts, activation keys, patient storage and the main screens.

Run from the flatfoot_app folder:  python -m pytest tests
"""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PIL import Image

from assistant import auth, engine, report
from assistant.store import Store, StoreError

KEYS = ["FXA-2345-6789-ABCD-EFGH", "FXA-JKLM-NPQR-STUV-WXYZ"]


@pytest.fixture
def store(tmp_path):
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps({"sha256": [auth.key_hash(k) for k in KEYS]}))
    s = Store(str(tmp_path / "data"), str(kf))
    yield s
    s.close()


@pytest.fixture
def doctor(store):
    return store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")


def test_key_normalising():
    assert auth.normalise_key("fxa 2345 6789 abcd efgh") == KEYS[0]
    assert auth.normalise_key("23456789ABCDEFGH") == KEYS[0]
    assert auth.normalise_key("FXA-1234") is None
    k = auth.new_key()
    assert auth.normalise_key(k) == k and not set(k.replace("-", "")[3:]) - set(auth.KEY_ALPHABET)


def test_password_rules():
    assert auth.password_problem("short1")
    assert auth.password_problem("onlyletters")
    assert auth.password_problem("clinic2026") is None
    h, salt, it = auth.hash_password("clinic2026")
    assert auth.check_password("clinic2026", h, salt, it)
    assert not auth.check_password("clinic2027", h, salt, it)


def test_activation_key_works_once(store, doctor):
    assert doctor["role"] == "doctor"
    with pytest.raises(StoreError, match="already been used"):
        store.activate(KEYS[0], "dr.two", "Two", "clinic2026")
    with pytest.raises(StoreError, match="isn't valid"):
        store.activate("FXA-2345-6789-ABCD-EFGJ", "dr.two", "Two", "clinic2026")
    with pytest.raises(StoreError, match="already taken"):
        store.activate(KEYS[1], "DR.MEHTA", "Two", "clinic2026")
    assert store.activate(KEYS[1], "dr.two", "Two", "clinic2026")["login"] == "dr.two"


def test_no_keys_file_means_no_activation(tmp_path):
    s = Store(str(tmp_path / "d"), str(tmp_path / "missing.json"))
    with pytest.raises(StoreError):
        s.activate(KEYS[0], "dr.x", "X", "clinic2026")


def test_sign_in_and_lockout(store, doctor):
    assert store.sign_in("Dr.Mehta", "clinic2026")["id"] == doctor["id"]
    with pytest.raises(StoreError, match="don't match"):
        store.sign_in("nobody", "clinic2026")
    for _ in range(4):
        with pytest.raises(StoreError, match="don't match"):
            store.sign_in("dr.mehta", "wrong", now=1000)
    with pytest.raises(StoreError, match="Too many"):
        store.sign_in("dr.mehta", "wrong", now=1000)
    with pytest.raises(StoreError, match="Too many"):
        store.sign_in("dr.mehta", "clinic2026", now=1030)    # right password, still locked
    assert store.sign_in("dr.mehta", "clinic2026", now=1061)


def test_assistant_accounts(store, doctor):
    a = store.add_assistant(doctor["id"], "nurse.joy", "Joy", "helper2026")
    assert a["role"] == "assistant"
    with pytest.raises(StoreError, match="Only a doctor"):
        store.add_assistant(a["id"], "x", "X", "helper2026")
    with pytest.raises(StoreError, match="Only a doctor"):
        store.sign_off(a["id"], [])


def test_patients_and_xrays(store, doctor, tmp_path):
    p = store.add_patient(doctor["id"], "P-1", "Ravi Kumar", "Male", "1979-05-02", "98450")
    with pytest.raises(StoreError, match="already exists"):
        store.add_patient(doctor["id"], "p-1", "Someone")
    store.add_patient(doctor["id"], "P-2", "Meera Shah")
    store.update_patient(doctor["id"], p["id"], flagged=1)
    assert [r["code"] for r in store.patients(only_flagged=True)] == ["P-1"]
    assert [r["code"] for r in store.patients("meera")] == ["P-2"]
    src = tmp_path / "x.png"
    Image.new("L", (40, 30)).save(src)
    x = store.add_xray(doctor["id"], p["id"], str(src), "side")
    assert os.path.exists(store.xray_path(x)) and x["status"] == "new"
    store.save_results(x["id"], {"view": "side"}, "left")
    assert store.xray(x["id"])["status"] == "review" and store.xray(x["id"])["side"] == "left"
    assert [r["code"] for r in store.patients(status="review")] == ["P-1"]
    store.sign_off(doctor["id"], [x["id"]])
    assert store.xray(x["id"])["status"] == "done"
    assert store.counts() == {"patients": 2, "flagged": 1, "to_review": 0, "checks": 1}
    # data survives a restart
    store.close()
    again = Store(store.folder)
    assert again.patient_by_code("P-1")["name"] == "Ravi Kumar"
    again.close()


def test_classes_and_edits():
    assert engine.arch_class(10, 17.7)[0] == "Flat foot range"
    assert engine.arch_class(25, 17.7)[0] == "Normal arch"
    assert engine.arch_class(31, 17.7)[0] == "High arch flag"
    assert engine.bunion_class(12, 7)[0] == "No bunion"
    assert engine.bunion_class(16, 7)[0] == "Mild bunion"
    assert engine.bunion_class(25, 10)[0] == "Moderate bunion"
    assert engine.bunion_class(45, 10)[0] == "Severe bunion"
    pts = [[100, 0], [100, 100], [100, 100], [100, 300], [150, 120], [160, 300]]
    res = {"feet": [{"side": "left", "points": pts, "hva": 0, "ima": 0}]}
    moved = [[60, 0], *pts[1:]]
    f = engine.bunion_feet(res, {"bunion": {"0": moved}})[0]
    assert f["edited"] and abs(f["hva"] - 21.8) < 0.1
    assert not engine.bunion_feet(res, {})[0]["edited"]


# ---------- models through the app's engine (skipped without the model files) ----------

@pytest.fixture(scope="module")
def eng():
    return engine.Engine()


def test_side_view_pipeline(eng):
    if not (eng.ready("flat") and eng.ready("spur")):
        pytest.skip("flat foot / heel spur models not found")
    img = Image.fromarray((np.random.default_rng(0).random((300, 400)) * 255).astype(np.uint8))
    r = eng.analyse(img, "side")
    assert 0 <= r["flat"]["p"] <= 1 and set(r["flat"]["points"]) == {"L10", "L11"}
    assert abs(sum(r["spur"]["probs"].values()) - 1) < 1e-4
    if eng.ready("fracture"):
        assert 0 <= r["fracture"]["p"] <= 1 and len(r["fracture"]["cells"]) == len(r["fracture"]["cells"][0])
    json.dumps(r)        # results must be storable as JSON


def test_top_view_pipeline(eng):
    if not eng.ready("bunion"):
        pytest.skip("bunion models not installed in models/bunion")
    img = Image.fromarray((np.random.default_rng(0).random((500, 300)) * 255).astype(np.uint8))
    r = eng.analyse(img, "top", "left")
    assert len(r["feet"]) == 1 and r["feet"][0]["side"] == "left" and len(r["feet"][0]["points"]) == 6
    json.dumps(r)


# ---------- screens ----------

@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from assistant import theme
    from assistant.app import MainWindow
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.STYLE)
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps({"sha256": [auth.key_hash(k) for k in KEYS]}))
    w = MainWindow(str(tmp_path / "data"), str(kf), load_models=False)
    yield w
    w.close()


def test_first_run_needs_a_key_then_signs_in(window):
    w = window
    assert w.stack.currentWidget() is w.activate
    a = w.activate
    a.key.setText("not a key")
    a.name.setText("Asha Mehta")
    a.login.setText("dr.mehta")
    a.pw.setText("clinic2026")
    a.pw2.setText("clinic2026")
    a.submit()
    assert w.stack.currentWidget() is w.activate and a.err.text()
    a.key.setText(KEYS[0])
    a.pw.setText("clinic2026")
    a.pw2.setText("clinic2026")
    a.submit()
    assert w.stack.currentWidget() is w.home
    w.sign_out()
    assert w.stack.currentWidget() is w.login and w.user is None
    w.login.login.setText("dr.mehta")
    w.login.password.setText("nope")
    w.login.submit()
    assert w.stack.currentWidget() is w.login
    w.login.password.setText("clinic2026")
    w.login.submit()
    assert w.stack.currentWidget() is w.home


def test_patient_page_and_report(window, tmp_path):
    w = window
    u = w.store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")
    w.enter(u)
    p = w.store.add_patient(u["id"], "P-1", "Ravi Kumar", "Male", "1979-05-02")
    src = tmp_path / "x.png"
    Image.new("L", (400, 300), 80).save(src)
    x = w.store.add_xray(u["id"], p["id"], str(src), "side")
    res = {"view": "side", "side": "right",
           "flat": {"p": 0.9, "cam": np.full((7, 7), 0.5).tolist(), "left": False, "how": "detected automatically",
                    "points": {"L10": [100.0, 200.0], "L11": [200.0, 220.0]}},
           "spur": {"probs": {"normal": 0.1, "spur": 0.85, "sever": 0.05}, "label": "spur", "box": [250, 150, 380, 290],
                    "crop_box": [240, 140, 390, 290], "heatmap": np.full((7, 7), 0.3).tolist()}}
    w.store.save_results(x["id"], res, "right")
    w.open_patient(p["id"])
    assert w.stack.currentWidget() is w.patient
    w.patient.points_edited("arch", 0, {"L10": [100.0, 200.0], "L11": [200.0, 180.0]})
    ed = json.loads(w.store.xray(x["id"])["edits"])
    assert ed["arch"]["L11"] == [200.0, 180.0]
    text = report.draft(w.patient.patient, w.store.xrays(p["id"]), u)
    assert "calcaneal pitch -11.3°" in text and "points adjusted by hand" in text and "Heel spur likely" in text
    w.patient.reset_points()
    assert json.loads(w.store.xray(x["id"])["edits"]) == {}
    w.go_home()
    assert w.home.rows.count() >= 1


def test_fracture_card_heatmap_and_report(window, tmp_path):
    w = window
    u = w.store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")
    w.enter(u)
    p = w.store.add_patient(u["id"], "P-2", "Mira Shah", "Female", "1990-01-01")
    src = tmp_path / "t.png"
    Image.new("L", (300, 400), 80).save(src)
    x = w.store.add_xray(u["id"], p["id"], str(src), "top")
    cells = np.zeros((16, 16))
    cells[4, 8] = 0.6
    # 300 x 400 X-ray letterboxed into a 400 px square: 50 px of padding on each side
    res = {"view": "top", "feet": [], "fracture": {"p": 0.4, "flag": True, "threshold": 0.15,
                                                   "cells": cells.tolist(), "square": [-50, 0, 350, 400]}}
    w.store.save_results(x["id"], res, "")
    w.open_patient(p["id"])
    heat = engine.fracture_heat(res["fracture"], 300, 400)
    assert heat.shape == (400, 300) and heat.max() == pytest.approx(1, abs=0.05)
    y, xx = np.unravel_index(heat.argmax(), heat.shape)
    assert abs(y - 4.5 * 25) < 15 and abs(xx - (8.5 * 25 - 50)) < 15
    assert "break" in w.patient.drawer.canvas.heats
    assert ("Possible fracture", "red") in engine.findings(res, {})
    text = report.draft(w.patient.patient, w.store.xrays(p["id"]), u)
    assert "Fracture flag (demo model" in text and "possible fracture flagged" in text
    assert engine.fracture_class({**res["fracture"], "flag": False})[0] == "No fracture flagged"


def test_named_foot_still_uses_foot_finder(eng, monkeypatch):
    if not eng.ready("bunion"):
        pytest.skip("bunion models not installed in models/bunion")
    seen = {}
    monkeypatch.setattr(eng.bunion, "finder", object())
    monkeypatch.setattr(eng.bunion, "find_feet", lambda g: {"left": (10.0, 20.0, 150.0, 480.0)})
    monkeypatch.setattr(eng.bunion, "predict", lambda g, boxes: seen.update(boxes) or [])
    img = Image.fromarray(np.zeros((500, 300), np.uint8))
    eng.analyse(img, "top", "left")
    assert seen == {"left": (10.0, 20.0, 150.0, 480.0)}
    seen.clear()
    eng.analyse(img, "top", "right")      # only one foot found: measure it as the foot named
    assert seen == {"right": (10.0, 20.0, 150.0, 480.0)}


def test_delete_xray_removes_file_and_marked_copy(store, doctor, tmp_path):
    p = store.add_patient(doctor["id"], "P-9", "Del Test")
    src = tmp_path / "x.png"
    Image.new("L", (40, 30)).save(src)
    x = store.add_xray(doctor["id"], p["id"], str(src), "top", "left")
    marked = store.marked_path(x)
    Image.new("RGB", (10, 10)).save(marked)
    store.set_marked(doctor["id"], x["id"], marked)
    path = store.xray_path(x)
    store.delete_xray(doctor["id"], x["id"])
    assert store.xray(x["id"]) is None and not os.path.exists(path) and not os.path.exists(marked)
    assert store.xrays(p["id"]) == []


def test_prefs_survive_restart(store):
    store.set_pref("theme", "dark")
    again = Store(store.folder)
    assert again.prefs()["theme"] == "dark"
    again.close()


def _side_results():
    return {"view": "side", "side": "right",
            "flat": {"p": 0.9, "cam": np.full((7, 7), 0.5).tolist(), "left": False, "how": "detected automatically",
                     "points": {"L10": [100.0, 200.0], "L11": [200.0, 220.0]}},
            "spur": {"probs": {"normal": 0.1, "spur": 0.85, "sever": 0.05}, "label": "spur", "box": [250, 150, 380, 290],
                     "crop_box": [240, 140, 390, 290], "heatmap": np.full((7, 7), 0.3).tolist()}}


def test_save_marked_delete_and_dark_mode(window, tmp_path):
    w = window
    u = w.store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")
    w.enter(u)
    p = w.store.add_patient(u["id"], "P-1", "Ravi Kumar")
    src = tmp_path / "x.png"
    Image.new("L", (400, 300), 80).save(src)
    x = w.store.add_xray(u["id"], p["id"], str(src), "side")
    w.store.save_results(x["id"], _side_results(), "right")
    w.open_patient(p["id"])
    out = tmp_path / "marked.png"
    assert w.patient.save_marked(str(out)) == str(out)
    img = Image.open(out)
    assert max(img.size) == 1600
    assert os.path.exists(w.store.marked_path(w.store.xray(x["id"])))
    w.toggle_theme()
    assert w.store.prefs()["theme"] == "dark" and w.home.theme_btn.text() == "Light mode"
    w.toggle_theme()
    w.patient.delete_xray(confirm=False)
    assert w.store.xrays(p["id"]) == [] and w.patient.current is None


def test_thumbnail_cross_deletes_that_xray(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w = window
    u = w.store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")
    w.enter(u)
    p = w.store.add_patient(u["id"], "P-1", "Ravi Kumar")
    src = tmp_path / "x.png"
    Image.new("L", (400, 300), 80).save(src)
    a = w.store.add_xray(u["id"], p["id"], str(src), "side")
    b = w.store.add_xray(u["id"], p["id"], str(src), "top", "left")
    w.open_patient(p["id"])
    thumbs = {t.xray_id: t for t in w.patient.drawer._thumb_buttons}
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    thumbs[a["id"]].close_btn.click()
    assert len(w.store.xrays(p["id"])) == 2          # said no
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    thumbs[a["id"]].close_btn.click()
    assert [x["id"] for x in w.store.xrays(p["id"])] == [b["id"]]
    assert [t.xray_id for t in w.patient.drawer._thumb_buttons] == [b["id"]]


def test_results_wait_for_a_full_scan(window, tmp_path):
    from assistant import patient as patient_page
    w = window
    u = w.store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")
    w.enter(u)
    p = w.store.add_patient(u["id"], "P-1", "Ravi Kumar")
    src = tmp_path / "x.png"
    Image.new("L", (400, 300), 80).save(src)
    x = w.store.add_xray(u["id"], p["id"], str(src), "side")
    w.open_patient(p["id"])
    w.patient.mark_busy(x["id"])
    w.store.save_results(x["id"], _side_results(), "right")
    shots = []
    import PySide6.QtCore as qc
    real = qc.QTimer.singleShot
    patient_page.QTimer.singleShot = staticmethod(lambda ms, fn: shots.append(ms))
    try:
        w.patient.analysis_done(x["id"])
    finally:
        patient_page.QTimer.singleShot = real
    assert x["id"] in w.patient.busy and shots and 4000 < shots[0] <= 5000


def test_pen_and_note_marks_are_saved_and_reported(window, tmp_path, monkeypatch):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QInputDialog
    from PySide6.QtCore import QEvent
    w = window
    u = w.store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")
    w.enter(u)
    p = w.store.add_patient(u["id"], "P-1", "Ravi Kumar")
    src = tmp_path / "x.png"
    Image.new("L", (400, 300), 80).save(src)
    x = w.store.add_xray(u["id"], p["id"], str(src), "side")
    w.store.save_results(x["id"], _side_results(), "right")
    w.open_patient(p["id"])
    w.resize(1440, 900)
    c = w.patient.drawer.canvas
    c.resize(800, 600)
    w.patient.drawer.pick_tool("pen")
    r = c.image_rect()

    def mouse(kind, fx, fy):
        pos = QPointF(r.x() + fx * r.width(), r.y() + fy * r.height())
        c.event(QMouseEvent(kind, pos, pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))

    mouse(QEvent.MouseButtonPress, 0.2, 0.2)
    mouse(QEvent.MouseMove, 0.4, 0.3)
    mouse(QEvent.MouseButtonRelease, 0.5, 0.4)
    marks = json.loads(w.store.xray(x["id"])["marks"])
    assert len(marks["strokes"]) == 1 and len(marks["strokes"][0]["pts"]) >= 2
    monkeypatch.setattr(QInputDialog, "exec", lambda self: True)
    monkeypatch.setattr(QInputDialog, "textValue", lambda self: "Swelling over the heel")
    w.patient.note_clicked(-1, 120.0, 90.0)
    marks = json.loads(w.store.xray(x["id"])["marks"])
    assert marks["notes"] == [{"x": 120.0, "y": 90.0, "text": "Swelling over the heel"}]
    w.store.save_results(x["id"], _side_results(), "right")       # a re-check keeps the marks
    assert json.loads(w.store.xray(x["id"])["marks"])["notes"]
    text = report.draft(w.store.patient(p["id"]), w.store.xrays(p["id"]), u)
    assert "1. Swelling over the heel" in text
    out = tmp_path / "m.png"
    w.patient.show_xray(x["id"])
    assert w.patient.save_marked(str(out))
    monkeypatch.setattr(QInputDialog, "textValue", lambda self: "")
    w.patient.note_clicked(0, 0, 0)                                   # empty text deletes the note
    assert json.loads(w.store.xray(x["id"])["marks"])["notes"] == []


def test_note_pins_can_be_dragged_and_still_clicked(window, tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "exec", lambda self: False)      # editor opens, then cancelled
    w = window
    u = w.store.activate(KEYS[0], "dr.mehta", "Asha Mehta", "clinic2026")
    w.enter(u)
    p = w.store.add_patient(u["id"], "P-1", "Ravi Kumar")
    src = tmp_path / "x.png"
    Image.new("L", (400, 300), 80).save(src)
    x = w.store.add_xray(u["id"], p["id"], str(src), "side")
    w.store.save_results(x["id"], _side_results(), "right")
    w.store.save_marks(u["id"], x["id"], {"strokes": [], "notes": [{"x": 100.0, "y": 100.0, "text": "Porous bone"}]})
    w.open_patient(p["id"])
    c = w.patient.drawer.canvas
    c.resize(800, 600)
    clicks = []
    c.note_clicked.connect(lambda i, nx, ny: clicks.append(i))

    def mouse(kind, ix, iy):
        pos = c.to_widget(ix, iy)
        c.event(QMouseEvent(kind, pos, pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))

    for tool in (None, "note"):
        c.set_tool(tool)
        mouse(QEvent.MouseButtonPress, 100, 100)                       # plain click opens the editor
        mouse(QEvent.MouseButtonRelease, 100, 100)
    assert clicks == [0, 0]
    clicks.clear()
    mouse(QEvent.MouseButtonPress, 100, 100)                           # drag moves the pin, no editor
    mouse(QEvent.MouseMove, 200, 150)
    mouse(QEvent.MouseButtonRelease, 250, 200)
    assert clicks == []
    n = json.loads(w.store.xray(x["id"])["marks"])["notes"][0]
    assert n["text"] == "Porous bone" and abs(n["x"] - 200) < 2 and abs(n["y"] - 150) < 2
