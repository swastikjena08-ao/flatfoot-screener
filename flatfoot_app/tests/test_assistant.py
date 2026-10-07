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
    json.dumps(r)        # results must be storable as JSON


def test_top_view_pipeline(eng):
    if not eng.ready("bunion"):
        pytest.skip("bunion models not installed in models/bunion")
    img = Image.fromarray((np.random.default_rng(0).random((500, 300)) * 255).astype(np.uint8))
    r = eng.analyse(img, "top", "left")
    assert len(r["feet"]) == 1 and r["feet"][0]["side"] == "left" and len(r["feet"][0]["points"]) == 6
    json.dumps(r)


def test_fracture_on_both_views(eng):
    if not eng.ready("fracture"):
        pytest.skip("fracture.onnx not found")
    img = Image.fromarray((np.random.default_rng(0).random((300, 400)) * 255).astype(np.uint8))
    for view in ("side", "top"):
        fr = eng.analyse(img, view)["fracture"]
        assert 0 <= fr["p"] <= 1 and fr["flag"] == (fr["p"] >= fr["threshold"])
        assert np.asarray(fr["heatmap"]).shape == (16, 16)
        x0, y0, x1, y1 = fr["map_box"]
        assert x0 <= 0 and y0 <= 0 and x1 >= 400 and y1 >= 300 and abs((x1 - x0) - (y1 - y0)) < 1e-6


def test_fracture_never_reads_as_all_clear():
    assert engine.fracture_class({"p": 0.6, "flag": True}) == ("Possible fracture", "red")
    assert engine.fracture_class({"p": 0.05, "flag": False}) == ("No fracture flagged", "grey")
    res = {"fracture": {"p": 0.6, "flag": True, "threshold": 0.15}}
    assert ("Possible fracture", "red") in engine.findings(res, {})
    assert engine.findings({"fracture": {"p": 0.05, "flag": False, "threshold": 0.15}}, {}) == []


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
                    "crop_box": [240, 140, 390, 290], "heatmap": np.full((7, 7), 0.3).tolist()},
           "fracture": {"p": 0.62, "flag": True, "threshold": 0.15, "heatmap": np.full((16, 16), 0.2).tolist(),
                        "map_box": [0.0, -50.0, 400.0, 350.0]}}
    w.store.save_results(x["id"], res, "right")
    w.open_patient(p["id"])
    assert w.stack.currentWidget() is w.patient
    w.patient.points_edited("arch", 0, {"L10": [100.0, 200.0], "L11": [200.0, 180.0]})
    ed = json.loads(w.store.xray(x["id"])["edits"])
    assert ed["arch"]["L11"] == [200.0, 180.0]
    text = report.draft(w.patient.patient, w.store.xrays(p["id"]), u)
    assert "calcaneal pitch -11.3°" in text and "points adjusted by hand" in text and "Heel spur likely" in text
    assert "Fracture: Possible fracture. Model score 62%" in text and "possible fracture on the side view" in text
    assert w.patient.drawer.heat_buttons["break"].isVisibleTo(w.patient.drawer)
    w.patient.reset_points()
    assert json.loads(w.store.xray(x["id"])["edits"]) == {}
    w.go_home()
    assert w.home.rows.count() >= 1
