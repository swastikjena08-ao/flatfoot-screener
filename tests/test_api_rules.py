"""Who can change what: signed-off visits, assistant accounts, new doctor accounts, voice and the dev server.

No model weights needed: X-rays are added straight through the store, and every refused call is
refused before any model would run.
"""
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
from PIL import Image

from remi_ai import api as api_mod
from remi_ai import server
from remi_ai.main import WEB


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod.Voice, "prepare", lambda self: self.status())  # no speech model download
    a = api_mod.Api(data_dir=str(tmp_path / "data"), open_files=False)
    assert "error" not in a.create_doctor("DR-1", "Dr One", "secret1")
    assert "error" not in a.add_patient({"patient_id": "RM-0001", "name": "Pat One"})
    yield a
    a._store.db.close()


@pytest.fixture
def xray_file(tmp_path):
    path = tmp_path / "side.png"
    Image.new("L", (64, 48), 120).save(path)
    return str(path)


def add_xray(app, xray_file):
    vid = app._store.todays_visit("RM-0001", app._doctor["doctor_id"])
    return app._store.add_xray(vid, xray_file, "side", "doctor")


def edits(xid):
    """Every call that changes an X-ray, with arguments that would otherwise be accepted."""
    return {
        "adjust_points": (xid, "bunion", 0, [[0, 0]] * 6),
        "save_marks": (xid, []),
        "set_view": (xid, "side"),
        "mark_reviewed": (xid, True),
        "delete_xray": (xid,),
    }


# ---- signed-off visits are closed ---------------------------------------------------

def test_signed_visit_refuses_every_xray_edit(app, xray_file):
    xid = add_xray(app, xray_file)
    assert "error" not in app.sign_off("RM-0001")
    for name, args in edits(xid).items():
        r = getattr(app, name)(*args)
        assert "signed off by DR-1" in r.get("error", ""), name
    x = app._store.xray(xid)
    assert x is not None and x["reviewed"] == 1 and x["marks"] == []  # nothing changed or deleted


def test_window_is_told_the_xray_is_locked(app, xray_file):
    xid = add_xray(app, xray_file)
    assert app.get_xray(xid)["locked"] is False
    app.sign_off("RM-0001")
    x = app.get_xray(xid)
    assert (x["locked"], x["signed_by"]) == (True, "DR-1")


def test_unsigned_visit_still_editable_and_new_xray_after_sign_off_opens_new_visit(app, xray_file):
    xid = add_xray(app, xray_file)
    assert app.save_marks(xid, [])["ok"] and app.mark_reviewed(xid, True)["ok"]
    app.sign_off("RM-0001")
    new = add_xray(app, xray_file)
    assert app._store.xray(new)["visit_id"] != app._store.xray(xid)["visit_id"]
    assert app.mark_reviewed(new, True)["ok"] and app.delete_xray(new)["ok"]


# ---- assistants ------------------------------------------------------------------------

def test_assistant_cannot_review_or_delete_but_can_prepare(app, xray_file):
    xid = add_xray(app, xray_file)
    app.add_assistant("AS-1", "Asha", "secret2")
    app.logout()
    app.login("AS-1", "secret2")
    for name in ("mark_reviewed", "delete_xray"):
        assert "Only a doctor" in getattr(app, name)(*edits(xid)[name]).get("error", ""), name
    assert "Only a doctor" in app.sign_off("RM-0001").get("error", "")
    assert app._store.xray(xid)["reviewed"] == 0
    assert app.save_marks(xid, [])["ok"]                                  # still allowed
    assert "error" not in app.add_patient({"patient_id": "RM-0002", "name": "Pat Two"})
    assert "path" in app.export_report("RM-0001")                         # draft PDF


# ---- new doctor accounts --------------------------------------------------------------

def test_only_a_signed_in_doctor_can_add_doctors(app):
    app.logout()
    assert "Ask a doctor" in app.create_doctor("DR-X", "Stranger", "secret9").get("error", "")
    app.login("DR-1", "secret1")
    app.add_assistant("AS-1", "Asha", "secret2")
    app.logout()
    app.login("AS-1", "secret2")
    assert "Only a doctor" in app.create_doctor("DR-X", "Stranger", "secret9").get("error", "")
    app.logout()
    app.login("DR-1", "secret1")
    d = app.create_doctor("DR-2", "Dr Two", "secret3")
    assert d["doctor_id"] == "DR-2" and d["role"] == "doctor"
    assert app._doctor["doctor_id"] == "DR-1"                             # still signed in as themselves
    app.logout()
    assert app.login("DR-2", "secret3")["doctor_id"] == "DR-2"


def test_first_doctor_on_a_new_computer_is_signed_in(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod.Voice, "prepare", lambda self: self.status())
    a = api_mod.Api(data_dir=str(tmp_path / "fresh"), open_files=False)
    assert a.app_info()["has_doctors"] is False
    assert a.create_doctor("DR-9", "Dr Nine", "secret1")["doctor_id"] == "DR-9"
    assert a._doctor["doctor_id"] == "DR-9"
    a._store.db.close()


# ---- voice doesn't wait behind an X-ray being analysed --------------------------------

def test_voice_poll_answers_while_the_app_is_busy(app):
    busy, done = threading.Event(), threading.Event()

    def hold_lock():
        with app._lock:
            busy.set()
            done.wait(5)

    threading.Thread(target=hold_lock, daemon=True).start()
    busy.wait(5)
    try:
        result = {}
        t = threading.Thread(target=lambda: result.update(app.voice_poll()), daemon=True)
        t.start()
        t.join(2)
        assert "listening" in result, "voice_poll waited for the app lock"
    finally:
        done.set()


# ---- browser dev server ---------------------------------------------------------------

@pytest.fixture
def dev_server(app, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(server, "ThreadingHTTPServer", _Recording(server.ThreadingHTTPServer))
    threading.Thread(target=server.serve, args=(app, WEB, port, False), daemon=True).start()
    for _ in range(50):
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
            break
        except OSError:
            time.sleep(0.1)
    token = page.split("window.REMI_TOKEN = ", 1)[1].split("<", 1)[0]
    yield port, json.loads(token)
    _Recording.last.shutdown()
    _Recording.last.server_close()


class _Recording:
    """Wraps the server class so the test can shut the server down afterwards."""
    last = None

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *a, **k):
        _Recording.last = self.cls(*a, **k)
        return _Recording.last


def post(port, name, args, headers=None, host=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/{name}", json.dumps(args).encode(), method="POST",
                                 headers={**(headers or {}), **({"Host": host} if host else {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def test_dev_server_needs_the_page_token(dev_server):
    port, token = dev_server
    assert len(token) >= 32
    assert post(port, "account", [])[0] == 403
    assert post(port, "account", [], {"X-Remi-Token": "wrong"})[0] == 403
    assert post(port, "account", [], {"X-Remi-Token": "wrongé"})[0] == 403  # non-ASCII header: refused, no crash
    code, body = post(port, "account", [], {"X-Remi-Token": token})
    assert code == 200 and body["me"]["doctor_id"] == "DR-1"


def test_dev_server_refuses_other_host_names(dev_server):
    port, token = dev_server
    assert post(port, "account", [], {"X-Remi-Token": token}, host=f"evil.example:{port}")[0] == 403
    req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403                                            # the page (and its token) isn't served
    assert post(port, "account", [], {"X-Remi-Token": token}, host=f"localhost:{port}")[0] == 200
