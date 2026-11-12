"""Everything the window can ask the app to do.

The web page calls these methods by name (through pywebview's js_api, or the local
development server in server.py). Every method returns plain JSON-safe data; errors come
back as {"error": "..."} so the page can show them.
"""
import base64
import datetime as dt
import functools
import io
import os
import subprocess
import sys
import tempfile
import threading

import shutil

import numpy as np
from PIL import Image

from . import pdf_report, report
from .engine import CHECKS, MODELS, VIEWS, Engine, adjust_points, findings, load_xray
from .store import DEFAULT_HISTORY, Store
from .voice import Voice

DISPLAY_MAX = 1800  # longest side of the copy sent to the window


def safe(fn):
    """Run under the app lock and turn exceptions into {"error": message}."""
    @functools.wraps(fn)
    def wrapper(self, *args):
        with self._lock:
            try:
                return fn(self, *args)
            except (ValueError, PermissionError, FileNotFoundError) as e:
                return {"error": str(e)}
            except Exception as e:  # unexpected: still keep the window alive
                return {"error": f"{type(e).__name__}: {e}"}
    return wrapper


def image_data_url(path):
    img = load_xray(path)
    if img.mode.startswith("I") or img.mode == "F":  # 16-bit / float exports: stretch to 8-bit
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))
    img = img.convert("L")
    w, h = img.size
    img.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), w, h


def open_path(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - opens the file with its default program
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class Api:
    def __init__(self, data_dir=None, models_dir=None, open_files=True):
        self._lock = threading.RLock()
        self._store = Store(data_dir)
        self._engine = Engine(models_dir) if models_dir else Engine()
        self._doctor = None
        self._open_files = open_files
        self._voice = Voice(self._store.dir)
        self._voice.prepare()  # fetch the offline speech model in the background on first run

    # ---- helpers (not exposed: leading underscore) ----------------------------------
    def _need_login(self):
        if self._doctor is None:
            raise PermissionError("Please sign in first.")

    def _need_doctor(self):
        self._need_login()
        if self._doctor.get("role") == "assistant":
            raise PermissionError("Only a doctor can do this. Assistant accounts can prepare reports but not sign off.")

    def _open(self, path):
        if self._open_files:
            try:
                open_path(path)
            except OSError:  # no program to open it with: the file is still saved
                pass

    def _xray(self, xray_id):
        x = self._store.xray(xray_id)
        if x is None:
            raise ValueError("That X-ray no longer exists.")
        return x

    def _editable_xray(self, xray_id):
        """The X-ray, if its visit is still open. A signed-off visit is a closed record: its X-rays,
        results and markings stay exactly as they were signed."""
        x = self._xray(xray_id)
        if self._store.visit(x["visit_id"])["signed_by"]:
            raise ValueError("This visit is signed off, so its X-rays can't be changed. "
                             "Anything new you add starts a new visit.")
        return x

    def _xray_summary(self, x, signed=None):
        if signed is None:
            signed = bool(self._store.visit(x["visit_id"])["signed_by"])
        return {"id": x["id"], "view": x["view"], "view_label": VIEWS.get(x["view"], x["view"]),
                "view_source": x["view_source"], "view_conf": x["view_conf"], "reviewed": bool(x["reviewed"]),
                "added": x["added"], "findings": [list(f) for f in findings(x["results"])],
                "analysed": bool(x["results"]), "signed": signed}

    # ---- start-up and sign-in --------------------------------------------------------
    @safe
    def app_info(self):
        return {"has_doctors": self._store.has_doctors(), "last_doctor_id": self._store.setting("last_doctor_id", ""),
                "models": self._engine.status(), "model_errors": self._engine.errors,
                "checks": CHECKS, "views": VIEWS, "model_info": MODELS, "history_items": DEFAULT_HISTORY,
                "doctor": self._doctor, "data_dir": self._store.dir}

    @safe
    def create_doctor(self, doctor_id, name, password, clinic=""):
        self._doctor = self._store.add_doctor(doctor_id, name, password, clinic)
        self._store.set_setting("last_doctor_id", self._doctor["doctor_id"])
        return self._doctor

    @safe
    def login(self, doctor_id, password):
        d = self._store.check_login(doctor_id, password)
        if d is None:
            raise ValueError("That doctor ID and password don't match.")
        self._doctor = d
        self._store.set_setting("last_doctor_id", d["doctor_id"])
        return d

    @safe
    def logout(self):
        self._doctor = None
        return {"ok": True}

    # ---- account -------------------------------------------------------------------
    @safe
    def account(self):
        self._need_login()
        d = self._store.doctor(self._doctor["doctor_id"])
        out = {"me": d, "assistants": [], "data_dir": self._store.dir}
        if d["role"] == "doctor":
            out["assistants"] = self._store.assistants(d["doctor_id"])
        else:
            out["supervisor"] = self._store.doctor(d["supervisor"]) if d["supervisor"] else None
        return out

    @safe
    def update_account(self, name, clinic=""):
        self._need_login()
        self._store.update_account(self._doctor["doctor_id"], name, clinic)
        self._doctor = self._store.doctor(self._doctor["doctor_id"])
        return self._doctor

    @safe
    def change_password(self, old, new):
        self._need_login()
        if self._store.check_login(self._doctor["doctor_id"], old) is None:
            raise ValueError("Your current password is not right.")
        self._store.set_password(self._doctor["doctor_id"], new)
        return {"ok": True}

    @safe
    def add_assistant(self, assistant_id, name, password):
        self._need_doctor()
        return self._store.add_doctor(assistant_id, name, password, self._doctor.get("clinic", ""),
                                      role="assistant", supervisor=self._doctor["doctor_id"])

    @safe
    def remove_assistant(self, assistant_id):
        self._need_doctor()
        a = self._store.doctor(assistant_id)
        if a is None or a["role"] != "assistant" or a["supervisor"] != self._doctor["doctor_id"]:
            raise ValueError("That isn't one of your assistant accounts.")
        self._store.delete_account(a["doctor_id"])
        return {"ok": True}

    @safe
    def model_status(self):
        """Every AI model: installed, passes a test run, how long it took, and its tested accuracy."""
        return list(self._engine.health().values())

    # ---- voice (offline speech recognition for Ask Remi) ------------------------------
    @safe
    def voice_status(self):
        return self._voice.status()

    @safe
    def voice_prepare(self):
        return self._voice.prepare()

    @safe
    def voice_start(self, mode="once"):
        return self._voice.start("handsfree" if mode == "handsfree" else "once")

    @safe
    def voice_poll(self):
        return self._voice.poll()

    @safe
    def voice_stop(self):
        return self._voice.stop()

    # ---- patients --------------------------------------------------------------------
    @safe
    def list_patients(self, query=""):
        self._need_login()
        out = []
        for p in self._store.patients(query):
            visits = self._store.visits(p["patient_id"])
            latest = []
            for v in visits:
                xs = [x for x in self._store.xrays(v["id"]) if x["results"]]
                if xs:
                    for x in xs:
                        latest += [list(f) for f in findings(x["results"])]
                    break
            p["findings"] = latest
            p["unreviewed"] = self._store.unreviewed_count(p["patient_id"])
            out.append(p)
        return out

    @safe
    def next_patient_id(self):
        n = len(self._store.patients()) + 1
        while self._store.patient(f"RM-{n:04d}"):
            n += 1
        return f"RM-{n:04d}"

    @safe
    def add_patient(self, data):
        self._need_login()
        age = data.get("age")
        p = self._store.add_patient(
            data.get("patient_id", ""), data.get("name", ""), self._doctor["doctor_id"],
            age=int(age) if str(age or "").strip().isdigit() else None, sex=data.get("sex", ""),
            phone=data.get("phone", ""), referred_by=data.get("referred_by", ""),
            complaint=data.get("complaint", ""), history=data.get("history") or {})
        self._store.add_visit(p["patient_id"], self._doctor["doctor_id"], "First visit")
        return p

    @safe
    def update_patient(self, patient_id, data):
        self._need_login()
        fields = {k: data[k] for k in ("name", "sex", "phone", "referred_by", "complaint") if k in data}
        if "age" in data:
            fields["age"] = int(data["age"]) if str(data["age"] or "").strip().isdigit() else None
        self._store.update_patient(patient_id, **fields)
        if "history" in data:
            self._store.set_history(patient_id, data["history"])
        return {"ok": True}

    @safe
    def get_patient(self, patient_id):
        self._need_login()
        p = self._store.patient(patient_id)
        if p is None:
            raise ValueError(f"No patient with ID {patient_id}.")
        visits = []
        for v in self._store.visits(patient_id):
            v["xrays"] = [self._xray_summary(x, bool(v["signed_by"])) for x in self._store.xrays(v["id"])]
            visits.append(v)
        return {"patient": p, "history": self._store.history(patient_id), "visits": visits,
                "today": dt.date.today().isoformat()}

    @safe
    def add_note(self, patient_id, text):
        self._need_login()
        vid = self._store.todays_visit(patient_id, self._doctor["doctor_id"])
        old = self._store.visit(vid)["note"]
        stamp = dt.datetime.now().strftime("%H:%M")
        note = (old + "\n" if old else "") + f"[{stamp}] {text.strip()}"
        self._store.set_visit_note(vid, note)
        return {"ok": True}

    # ---- X-rays ----------------------------------------------------------------------
    @safe
    def add_xray(self, patient_id, filename, b64):
        """Save an uploaded X-ray into the patient's folder, check its type and run the checks that fit."""
        self._need_login()
        if "," in b64[:100]:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        ext = os.path.splitext(filename)[1].lower() or ".png"
        fd, tmp = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            try:
                img = load_xray(tmp)
            except Exception:
                raise ValueError("That file isn't an image Remi can open. Use a JPG, PNG or TIFF X-ray.")
            view, conf = self._engine.detect_view(img)
            vid = self._store.todays_visit(patient_id, self._doctor["doctor_id"])
            if view is None or view == "unsure":
                xid = self._store.add_xray(vid, tmp, "unsure", "x-ray check" if view else "none", conf)
                return {"id": xid, "needs_view": True, "view_conf": conf}
            xid = self._store.add_xray(vid, tmp, view, "x-ray check", conf)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        x = self._store.xray(xid)
        self._store.set_results(xid, self._engine.analyse(x["path"], view))
        return {"id": xid, "needs_view": False, "view": view, "view_conf": conf}

    @safe
    def set_view(self, xray_id, view):
        """The doctor says what kind of X-ray this is; rerun only the checks that fit."""
        self._need_login()
        if view not in VIEWS:
            raise ValueError(f"Unknown view {view}.")
        x = self._editable_xray(xray_id)
        self._store.set_results(xray_id, self._engine.analyse(x["path"], view), view, "doctor")
        self._store.set_reviewed(xray_id, False)
        return {"ok": True}

    @safe
    def get_xray(self, xray_id):
        self._need_login()
        x = self._xray(xray_id)
        url, w, h = image_data_url(x["path"])
        visit = self._store.visit(x["visit_id"])
        out = self._xray_summary(x)
        out.update({"image": url, "width": w, "height": h, "results": x["results"], "date": visit["date"],
                    "marks": x["marks"]})
        prev = self._store.previous_xray(visit["patient_id"], x["visit_id"], x["view"])
        if prev is not None and prev["results"]:
            purl, pw, ph = image_data_url(prev["path"])
            out["previous"] = {"id": prev["id"], "date": self._store.visit(prev["visit_id"])["date"],
                               "image": purl, "width": pw, "height": ph, "results": prev["results"]}
        return out

    @safe
    def adjust_points(self, xray_id, key, index, points):
        """The doctor dragged the measuring points; save them and return the recomputed finding."""
        self._need_login()
        x = self._editable_xray(xray_id)
        results = x["results"]
        finding = adjust_points(results, key, index, points)
        self._store.set_results(xray_id, results)
        return finding

    @safe
    def save_marks(self, xray_id, marks):
        """The doctor's pen strokes and flag labels drawn on this X-ray (in image pixels)."""
        self._need_login()
        self._editable_xray(xray_id)
        clean = []
        for m in marks if isinstance(marks, list) else []:
            if m.get("type") == "pen" and len(m.get("points", [])) >= 2:
                clean.append({"type": "pen", "color": str(m.get("color", "#ff5a5f"))[:16],
                              "points": [[round(float(a), 1), round(float(b), 1)] for a, b in m["points"][:4000]]})
            elif m.get("type") == "flag":
                clean.append({"type": "flag", "color": str(m.get("color", "#ff5a5f"))[:16],
                              "text": str(m.get("text", ""))[:200],
                              "pin": [round(float(v), 1) for v in m["pin"]], "at": [round(float(v), 1) for v in m["at"]]})
        self._store.set_marks(xray_id, clean)
        return {"ok": True, "count": len(clean)}

    @safe
    def mark_reviewed(self, xray_id, reviewed=True):
        self._need_login()
        if reviewed:
            self._xray(xray_id)
        else:
            self._editable_xray(xray_id)  # a signed-off X-ray stays reviewed
        self._store.set_reviewed(xray_id, reviewed)
        return {"ok": True}

    @safe
    def delete_xray(self, xray_id):
        self._need_login()
        self._editable_xray(xray_id)
        self._store.delete_xray(xray_id)
        return {"ok": True}

    # ---- reports ---------------------------------------------------------------------
    def _patient_bundle(self, patient_id):
        p = self._store.patient(patient_id)
        visits = []
        for v in self._store.visits(patient_id):
            v["xrays"] = self._store.xrays(v["id"])
            visits.append(v)
        return p, self._store.history(patient_id), visits

    def _visit_for_report(self, patient_id):
        """The latest visit (the one being reported on) and the visits before it."""
        visits = self._store.visits(patient_id)
        if not visits:
            raise ValueError("This patient has no visits yet.")
        for v in visits:
            v["xrays"] = self._store.xrays(v["id"])
        return visits[0], visits[1:]

    def _deliver(self, path):
        """Keep the PDF in the patient's folder, put a copy in Downloads and open it."""
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        copy = None
        if os.path.isdir(downloads):
            copy = os.path.join(downloads, os.path.basename(path))
            shutil.copy2(path, copy)
        self._open(copy or path)
        return {"path": path, "download": copy, "file": os.path.basename(path)}

    @safe
    def export_report(self, patient_id):
        """PDF of the latest visit: signed if a doctor already signed it off, otherwise a draft."""
        self._need_login()
        p, hist = self._store.patient(patient_id), self._store.history(patient_id)
        visit, earlier = self._visit_for_report(patient_id)
        signer = self._store.doctor(visit["signed_by"]) if visit["signed_by"] else None
        path = pdf_report.write_pdf(self._store.patient_dir(patient_id), p, hist, visit, visit["xrays"],
                                    self._doctor, signer, earlier)
        if not signer:
            self._store.set_visit_report(visit["id"], os.path.basename(path))
        return self._deliver(path)

    @safe
    def sign_off(self, patient_id):
        """The doctor signs off the latest visit: a signed PDF report is saved in the patient's files."""
        self._need_doctor()
        p, hist = self._store.patient(patient_id), self._store.history(patient_id)
        visit, earlier = self._visit_for_report(patient_id)
        if visit["signed_by"]:
            raise ValueError(f"This visit was already signed off by {visit['signed_by']}.")
        if any(x["view"] == "unsure" for x in visit["xrays"]):
            raise ValueError("Pick the X-ray type for every X-ray in this visit before signing off.")
        for x in visit["xrays"]:  # signing off means the doctor has reviewed every X-ray in the visit
            if not x["reviewed"]:
                self._store.set_reviewed(x["id"], True)
                x["reviewed"] = 1
        path = pdf_report.write_pdf(self._store.patient_dir(patient_id), p, hist, visit, visit["xrays"],
                                    self._doctor, self._doctor, earlier)
        self._store.sign_off(visit["id"], self._doctor["doctor_id"], os.path.basename(path))
        return self._deliver(path)

    @safe
    def open_report(self, patient_id, file):
        self._need_login()
        path = os.path.join(self._store.patient_dir(patient_id), os.path.basename(file))
        if not os.path.exists(path):
            raise ValueError("That report file is no longer in the patient's folder.")
        self._open(path)
        return {"path": path}

    @safe
    def refer_patient(self, patient_id, to, reason):
        self._need_login()
        p, hist, visits = self._patient_bundle(patient_id)
        path = report.write_referral(self._store.patient_dir(patient_id), p, hist, visits, self._doctor, to, reason)
        vid = self._store.todays_visit(patient_id, self._doctor["doctor_id"])
        old = self._store.visit(vid)["note"]
        self._store.set_visit_note(vid, (old + "\n" if old else "") + f"Referred to {to or 'a specialist'}.")
        self._open(path)
        return {"path": path}

    @safe
    def open_patient_folder(self, patient_id):
        self._need_login()
        folder = self._store.patient_dir(patient_id)
        os.makedirs(folder, exist_ok=True)
        self._open(folder)
        return {"path": folder}
