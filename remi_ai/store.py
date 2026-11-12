"""Local storage: doctor accounts, patients, history, visits and X-rays.

Everything lives in one folder on this computer (default %APPDATA%\\Remi AI):
    remi.db                   SQLite database
    patients/<patient id>/    each patient's own X-ray files
"""
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    clinic    TEXT NOT NULL DEFAULT '',
    salt      BLOB NOT NULL,
    pw_hash   BLOB NOT NULL,
    created   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    age        INTEGER,
    sex        TEXT NOT NULL DEFAULT '',
    phone      TEXT NOT NULL DEFAULT '',
    referred_by TEXT NOT NULL DEFAULT '',
    complaint  TEXT NOT NULL DEFAULT '',
    created    TEXT NOT NULL,
    created_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    item       TEXT NOT NULL,
    value      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visits (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    date       TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    note       TEXT NOT NULL DEFAULT '',
    doctor_id  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS xrays (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id  INTEGER NOT NULL REFERENCES visits(id) ON DELETE CASCADE,
    file      TEXT NOT NULL,
    view      TEXT NOT NULL,
    view_source TEXT NOT NULL,
    view_conf REAL,
    results   TEXT NOT NULL DEFAULT '{}',
    reviewed  INTEGER NOT NULL DEFAULT 0,
    added     TEXT NOT NULL
);
"""

DEFAULT_HISTORY = ["Diabetes", "Blood pressure", "Medicines", "Past foot surgery", "Allergies"]


def default_data_dir():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "Remi AI")


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


def today():
    return dt.date.today().isoformat()


class Store:
    def __init__(self, data_dir=None):
        self.dir = data_dir or default_data_dir()
        os.makedirs(os.path.join(self.dir, "patients"), exist_ok=True)
        # The window bridge calls in from worker threads; Api serialises access with a lock.
        self.db = sqlite3.connect(os.path.join(self.dir, "remi.db"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()

    def _migrate(self):
        """Add columns introduced after the first release to an existing database."""
        def add(table, column, ddl):
            cols = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        add("doctors", "role", "TEXT NOT NULL DEFAULT 'doctor'")      # doctor | assistant
        add("doctors", "supervisor", "TEXT NOT NULL DEFAULT ''")      # an assistant's doctor
        add("visits", "signed_by", "TEXT NOT NULL DEFAULT ''")
        add("visits", "signed_at", "TEXT NOT NULL DEFAULT ''")
        add("visits", "report", "TEXT NOT NULL DEFAULT ''")           # PDF file name in the patient folder
        add("xrays", "marks", "TEXT NOT NULL DEFAULT '[]'")           # the doctor's pen strokes and flag labels

    # ---- settings ------------------------------------------------------------------
    def setting(self, key, default=None):
        r = self.db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set_setting(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    # ---- doctors -------------------------------------------------------------------
    def has_doctors(self):
        return self.db.execute("SELECT 1 FROM doctors LIMIT 1").fetchone() is not None

    def add_doctor(self, doctor_id, name, password, clinic="", role="doctor", supervisor=""):
        doctor_id = doctor_id.strip().upper()
        if not doctor_id or not name.strip():
            raise ValueError("ID and name are required.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")
        if self.doctor(doctor_id):
            raise ValueError(f"ID {doctor_id} already exists.")
        salt = secrets.token_bytes(16)
        self.db.execute("INSERT INTO doctors (doctor_id, name, clinic, salt, pw_hash, created, role, supervisor) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (doctor_id, name.strip(), clinic.strip(), salt, hash_password(password, salt),
                         dt.datetime.now().isoformat(timespec="seconds"), role, supervisor))
        self.db.commit()
        return self.doctor(doctor_id)

    def doctor(self, doctor_id):
        r = self.db.execute("SELECT doctor_id, name, clinic, role, supervisor, created FROM doctors "
                            "WHERE doctor_id = ?", (doctor_id.strip().upper(),)).fetchone()
        return dict(r) if r else None

    def check_login(self, doctor_id, password):
        r = self.db.execute("SELECT salt, pw_hash FROM doctors WHERE doctor_id = ?",
                            (doctor_id.strip().upper(),)).fetchone()
        if r is None or not hmac.compare_digest(hash_password(password, r["salt"]), r["pw_hash"]):
            return None
        return self.doctor(doctor_id)

    def set_password(self, doctor_id, password):
        if len(password) < 6:
            raise ValueError("New password must be at least 6 characters.")
        salt = secrets.token_bytes(16)
        self.db.execute("UPDATE doctors SET salt = ?, pw_hash = ? WHERE doctor_id = ?",
                        (salt, hash_password(password, salt), doctor_id))
        self.db.commit()

    def update_account(self, doctor_id, name, clinic):
        if not name.strip():
            raise ValueError("Name can't be empty.")
        self.db.execute("UPDATE doctors SET name = ?, clinic = ? WHERE doctor_id = ?",
                        (name.strip(), clinic.strip(), doctor_id))
        self.db.commit()

    def assistants(self, supervisor):
        rows = self.db.execute("SELECT doctor_id, name, created FROM doctors WHERE role = 'assistant' "
                               "AND supervisor = ? ORDER BY created", (supervisor,)).fetchall()
        return [dict(r) for r in rows]

    def delete_account(self, doctor_id):
        self.db.execute("DELETE FROM doctors WHERE doctor_id = ?", (doctor_id,))
        self.db.commit()

    # ---- patients ------------------------------------------------------------------
    def add_patient(self, patient_id, name, doctor_id, age=None, sex="", phone="", referred_by="",
                    complaint="", history=None):
        patient_id = patient_id.strip().upper()
        if not patient_id or not name.strip():
            raise ValueError("Patient ID and name are required.")
        if self.patient(patient_id):
            raise ValueError(f"Patient ID {patient_id} already exists.")
        self.db.execute("INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (patient_id, name.strip(), age, sex, phone.strip(), referred_by.strip(),
                         complaint.strip(), today(), doctor_id))
        for item, value in (history or {}).items():
            if value.strip():
                self.db.execute("INSERT INTO history (patient_id, item, value) VALUES (?, ?, ?)",
                                (patient_id, item, value.strip()))
        self.db.commit()
        os.makedirs(self.patient_dir(patient_id), exist_ok=True)
        return self.patient(patient_id)

    def update_patient(self, patient_id, **fields):
        allowed = {"name", "age", "sex", "phone", "referred_by", "complaint"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if sets:
            cols = ", ".join(f"{k} = ?" for k in sets)
            self.db.execute(f"UPDATE patients SET {cols} WHERE patient_id = ?", (*sets.values(), patient_id))
            self.db.commit()

    def set_history(self, patient_id, history):
        self.db.execute("DELETE FROM history WHERE patient_id = ?", (patient_id,))
        for item, value in history.items():
            if value.strip():
                self.db.execute("INSERT INTO history (patient_id, item, value) VALUES (?, ?, ?)",
                                (patient_id, item, value.strip()))
        self.db.commit()

    def patient(self, patient_id):
        r = self.db.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        return dict(r) if r else None

    def patients(self, query=""):
        q = f"%{query.strip()}%"
        rows = self.db.execute(
            "SELECT p.*, (SELECT MAX(date) FROM visits v WHERE v.patient_id = p.patient_id) AS last_visit, "
            "(SELECT COUNT(*) FROM xrays x JOIN visits v ON x.visit_id = v.id "
            " WHERE v.patient_id = p.patient_id) AS n_xrays "
            "FROM patients p WHERE p.name LIKE ? OR p.patient_id LIKE ? "
            "ORDER BY COALESCE(last_visit, p.created) DESC, p.name", (q, q)).fetchall()
        return [dict(r) for r in rows]

    def history(self, patient_id):
        rows = self.db.execute("SELECT item, value FROM history WHERE patient_id = ? ORDER BY id",
                               (patient_id,)).fetchall()
        return {r["item"]: r["value"] for r in rows}

    def patient_dir(self, patient_id):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in patient_id)
        return os.path.join(self.dir, "patients", safe)

    # ---- visits --------------------------------------------------------------------
    def add_visit(self, patient_id, doctor_id, label="", date=None):
        if not label:
            label = "First visit" if not self.visits(patient_id) else "Follow-up"
        cur = self.db.execute("INSERT INTO visits (patient_id, date, label, doctor_id) VALUES (?, ?, ?, ?)",
                              (patient_id, date or today(), label, doctor_id))
        self.db.commit()
        return cur.lastrowid

    def todays_visit(self, patient_id, doctor_id):
        # A signed-off visit is closed: anything added afterwards starts a new visit.
        r = self.db.execute("SELECT id FROM visits WHERE patient_id = ? AND date = ? AND signed_by = '' "
                            "ORDER BY id DESC", (patient_id, today())).fetchone()
        return r["id"] if r else self.add_visit(patient_id, doctor_id)

    def visits(self, patient_id):
        rows = self.db.execute("SELECT * FROM visits WHERE patient_id = ? ORDER BY date DESC, id DESC",
                               (patient_id,)).fetchall()
        return [dict(r) for r in rows]

    def visit(self, visit_id):
        r = self.db.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
        return dict(r) if r else None

    def sign_off(self, visit_id, doctor_id, report_file):
        self.db.execute("UPDATE visits SET signed_by = ?, signed_at = ?, report = ? WHERE id = ?",
                        (doctor_id, dt.datetime.now().isoformat(timespec="seconds"), report_file, visit_id))
        self.db.commit()

    def set_visit_report(self, visit_id, report_file):
        self.db.execute("UPDATE visits SET report = ? WHERE id = ?", (report_file, visit_id))
        self.db.commit()

    def set_visit_note(self, visit_id, note):
        self.db.execute("UPDATE visits SET note = ? WHERE id = ?", (note, visit_id))
        self.db.commit()

    # ---- X-rays --------------------------------------------------------------------
    def add_xray(self, visit_id, source_path, view, view_source, view_conf=None, results=None):
        pid = self.db.execute("SELECT patient_id FROM visits WHERE id = ?", (visit_id,)).fetchone()[0]
        folder = self.patient_dir(pid)
        os.makedirs(folder, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{secrets.token_hex(2)}{os.path.splitext(source_path)[1].lower() or '.png'}"
        shutil.copy2(source_path, os.path.join(folder, name))
        cur = self.db.execute(
            "INSERT INTO xrays (visit_id, file, view, view_source, view_conf, results, added) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (visit_id, name, view, view_source, view_conf, json.dumps(results or {}),
             dt.datetime.now().isoformat(timespec="seconds")))
        self.db.commit()
        return cur.lastrowid

    def set_results(self, xray_id, results, view=None, view_source=None):
        if view is not None:
            self.db.execute("UPDATE xrays SET view = ?, view_source = ? WHERE id = ?", (view, view_source, xray_id))
        self.db.execute("UPDATE xrays SET results = ? WHERE id = ?", (json.dumps(results), xray_id))
        self.db.commit()

    def set_marks(self, xray_id, marks):
        self.db.execute("UPDATE xrays SET marks = ? WHERE id = ?", (json.dumps(marks), xray_id))
        self.db.commit()

    def set_reviewed(self, xray_id, reviewed=True):
        self.db.execute("UPDATE xrays SET reviewed = ? WHERE id = ?", (int(reviewed), xray_id))
        self.db.commit()

    def delete_xray(self, xray_id):
        x = self.xray(xray_id)
        if x is None:
            return
        self.db.execute("DELETE FROM xrays WHERE id = ?", (xray_id,))
        self.db.commit()
        try:
            os.remove(x["path"])
        except OSError:
            pass

    def _xray_row(self, r):
        d = dict(r)
        d["results"] = json.loads(d["results"] or "{}")
        d["marks"] = json.loads(d.get("marks") or "[]")
        pid = self.db.execute("SELECT patient_id FROM visits WHERE id = ?", (d["visit_id"],)).fetchone()[0]
        d["path"] = os.path.join(self.patient_dir(pid), d["file"])
        d["patient_id"] = pid
        return d

    def xray(self, xray_id):
        r = self.db.execute("SELECT * FROM xrays WHERE id = ?", (xray_id,)).fetchone()
        return self._xray_row(r) if r else None

    def xrays(self, visit_id):
        rows = self.db.execute("SELECT * FROM xrays WHERE visit_id = ? ORDER BY id", (visit_id,)).fetchall()
        return [self._xray_row(r) for r in rows]

    def previous_xray(self, patient_id, before_visit_id, view):
        """Most recent X-ray of the same view from an earlier visit, for comparison."""
        row = self.db.execute(
            "SELECT x.* FROM xrays x JOIN visits v ON x.visit_id = v.id "
            "WHERE v.patient_id = ? AND x.view = ? AND x.results != '{}' "
            "AND (v.date < (SELECT date FROM visits WHERE id = ?) "
            "     OR (v.date = (SELECT date FROM visits WHERE id = ?) AND v.id < ?)) "
            "ORDER BY v.date DESC, v.id DESC, x.id DESC LIMIT 1",
            (patient_id, view, before_visit_id, before_visit_id, before_visit_id)).fetchone()
        return self._xray_row(row) if row else None

    def unreviewed_count(self, patient_id):
        return self.db.execute(
            "SELECT COUNT(*) FROM xrays x JOIN visits v ON x.visit_id = v.id "
            "WHERE v.patient_id = ? AND x.reviewed = 0 AND x.results != '{}'", (patient_id,)).fetchone()[0]
