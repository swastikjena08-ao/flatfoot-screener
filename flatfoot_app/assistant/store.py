"""Everything the app saves: accounts, patients, X-rays and model results.

One SQLite file plus a folder of X-ray copies, by default in
%APPDATA%\\Foot X-ray Assistant on Windows (~/.foot-xray-assistant elsewhere).
Nothing leaves the PC.
"""
import json
import os
import shutil
import sqlite3
import sys
import time
import uuid

from . import auth

LOCK_AFTER = 5        # wrong passwords in a row
LOCK_SECONDS = 60
ROLES = ("doctor", "assistant")
STATUSES = ("new", "review", "done")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    login TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('doctor', 'assistant')),
    pw_hash TEXT NOT NULL, pw_salt TEXT NOT NULL, pw_iter INTEGER NOT NULL,
    created_at REAL NOT NULL, created_by INTEGER,
    failed INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0,
    last_login REAL
);
CREATE TABLE IF NOT EXISTS activations (
    key_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    used_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name TEXT NOT NULL,
    sex TEXT NOT NULL DEFAULT '',
    birth_date TEXT NOT NULL DEFAULT '',
    phone TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    flagged INTEGER NOT NULL DEFAULT 0,
    created_by INTEGER REFERENCES users(id),
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS xrays (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    file TEXT NOT NULL,
    original_name TEXT NOT NULL,
    view TEXT NOT NULL CHECK (view IN ('side', 'top')),
    side TEXT NOT NULL DEFAULT '',
    added_by INTEGER REFERENCES users(id),
    added_at REAL NOT NULL,
    results TEXT,
    edits TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'new',
    note TEXT NOT NULL DEFAULT '',
    signed_by INTEGER REFERENCES users(id),
    signed_at REAL,
    marked TEXT NOT NULL DEFAULT '',
    marks TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    at REAL NOT NULL
);
"""


class StoreError(Exception):
    """A problem the person can fix; the message is shown as is."""


def default_dir():
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Foot X-ray Assistant")
    return os.path.join(os.path.expanduser("~"), ".foot-xray-assistant")


class Store:
    def __init__(self, folder=None, key_file=None):
        self.folder = folder or default_dir()
        os.makedirs(os.path.join(self.folder, "xrays"), exist_ok=True)
        self.db = sqlite3.connect(os.path.join(self.folder, "assistant.db"))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(xrays)")}
        if "marked" not in cols:          # databases made before marked copies existed
            self.db.execute("ALTER TABLE xrays ADD COLUMN marked TEXT NOT NULL DEFAULT ''")
        if "marks" not in cols:           # ... and before the doctor's drawings and notes
            self.db.execute("ALTER TABLE xrays ADD COLUMN marks TEXT NOT NULL DEFAULT '{}'")
        self.db.commit()
        self.key_hashes = auth.load_key_hashes(key_file) if key_file else set()

    # ---------- preferences for this PC (light/dark look) ----------

    def prefs(self):
        try:
            with open(os.path.join(self.folder, "prefs.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def set_pref(self, key, value):
        p = self.prefs()
        p[key] = value
        with open(os.path.join(self.folder, "prefs.json"), "w", encoding="utf-8") as f:
            json.dump(p, f)

    def close(self):
        self.db.close()

    def log(self, user_id, action, detail=""):
        self.db.execute("INSERT INTO audit (user_id, action, detail, at) VALUES (?, ?, ?, ?)",
                        (user_id, action, detail, time.time()))
        self.db.commit()

    # ---------- accounts ----------

    def has_users(self):
        return self.db.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None

    def user(self, user_id):
        return self.db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def users(self):
        return self.db.execute("SELECT * FROM users ORDER BY role, name").fetchall()

    def _new_user(self, login, name, role, password, created_by=None):
        login, name = login.strip(), name.strip()
        if not login or not name:
            raise StoreError("Fill in the ID and the name.")
        if any(c.isspace() for c in login):
            raise StoreError("The ID can't contain spaces.")
        if role not in ROLES:
            raise StoreError("Unknown role.")
        problem = auth.password_problem(password)
        if problem:
            raise StoreError(problem)
        if self.db.execute("SELECT 1 FROM users WHERE login = ?", (login,)).fetchone():
            raise StoreError(f"The ID {login} is already taken on this PC.")
        h, salt, it = auth.hash_password(password)
        cur = self.db.execute(
            "INSERT INTO users (login, name, role, pw_hash, pw_salt, pw_iter, created_at, created_by)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (login, name, role, h, salt, it, time.time(), created_by))
        return cur.lastrowid

    def activate(self, key_text, login, name, password, role="doctor"):
        """Create an account with a one-time activation key. Returns the new user row."""
        key = auth.normalise_key(key_text or "")
        if key is None:
            raise StoreError("That doesn't look like an activation key. It has 16 letters and numbers after FXA.")
        kh = auth.key_hash(key)
        if kh not in self.key_hashes:
            raise StoreError("This activation key isn't valid. Check it for typos.")
        if self.db.execute("SELECT 1 FROM activations WHERE key_hash = ?", (kh,)).fetchone():
            raise StoreError("This activation key has already been used. Each key works once.")
        uid = self._new_user(login, name, role, password)
        self.db.execute("INSERT INTO activations (key_hash, user_id, used_at) VALUES (?, ?, ?)",
                        (kh, uid, time.time()))
        self.db.commit()
        self.log(uid, "activate", f"{role} account {login}")
        return self.user(uid)

    def add_assistant(self, doctor_id, login, name, password):
        """A signed-in doctor adds an assistant account (no key needed)."""
        doc = self.user(doctor_id)
        if doc is None or doc["role"] != "doctor":
            raise StoreError("Only a doctor can add accounts.")
        uid = self._new_user(login, name, "assistant", password, created_by=doctor_id)
        self.db.commit()
        self.log(doctor_id, "add_assistant", login)
        return self.user(uid)

    def sign_in(self, login, password, now=None):
        """Returns the user row, or raises StoreError with the reason."""
        now = time.time() if now is None else now
        row = self.db.execute("SELECT * FROM users WHERE login = ?", (login.strip(),)).fetchone()
        if row is None:
            raise StoreError("That ID and password don't match an account on this PC.")
        if row["locked_until"] > now:
            wait = int(row["locked_until"] - now) + 1
            raise StoreError(f"Too many wrong passwords. Try again in {wait} seconds.")
        if not auth.check_password(password, row["pw_hash"], row["pw_salt"], row["pw_iter"]):
            failed = row["failed"] + 1
            locked = now + LOCK_SECONDS if failed >= LOCK_AFTER else 0
            self.db.execute("UPDATE users SET failed = ?, locked_until = ? WHERE id = ?",
                            (0 if locked else failed, locked, row["id"]))
            self.db.commit()
            self.log(row["id"], "sign_in_failed")
            if locked:
                raise StoreError(f"Too many wrong passwords. Try again in {LOCK_SECONDS} seconds.")
            raise StoreError("That ID and password don't match an account on this PC.")
        self.db.execute("UPDATE users SET failed = 0, locked_until = 0, last_login = ? WHERE id = ?", (now, row["id"]))
        self.db.commit()
        self.log(row["id"], "sign_in")
        return self.user(row["id"])

    def change_password(self, user_id, old, new):
        row = self.user(user_id)
        if not auth.check_password(old, row["pw_hash"], row["pw_salt"], row["pw_iter"]):
            raise StoreError("Your current password is wrong.")
        problem = auth.password_problem(new)
        if problem:
            raise StoreError(problem)
        h, salt, it = auth.hash_password(new)
        self.db.execute("UPDATE users SET pw_hash = ?, pw_salt = ?, pw_iter = ? WHERE id = ?", (h, salt, it, user_id))
        self.db.commit()
        self.log(user_id, "change_password")

    # ---------- patients ----------

    def add_patient(self, user_id, code, name, sex="", birth_date="", phone="", notes=""):
        code, name = code.strip(), name.strip()
        if not code:
            raise StoreError("Enter a Patient ID.")
        if not name:
            raise StoreError("Enter the patient's name.")
        if self.patient_by_code(code):
            raise StoreError(f"A patient with ID {code} already exists. Open that profile instead.")
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO patients (code, name, sex, birth_date, phone, notes, created_by, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (code, name, sex, birth_date, phone.strip(), notes.strip(), user_id, now, now))
        self.db.commit()
        self.log(user_id, "add_patient", code)
        return self.patient(cur.lastrowid)

    def update_patient(self, user_id, patient_id, **fields):
        allowed = {"name", "sex", "birth_date", "phone", "notes", "flagged"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if "name" in fields and not str(fields["name"]).strip():
            raise StoreError("Enter the patient's name.")
        if not fields:
            return self.patient(patient_id)
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE patients SET {sets}, updated_at = ? WHERE id = ?",
                        (*fields.values(), time.time(), patient_id))
        self.db.commit()
        self.log(user_id, "update_patient", f"{patient_id}: {', '.join(fields)}")
        return self.patient(patient_id)

    def patient(self, patient_id):
        return self.db.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()

    def patient_by_code(self, code):
        return self.db.execute("SELECT * FROM patients WHERE code = ?", (code.strip(),)).fetchone()

    def patients(self, query="", only_flagged=False, status=None):
        """Patients with their latest X-ray time and how many X-rays are in each status."""
        sql = ("SELECT p.*, MAX(x.added_at) AS last_xray, COUNT(x.id) AS n_xrays,"
               " SUM(x.status = 'new') AS n_new, SUM(x.status = 'review') AS n_review"
               " FROM patients p LEFT JOIN xrays x ON x.patient_id = p.id WHERE 1=1")
        args = []
        if query.strip():
            sql += " AND (p.name LIKE ? OR p.code LIKE ? OR p.phone LIKE ?)"
            q = f"%{query.strip()}%"
            args += [q, q, q]
        if only_flagged:
            sql += " AND p.flagged = 1"
        sql += " GROUP BY p.id"
        if status == "review":
            sql += " HAVING SUM(x.status IN ('new', 'review')) > 0"
        sql += " ORDER BY COALESCE(MAX(x.added_at), p.created_at) DESC"
        return self.db.execute(sql, args).fetchall()

    # ---------- X-rays ----------

    def add_xray(self, user_id, patient_id, src_path, view, side=""):
        if view not in ("side", "top"):
            raise StoreError("Choose side view or top-down view.")
        p = self.patient(patient_id)
        ext = os.path.splitext(src_path)[1].lower() or ".png"
        rel = os.path.join("xrays", _safe(p["code"]), uuid.uuid4().hex + ext)
        os.makedirs(os.path.dirname(os.path.join(self.folder, rel)), exist_ok=True)
        shutil.copyfile(src_path, os.path.join(self.folder, rel))
        cur = self.db.execute(
            "INSERT INTO xrays (patient_id, file, original_name, view, side, added_by, added_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (patient_id, rel, os.path.basename(src_path), view, side, user_id, time.time()))
        self.db.execute("UPDATE patients SET updated_at = ? WHERE id = ?", (time.time(), patient_id))
        self.db.commit()
        self.log(user_id, "add_xray", f"{p['code']}: {os.path.basename(src_path)} ({view})")
        return self.xray(cur.lastrowid)

    def xray(self, xray_id):
        return self.db.execute("SELECT * FROM xrays WHERE id = ?", (xray_id,)).fetchone()

    def xrays(self, patient_id):
        return self.db.execute("SELECT * FROM xrays WHERE patient_id = ? ORDER BY added_at DESC, id DESC",
                               (patient_id,)).fetchall()

    def xray_path(self, row):
        return os.path.join(self.folder, row["file"])

    def delete_xray(self, user_id, xray_id):
        """Removes the X-ray, its results and its saved copies from this PC."""
        x = self.xray(xray_id)
        if x is None:
            return
        p = self.patient(x["patient_id"])
        for rel in (x["file"], x["marked"]):
            if rel:
                try:
                    os.remove(os.path.join(self.folder, rel))
                except OSError:
                    pass
        self.db.execute("DELETE FROM xrays WHERE id = ?", (xray_id,))
        self.db.execute("UPDATE patients SET updated_at = ? WHERE id = ?", (time.time(), x["patient_id"]))
        self.db.commit()
        self.log(user_id, "delete_xray", f"{p['code'] if p else '?'}: {x['original_name']} ({x['view']})")

    def marked_path(self, row):
        """Where the app keeps this X-ray's marked copy (in the data folder, next to the X-ray)."""
        base = os.path.splitext(row["file"])[0]
        return os.path.join(self.folder, base + "_marked.png")

    def set_marked(self, user_id, xray_id, path):
        self.db.execute("UPDATE xrays SET marked = ? WHERE id = ?", (os.path.relpath(path, self.folder), xray_id))
        self.db.commit()
        self.log(user_id, "save_marked", str(xray_id))

    def save_results(self, xray_id, results, side=None):
        """Model output for one X-ray. A fresh run clears the hand edits and needs review again."""
        self.db.execute("UPDATE xrays SET results = ?, edits = '{}', status = 'review', signed_by = NULL,"
                        " signed_at = NULL, side = COALESCE(?, side) WHERE id = ?",
                        (json.dumps(results), side, xray_id))
        self.db.commit()

    def save_edits(self, user_id, xray_id, edits):
        self.db.execute("UPDATE xrays SET edits = ? WHERE id = ?", (json.dumps(edits), xray_id))
        self.db.commit()
        self.log(user_id, "edit_points", str(xray_id))

    def save_marks(self, user_id, xray_id, marks):
        """The doctor's drawings and typed notes on the X-ray. Kept when the models re-check it."""
        self.db.execute("UPDATE xrays SET marks = ? WHERE id = ?", (json.dumps(marks), xray_id))
        self.db.commit()
        self.log(user_id, "edit_marks", str(xray_id))

    def save_note(self, user_id, xray_id, note):
        self.db.execute("UPDATE xrays SET note = ? WHERE id = ?", (note, xray_id))
        self.db.commit()

    def sign_off(self, user_id, xray_ids):
        u = self.user(user_id)
        if u is None or u["role"] != "doctor":
            raise StoreError("Only a doctor can sign off a report.")
        now = time.time()
        self.db.executemany("UPDATE xrays SET status = 'done', signed_by = ?, signed_at = ? WHERE id = ?",
                            [(user_id, now, i) for i in xray_ids])
        self.db.commit()
        self.log(user_id, "sign_off", ",".join(map(str, xray_ids)))

    def counts(self):
        one = lambda sql: self.db.execute(sql).fetchone()[0] or 0  # noqa: E731
        return {"patients": one("SELECT COUNT(*) FROM patients"),
                "flagged": one("SELECT COUNT(*) FROM patients WHERE flagged = 1"),
                "to_review": one("SELECT COUNT(*) FROM xrays WHERE status IN ('new', 'review')"),
                "checks": one("SELECT COUNT(*) FROM xrays WHERE results IS NOT NULL")}


def _safe(code):
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in code)
    return keep or "patient"


def loads(text, default=None):
    if not text:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default
