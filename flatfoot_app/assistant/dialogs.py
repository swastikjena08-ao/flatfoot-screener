"""Pop-up windows: patient details, adding an X-ray, the draft report and settings."""
import datetime as dt
import os

from PySide6.QtCore import QMarginsF, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication, QPageLayout, QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QFileDialog, QHBoxLayout, QPlainTextEdit,
                               QRadioButton, QScrollArea, QVBoxLayout, QWidget)

from PIL import Image

import flatfoot_app as ff

from . import engine, theme
from .store import StoreError
from .widgets import button, card, chip, clear_layout, field, label, line_edit

SEXES = ("", "Female", "Male", "Other")


class Dialog(QDialog):
    def __init__(self, parent, title, width=520):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(width)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(26, 22, 26, 22)
        self.lay.setSpacing(12)
        self.lay.addWidget(label(title, h2=True))
        self.err = label("", err=True, wrap=True)
        self.err.hide()

    def fail(self, msg):
        self.err.setText(msg)
        self.err.show()

    def buttons(self, ok_text, ok):
        self.lay.addWidget(self.err)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Cancel", "ghost", self.reject))
        b = button(ok_text, "primary", ok)
        b.setDefault(True)
        row.addWidget(b)
        self.lay.addLayout(row)
        return b


class PatientDialog(Dialog):
    """New patient (patient=None) or edit details. self.result_row holds the saved patient."""

    def __init__(self, parent, store, user, patient=None):
        super().__init__(parent, "Edit patient details" if patient else "New patient")
        self.store, self.user, self.patient, self.result_row = store, user, patient, None
        self.code = line_edit("Your clinic's patient ID, e.g. P-10234")
        self.name = line_edit("Full name")
        self.sex = QComboBox()
        self.sex.addItems(["Not set", *SEXES[1:]])
        self.birth = line_edit("YYYY-MM-DD")
        self.phone = line_edit("Optional")
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("History, complaints, anything the doctor should know")
        self.notes.setFixedHeight(90)
        if patient:
            self.code.setText(patient["code"])
            self.code.setEnabled(False)
            self.name.setText(patient["name"])
            self.sex.setCurrentIndex(SEXES.index(patient["sex"]) if patient["sex"] in SEXES else 0)
            self.birth.setText(patient["birth_date"] or "")
            self.phone.setText(patient["phone"] or "")
            self.notes.setPlainText(patient["notes"] or "")
        self.lay.addWidget(field("Patient ID", self.code))
        self.lay.addWidget(field("Name", self.name))
        row = QHBoxLayout()
        row.addWidget(field("Sex", self.sex))
        row.addWidget(field("Date of birth", self.birth))
        self.lay.addLayout(row)
        self.lay.addWidget(field("Phone", self.phone))
        self.lay.addWidget(field("Notes", self.notes))
        self.buttons("Save" if patient else "Create patient", self.save)

    def save(self):
        birth = self.birth.text().strip()
        if birth:
            try:
                d = dt.date.fromisoformat(birth)
            except ValueError:
                return self.fail("Write the date of birth as YYYY-MM-DD, for example 1984-03-21.")
            if d > dt.date.today():
                return self.fail("The date of birth is in the future.")
        sex = SEXES[self.sex.currentIndex()]
        try:
            if self.patient:
                self.result_row = self.store.update_patient(
                    self.user["id"], self.patient["id"], name=self.name.text().strip(), sex=sex, birth_date=birth,
                    phone=self.phone.text().strip(), notes=self.notes.toPlainText().strip())
            else:
                self.result_row = self.store.add_patient(self.user["id"], self.code.text(), self.name.text(), sex,
                                                         birth, self.phone.text(), self.notes.toPlainText())
        except StoreError as e:
            return self.fail(str(e))
        self.accept()


class AddXrayDialog(Dialog):
    """Pick a file, its view and the foot. Result in self.choice = (path, view, side or None)."""

    def __init__(self, parent, engine_ready, has_finder, path=None):
        super().__init__(parent, "Add X-ray")
        self.has_finder = has_finder
        self.choice = None
        self.path = None
        self.file = label("No file chosen", muted=True, wrap=True)
        pick = button("Choose X-ray file…", "ghost", self.pick)
        self.lay.addWidget(field("X-ray image (PNG, JPG, TIFF, BMP)", pick))
        self.lay.addWidget(self.file)
        self.views = QButtonGroup(self)
        self.side_view = QRadioButton("Side view: flat foot and heel spur")
        self.top_view = QRadioButton("Top-down view: bunion angles")
        for i, b in enumerate((self.side_view, self.top_view)):
            self.views.addButton(b, i)
            self.lay.addWidget(b)
        self.side_view.setChecked(True)
        self.foot = QComboBox()
        self.lay.addWidget(field("Which foot", self.foot))
        self.hint = label("", muted=True, wrap=True)
        self.lay.addWidget(self.hint)
        self.views.idToggled.connect(lambda *_: self.fill_feet())
        self.fill_feet()
        if not engine_ready:
            self.hint.setText("The models are still loading. The X-ray is saved now and checked when they're ready.")
        self.buttons("Add and check", self.save)
        if path:
            self.set_path(path)

    def fill_feet(self):
        self.foot.clear()
        if self.side_view.isChecked():
            self.foot.addItems(["Find automatically", "Left foot", "Right foot"])
            self.hint.setText("The app finds the foot side from the X-ray if you leave it on automatic.")
        elif self.has_finder:
            self.foot.addItems(["Find both feet automatically", "Left foot only", "Right foot only"])
            self.hint.setText("The foot finder boxes each foot and measures both.")
        else:
            self.foot.addItems(["Left foot", "Right foot"])
            self.hint.setText("Use an X-ray of one foot. The foot finder model isn't installed on this PC,"
                              " so the whole image is measured as the foot you pick.")

    def pick(self):
        exts = " ".join("*" + e for e in ff.IMAGE_EXTS)
        path, _ = QFileDialog.getOpenFileName(self, "Choose an X-ray", "", f"Images ({exts})")
        if path:
            self.set_path(path)

    def set_path(self, path):
        try:
            img = engine.open_xray(path)
        except Exception:  # noqa: BLE001
            self.fail("That file couldn't be opened as an image.")
            return
        self.err.hide()
        self.path = path
        self.file.setText(f"{os.path.basename(path)}  ·  {img.width} × {img.height}")
        (self.side_view if engine.guess_view(img) == "side" else self.top_view).setChecked(True)

    def save(self):
        if not self.path:
            return self.fail("Choose an X-ray file first.")
        view = "side" if self.side_view.isChecked() else "top"
        text = self.foot.currentText().lower()
        side = "left" if text.startswith("left") else "right" if text.startswith("right") else None
        self.choice = (self.path, view, side)
        self.accept()


class ReportDialog(Dialog):
    def __init__(self, parent, store, user, patient, text, xray_ids):
        super().__init__(parent, f"Report · {patient['name']}", width=720)
        self.store, self.user, self.patient, self.xray_ids = store, user, patient, xray_ids
        self.signed = False
        self.edit = QPlainTextEdit(text)
        self.edit.setFont(QFont("Segoe UI", 11))
        self.edit.setMinimumHeight(440)
        self.lay.addWidget(self.edit)
        self.lay.addWidget(self.err)
        row = QHBoxLayout()
        row.addWidget(button("Copy text", "ghost", self.copy))
        row.addWidget(button("Save as PDF…", "ghost", self.save_pdf))
        row.addStretch(1)
        self.note = label("", muted=True)
        row.addWidget(self.note)
        row.addWidget(button("Close", "ghost", self.reject))
        self.sign = button("Sign off", "primary", self.sign_off)
        if user["role"] != "doctor":
            self.sign.setEnabled(False)
            self.sign.setToolTip("Only a doctor can sign off a report.")
        if not xray_ids:
            self.sign.setEnabled(False)
        row.addWidget(self.sign)
        self.lay.addLayout(row)

    def copy(self):
        QGuiApplication.clipboard().setText(self.edit.toPlainText())
        self.note.setText("Copied")

    def save_pdf(self, path=None):
        if not path:
            name = f"report_{self.patient['code']}_{dt.date.today():%Y%m%d}.pdf"
            path, _ = QFileDialog.getSaveFileName(self, "Save report", name, "PDF (*.pdf)")
        if not path:
            return None
        write_pdf(path, self.edit.toPlainText())
        self.note.setText("PDF saved")
        return path

    def sign_off(self):
        try:
            self.store.sign_off(self.user["id"], self.xray_ids)
        except StoreError as e:
            return self.fail(str(e))
        text = self.edit.toPlainText().rstrip() + f"\n\nSigned off by {self.user['name']} on {dt.datetime.now():%d %b %Y %H:%M}."
        self.edit.setPlainText(text)
        self.signed = True
        self.sign.setEnabled(False)
        self.note.setText("Signed off")


def write_pdf(path, text):
    pdf = QPdfWriter(path)
    pdf.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait, QMarginsF(18, 18, 18, 18),
                                  QPageLayout.Millimeter))
    pdf.setResolution(150)
    doc = QTextDocument()
    doc.setDefaultFont(QFont("Segoe UI", 10))
    doc.setPlainText(text)
    doc.setPageSize(pdf.pageLayout().paintRectPixels(pdf.resolution()).size().toSizeF())
    doc.print_(pdf)


class SettingsDialog(Dialog):
    toggle_theme = Signal()

    def __init__(self, parent, store, user, eng, scores):
        super().__init__(parent, "Settings", width=640)
        self.store, self.user = store, user
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        b = QVBoxLayout(body)
        b.setContentsMargins(0, 0, 0, 0)
        b.setSpacing(14)
        scroll.setWidget(body)
        scroll.setMinimumHeight(520)
        self.lay.addWidget(scroll)

        box, lay = card()
        lay.addWidget(label("Look", h2=True))
        lay.addWidget(label("Dark mode turns the app dark; the X-ray viewer stays black either way.", muted=True, wrap=True))
        self.theme_btn = button("Switch to light mode" if theme.MODE == "dark" else "Switch to dark mode", "ghost",
                                self._toggle_theme)
        lay.addWidget(self.theme_btn)
        b.addWidget(box)

        box, lay = card()
        lay.addWidget(label("Your password", h2=True))
        self.old = line_edit("Current password", password=True)
        self.new = line_edit("New password", password=True)
        self.new2 = line_edit("New password again", password=True)
        for e in (self.old, self.new, self.new2):
            lay.addWidget(e)
        self.pw_msg = label("", wrap=True)
        lay.addWidget(self.pw_msg)
        lay.addWidget(button("Change password", "primary", self.change_password))
        b.addWidget(box)

        box, lay = card()
        lay.addWidget(label("Accounts on this PC", h2=True))
        self.accounts = QVBoxLayout()
        lay.addLayout(self.accounts)
        if user["role"] == "doctor":
            lay.addWidget(label("Add an assistant. Assistants can add patients and X-rays but can't sign off reports.",
                                muted=True, wrap=True))
            self.a_name = line_edit("Assistant's name")
            self.a_login = line_edit("Assistant ID (no spaces)")
            self.a_pw = line_edit("Password for them", password=True)
            for e in (self.a_name, self.a_login, self.a_pw):
                lay.addWidget(e)
            self.a_msg = label("", wrap=True)
            lay.addWidget(self.a_msg)
            lay.addWidget(button("Add assistant", "primary", self.add_assistant))
        lay.addWidget(label("Another doctor needs their own activation key (Sign out, then Set up an account).",
                            muted=True, wrap=True))
        b.addWidget(box)
        self.fill_accounts()

        box, lay = card()
        lay.addWidget(label("Models", h2=True))
        names = {"flat": "Flat foot (heel bone angle)", "spur": "Heel spur", "bunion": "Bunion angles",
                 "fracture": "Fracture flag"}
        for k, title in names.items():
            st = eng.status.get(k, "loading") if eng else "loading"
            colour = "green" if st.startswith("ready") else "amber"
            row = QHBoxLayout()
            row.addWidget(label(title))
            row.addStretch(1)
            row.addWidget(chip(st if len(st) < 60 else st[:57] + "…", colour))
            lay.addLayout(row)
            s = scores.get(k, {})
            if s.get("line"):
                lay.addWidget(label(s["line"], muted=True, wrap=True))
        b.addWidget(box)

        box, lay = card()
        lay.addWidget(label("Data folder", h2=True))
        lay.addWidget(label("Patients, X-ray copies and results are stored only on this PC, in:", muted=True, wrap=True))
        lay.addWidget(label(store.folder, wrap=True))
        lay.addWidget(button("Open folder", "ghost",
                             lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(store.folder))))
        b.addWidget(box)
        b.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Close", "ghost", self.accept))
        self.lay.addLayout(row)

    def _toggle_theme(self):
        self.toggle_theme.emit()
        self.theme_btn.setText("Switch to light mode" if theme.MODE == "dark" else "Switch to dark mode")

    def fill_accounts(self):
        clear_layout(self.accounts)
        for u in self.store.users():
            row = QHBoxLayout()
            row.addWidget(label(f"{u['name']}  ·  {u['login']}"))
            row.addStretch(1)
            row.addWidget(chip(u["role"].capitalize(), "blue" if u["role"] == "doctor" else "grey"))
            self.accounts.addLayout(row)

    def change_password(self):
        if self.new.text() != self.new2.text():
            self.pw_msg.setProperty("err", "true")
            self.pw_msg.setText("The two new passwords don't match.")
            return
        try:
            self.store.change_password(self.user["id"], self.old.text(), self.new.text())
        except StoreError as e:
            self.pw_msg.setText(str(e))
            return
        for e in (self.old, self.new, self.new2):
            e.clear()
        self.pw_msg.setText("Password changed.")

    def add_assistant(self):
        try:
            self.store.add_assistant(self.user["id"], self.a_login.text(), self.a_name.text(), self.a_pw.text())
        except StoreError as e:
            self.a_msg.setText(str(e))
            return
        for e in (self.a_name, self.a_login, self.a_pw):
            e.clear()
        self.a_msg.setText("Assistant added. They can sign in now.")
        self.fill_accounts()


def thumb_of(path, max_side=360):
    img = engine.open_xray(path)
    img.thumbnail((max_side, max_side), Image.BILINEAR)
    return engine.gray_of(img)
