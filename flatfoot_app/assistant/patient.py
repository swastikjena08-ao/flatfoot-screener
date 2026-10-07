"""One patient's page: the black X-ray lightbox on the left, result cards on the right."""
import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMessageBox, QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget

import flatfoot_app as ff

from . import engine, report
from .dialogs import AddXrayDialog, PatientDialog, ReportDialog, thumb_of
from .pages import patient_facts, when
from .store import loads
from .viewer import Drawer, to_pixmap_gray
from .widgets import button, card, chips, clear_layout, label, set_on

VIEW_NAMES = {"side": "Side view", "top": "Top-down view"}
STATUS_CHIPS = {"new": ("Waiting for the models", "grey"), "review": ("To review", "blue"), "done": ("Signed off", "green")}


def upsample(heat, w, h, max_side=512):
    heat = np.asarray(heat, np.float32)
    s = min(1.0, max_side / max(w, h))
    return ff.upsample_cam(heat, max(2, round(w * s)), max(2, round(h * s)))


class ModelCard(QFrame):
    def __init__(self, title, score_line):
        super().__init__()
        self.setProperty("card", "true")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(6)
        lay.addWidget(label(title, h2=True))
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        lay.addLayout(self.body)
        self.score = label(score_line, muted=True, wrap=True, size=12)
        lay.addWidget(self.score)

    def show_empty(self, text):
        clear_layout(self.body)
        self.body.addWidget(label(text, muted=True, wrap=True))

    def show_rows(self, rows, source=""):
        """rows: [(big text or None, [(chip, colour)], detail)]"""
        clear_layout(self.body)
        for big, cs, detail in rows:
            if big:
                self.body.addWidget(label(big, big=True))
            if cs:
                self.body.addWidget(chips(cs))
            if detail:
                self.body.addWidget(label(detail, muted=True, wrap=True))
        if source:
            self.body.addWidget(label(source, muted=True, size=11))


class PatientPage(QWidget):
    back = Signal()
    analyse = Signal(int)       # xray id
    changed = Signal()          # anything the home list shows

    def __init__(self, store, scores):
        super().__init__()
        self.store, self.scores = store, scores
        self.user = self.patient = None
        self.engine = None
        self.xrays, self.current = [], None
        self.busy = set()
        self.threshold = 17.7

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(14)
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(button("←", "round", self.back.emit, tip="Back to patients"))
        who = QVBoxLayout()
        who.setSpacing(0)
        self.name = label("", h1=True)
        self.facts = label("", muted=True)
        who.addWidget(self.name)
        who.addWidget(self.facts)
        top.addLayout(who, 1)
        self.flag = button("⚑  Flag", "ghost", self.toggle_flag)
        top.addWidget(self.flag)
        top.addWidget(button("Edit details", "ghost", self.edit_patient))
        top.addWidget(button("Draft report", "primary", self.open_report))
        outer.addLayout(top)

        main = QHBoxLayout()
        main.setSpacing(16)
        self.drawer = Drawer()
        self.drawer.picked.connect(self.show_xray)
        self.drawer.add_clicked.connect(self.add_xray)
        self.drawer.reset_clicked.connect(self.reset_points)
        self.drawer.canvas.points_edited.connect(self.points_edited)
        main.addWidget(self.drawer, 7)

        side = QWidget()
        col = QVBoxLayout(side)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(12)
        self.arch_card = ModelCard("Arch · flat foot", self.score_line("flat"))
        self.spur_card = ModelCard("Heel spur", self.score_line("spur"))
        self.bunion_card = ModelCard("Bunion angles", self.score_line("bunion"))
        self.fracture_card = ModelCard("Fracture flag", self.score_line("fracture"))
        for c in (self.arch_card, self.spur_card, self.bunion_card, self.fracture_card):
            col.addWidget(c)
        self.xray_box, self.xray_lay = card(spacing=8)
        col.addWidget(self.xray_box)
        self.details_box, self.details_lay = card(spacing=6)
        col.addWidget(self.details_box)
        col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(400)
        main.addWidget(scroll)
        outer.addLayout(main, 1)

    def score_line(self, key):
        return self.scores.get(key, {}).get("line", "")

    # ---------- loading ----------

    def open(self, user, patient_id, engine_obj):
        self.user, self.engine = user, engine_obj
        if engine_obj and engine_obj.flat_lm:
            self.threshold = engine_obj.lm_meta.get("threshold_deg", 17.7)
        self.patient = self.store.patient(patient_id)
        self.show_header()
        self.reload(select=None)

    def show_header(self):
        p = self.patient
        self.name.setText(p["name"])
        self.facts.setText(" · ".join(x for x in (f"Patient ID {p['code']}", patient_facts(p), p["phone"]) if x))
        set_on(self.flag, bool(p["flagged"]))
        self.flag.setText("⚑  Flagged" if p["flagged"] else "⚑  Flag")
        clear_layout(self.details_lay)
        self.details_lay.addWidget(label("Patient details", h2=True))
        rows = [("Patient ID", p["code"]), ("Date of birth", p["birth_date"] or "Not set"),
                ("Sex", p["sex"] or "Not set"), ("Phone", p["phone"] or "Not set"),
                ("Added", when(p["created_at"]))]
        for k, v in rows:
            r = QHBoxLayout()
            r.addWidget(label(k, muted=True))
            r.addStretch(1)
            r.addWidget(label(v))
            self.details_lay.addLayout(r)
        if p["notes"]:
            self.details_lay.addWidget(label(p["notes"], wrap=True))

    def reload(self, select=None):
        self.xrays = self.store.xrays(self.patient["id"])
        items = []
        for x in self.xrays:
            try:
                pix = to_pixmap_gray(thumb_of(self.store.xray_path(x)))
            except Exception:  # noqa: BLE001  (file removed by hand)
                pix = to_pixmap_gray(np.zeros((60, 90), np.uint8))
            items.append((x["id"], pix, f"{'SIDE' if x['view'] == 'side' else 'TOP'} · {when(x['added_at'])}"))
        want = select or (self.current["id"] if self.current and any(x["id"] == self.current["id"] for x in self.xrays)
                          else (self.xrays[0]["id"] if self.xrays else None))
        self.drawer.set_thumbs(items, want)
        if want:
            self.show_xray(want)
        else:
            self.current = None
            self.drawer.canvas.set_image(None)
            self.drawer.canvas.hud = {"tl": [], "tr": [], "bl": []}
            self.drawer.available_heats([])
            self.show_cards()

    def show_xray(self, xray_id):
        self.current = self.store.xray(xray_id)
        self.drawer.mark_current(xray_id)
        x = self.current
        try:
            img = engine.open_xray(self.store.xray_path(x))
            gray = engine.gray_of(img)
        except Exception:  # noqa: BLE001
            self.drawer.canvas.set_image(None)
            self.drawer.canvas.empty_text = "This X-ray file is missing from the data folder"
            self.show_cards()
            return
        c = self.drawer.canvas
        c.set_image(gray)
        self.apply_results()

    def apply_results(self):
        x, c = self.current, self.drawer.canvas
        res, ed = loads(x["results"], {}), loads(x["edits"], {})
        w, h = c.size_
        heats = []
        arch, feet, labels, heel = None, [], [], None
        if res.get("flat") and res["flat"].get("points"):
            pts = ed.get("arch") or res["flat"]["points"]
            arch = {"L10": pts["L10"], "L11": pts["L11"], "left": res["flat"]["left"]}
            c.set_heat("arch", upsample(res["flat"]["cam"], w, h))
            heats.append("arch")
        elif res.get("flat"):
            c.set_heat("arch", upsample(res["flat"]["cam"], w, h))
            heats.append("arch")
        if res.get("spur"):
            s = res["spur"]
            x0, y0, x1, y1 = s["crop_box"]
            c.set_heat("heel", upsample(s["heatmap"], x1 - x0, y1 - y0, 256), tuple(s["crop_box"]))
            heats.append("heel")
            heel = s["box"]
        if res.get("fracture"):
            fr = res["fracture"]
            x0, y0, x1, y1 = fr["map_box"]
            c.set_heat("break", upsample(fr["heatmap"], x1 - x0, y1 - y0), tuple(fr["map_box"]))
            heats.append("break")
        for f in engine.bunion_feet(res, ed):
            feet.append(f["points"])
            labels.append(f"{f['side'].upper()}  HVA {f['hva']:.1f}°  IMA {f['ima']:.1f}°")
        c.set_overlays(arch, feet, labels, heel)
        self.drawer.available_heats(heats)
        self.drawer.reset.setVisible(bool(ed))
        side = (x["side"] or res.get("side") or "").upper()
        c.hud["tl"] = [f"{self.patient['code']}  ·  {self.patient['name'].upper()}",
                       f"{VIEW_NAMES[x['view']].upper()}{('  ·  ' + side) if side else ''}",
                       when(x["added_at"]).upper()]
        c.hud["tr"] = [f"{w} × {h} PX", os.path.basename(x["original_name"] or "")[:28].upper()]
        if x["id"] in self.busy:
            c.hud["bl"] = [("ANALYSING…", "")]
        else:
            c.hud["bl"] = [(t.upper(), col) for t, col in engine.findings(res, ed, self.threshold)]
        c.scanning(x["id"] in self.busy)
        c.update()
        self.show_cards()

    # ---------- cards ----------

    def source_for(self, view):
        if self.current is not None and self.current["view"] == view:
            return self.current
        return report.latest_by_view(self.xrays).get(view)

    def show_cards(self):
        side, top = self.source_for("side"), self.source_for("top")
        note = lambda x: "" if x is self.current else f"From the {VIEW_NAMES[x['view']].lower()} X-ray of {when(x['added_at'])}"  # noqa: E731
        if side is None:
            self.arch_card.show_empty("Add a side view X-ray to measure the arch.")
            self.spur_card.show_empty("Add a side view X-ray to check the heel.")
        elif not side["results"]:
            msg = "Analysing…" if side["id"] in self.busy else "Not checked yet."
            self.arch_card.show_empty(msg)
            self.spur_card.show_empty(msg)
        else:
            res, ed = loads(side["results"], {}), loads(side["edits"], {})
            f = res.get("flat")
            ang = engine.arch_angle(res, ed)
            if f is None:
                self.arch_card.show_empty("The flat foot models aren't loaded.")
            else:
                rows = []
                if ang is not None:
                    t, col = engine.arch_class(ang, self.threshold)
                    cs = [(t, col)] + ([("Points moved by hand", "grey")] if ed.get("arch") else [])
                    rows.append((f"{ang:.1f}°", cs, f"Calcaneal pitch (heel bone angle). Under {self.threshold}° reads as"
                                                    " flat foot, over 30° is flagged as a possible high arch."))
                rows.append((None, [], f"Classifier: {f['p']:.0%} flat foot likelihood. Foot side {f['how']}."))
                self.arch_card.show_rows(rows, note(side))
            s = res.get("spur")
            if s is None:
                self.spur_card.show_empty("The heel spur models aren't loaded.")
            else:
                t, col = engine.spur_class(s)
                pr = s["probs"]
                self.spur_card.show_rows([(f"{pr['spur']:.0%}", [(t, col)],
                                      f"Spur {pr['spur']:.0%} · normal {pr['normal']:.0%} · child's heel (Sever's)"
                                      f" {pr['sever']:.0%}. The amber box is the heel the model looked at.")], note(side))
        if top is None:
            self.bunion_card.show_empty("Add a top-down X-ray to measure bunion angles.")
        elif not top["results"]:
            self.bunion_card.show_empty("Analysing…" if top["id"] in self.busy else "Not checked yet.")
        else:
            feet = engine.bunion_feet(loads(top["results"], {}), loads(top["edits"], {}))
            if not feet:
                self.bunion_card.show_empty("No foot was measured on this X-ray (bunion models not installed?).")
            else:
                rows = []
                for f in feet:
                    t, col = engine.bunion_class(f["hva"], f["ima"])
                    cs = [(f["side"].capitalize(), "grey"), (t, col)] + ([("Points moved by hand", "grey")] if f["edited"] else [])
                    rows.append((None, cs, f"HVA {f['hva']:.1f}°  ·  IMA {f['ima']:.1f}°"))
                rows.append((None, [], "HVA is the big toe angle, IMA the angle between the 1st and 2nd"
                                       " metatarsals. Drag the dots to correct a line."))
                self.bunion_card.show_rows(rows, note(top))
        self.show_fracture_card()
        self.show_xray_card()

    def show_fracture_card(self):
        """Fracture flag of this X-ray (either view), else of the newest checked one."""
        x = self.current if self.current is not None and self.current["results"] else \
            next((x for x in self.xrays if x["results"]), None)
        if x is None:
            busy = self.current is not None and self.current["id"] in self.busy
            self.fracture_card.show_empty("Analysing…" if busy else "Add a foot or ankle X-ray (either view) to check for a fracture.")
            return
        fr = loads(x["results"], {}).get("fracture")
        if fr is None:
            self.fracture_card.show_empty("The fracture model isn't loaded, or this X-ray was checked before it was added."
                                          " Use Check again.")
            return
        t, col = engine.fracture_class(fr)
        src = "" if x is self.current else f"From the {VIEW_NAMES[x['view']].lower()} X-ray of {when(x['added_at'])}"
        self.fracture_card.show_rows([(f"{fr['p']:.0%}", [(t, col)],
                                       f"Fracture likelihood {fr['p']:.0%} (flagged from {fr['threshold']:.0%})."
                                       " The fracture heatmap shows where the model looked. It misses about half of"
                                       " foot and ankle fractures, so no flag is not an all-clear.")], src)

    def show_xray_card(self):
        clear_layout(self.xray_lay)
        x = self.current
        self.xray_lay.addWidget(label("This X-ray", h2=True))
        if x is None:
            self.xray_lay.addWidget(label("No X-rays yet. Add a side view for flat foot and heel spur,"
                                          " or a top-down view for bunion angles.", muted=True, wrap=True))
            self.xray_lay.addWidget(button("+  Add X-ray", "primary", self.add_xray))
            return
        t, col = STATUS_CHIPS["new" if x["id"] in self.busy else x["status"]]
        if x["id"] in self.busy:
            t = "Analysing…"
        cs = [(VIEW_NAMES[x["view"]], "grey"), (t, col)]
        self.xray_lay.addWidget(chips(cs))
        added = self.store.user(x["added_by"])
        self.xray_lay.addWidget(label(f"{x['original_name']} · added {when(x['added_at'])}"
                                      f"{(' by ' + added['name']) if added else ''}", muted=True, wrap=True))
        if x["signed_by"]:
            s = self.store.user(x["signed_by"])
            self.xray_lay.addWidget(label(f"Signed off by {s['name'] if s else 'a doctor'} {when(x['signed_at'])}",
                                          muted=True))
        self.note = QPlainTextEdit(x["note"] or "")
        self.note.setPlaceholderText("Note on this X-ray")
        self.note.setFixedHeight(80)
        self.xray_lay.addWidget(self.note)
        row = QHBoxLayout()
        row.addWidget(button("Save note", "ghost", self.save_note))
        again = button("Check again", "ghost", lambda: self.run(x["id"]))
        again.setEnabled(x["id"] not in self.busy)
        row.addWidget(again)
        self.xray_lay.addLayout(row)

    # ---------- actions ----------

    def run(self, xray_id):
        self.busy.add(xray_id)
        if self.current and self.current["id"] == xray_id:
            self.apply_results()
        self.analyse.emit(xray_id)

    def analysis_done(self, xray_id):
        self.busy.discard(xray_id)
        if self.patient and any(x["id"] == xray_id for x in self.store.xrays(self.patient["id"])):
            self.xrays = self.store.xrays(self.patient["id"])
            if self.current and self.current["id"] == xray_id:
                self.current = self.store.xray(xray_id)
                self.apply_results()
            else:
                self.show_cards()
        self.changed.emit()

    def analysis_failed(self, xray_id, msg):
        self.busy.discard(xray_id)
        if self.current and self.current["id"] == xray_id:
            self.apply_results()
        QMessageBox.warning(self, "Check failed", f"The models couldn't read this X-ray.\n\n{msg}")

    def add_xray(self, path=None):
        has_finder = bool(self.engine and self.engine.bunion and self.engine.bunion.finder)
        d = AddXrayDialog(self, self.engine is not None, has_finder, path)
        if not d.exec() or not d.choice:
            return
        src, view, side = d.choice
        x = self.store.add_xray(self.user["id"], self.patient["id"], src, view, side or "")
        self.current = x
        self.busy.add(x["id"])
        self.reload(select=x["id"])
        self.analyse.emit(x["id"])
        self.changed.emit()

    def points_edited(self, kind, index, pts):
        x = self.current
        ed = loads(x["edits"], {})
        if kind == "arch":
            ed["arch"] = {k: [float(v[0]), float(v[1])] for k, v in pts.items()}
        else:
            ed.setdefault("bunion", {})[str(index)] = [[float(a), float(b)] for a, b in pts]
        self.store.save_edits(self.user["id"], x["id"], ed)
        self.current = self.store.xray(x["id"])
        self.xrays = self.store.xrays(self.patient["id"])
        self.apply_results()

    def reset_points(self):
        if self.current:
            self.store.save_edits(self.user["id"], self.current["id"], {})
            self.current = self.store.xray(self.current["id"])
            self.xrays = self.store.xrays(self.patient["id"])
            self.apply_results()

    def save_note(self):
        if self.current:
            self.store.save_note(self.user["id"], self.current["id"], self.note.toPlainText().strip())
            self.current = self.store.xray(self.current["id"])

    def toggle_flag(self):
        self.patient = self.store.update_patient(self.user["id"], self.patient["id"], flagged=0 if self.patient["flagged"] else 1)
        self.show_header()
        self.changed.emit()

    def edit_patient(self):
        d = PatientDialog(self, self.store, self.user, self.patient)
        if d.exec():
            self.patient = d.result_row
            self.show_header()
            if self.current:
                self.apply_results()
            self.changed.emit()

    def open_report(self):
        self.xrays = self.store.xrays(self.patient["id"])
        text = report.draft(self.patient, self.xrays, self.user, self.threshold)
        ids = [x["id"] for x in report.latest_by_view(self.xrays).values() if x["status"] != "done"]
        d = ReportDialog(self, self.store, self.user, self.patient, text, ids)
        d.exec()
        if d.signed:
            self.reload()
            self.changed.emit()
