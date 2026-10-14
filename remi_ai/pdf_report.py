"""Signed patient report as a PDF, saved in the patient's folder.

Each X-ray in the visit is shown with Remi's measuring lines drawn on it, followed by the
findings, the doctor's notes and who signed the visit off (or that it is still a draft).
"""
import datetime as dt
import io
import os

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageOps

from .engine import CHECKS, VIEWS, findings

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "web", "img", "remi_wordmark.png")

INK, MUTED, LINE = (23, 24, 26), (107, 109, 114), (227, 227, 224)
SEV = {"alert": (196, 40, 45), "warn": (165, 98, 11), "ok": (31, 122, 77)}
# Same colours as the measuring lines in the app.
TOE, MT1, MT2, ARCH, SPUR = (245, 181, 68), (46, 196, 182), (122, 167, 255), (46, 196, 182), (229, 72, 77)

# Fonts with full Unicode (°, ·, —). Windows has Segoe UI; elsewhere fall back to DejaVu, then Helvetica.
FONT_CHOICES = [
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def nice_date(iso):
    try:
        return dt.date.fromisoformat(iso[:10]).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return iso or ""


def annotated(path, results, size=900):
    """The X-ray with Remi's lines drawn on it, as an RGB image (longest side `size`)."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode.startswith("I") or img.mode == "F":
        import numpy as np
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))
    img = img.convert("RGB")
    w0, h0 = img.size
    img.thumbnail((size, size))
    s = img.width / w0
    d = ImageDraw.Draw(img)
    lw = max(2, round(img.width / 300))
    c = results.get("checks", {})

    def P(q):
        return (q[0] * s, q[1] * s)

    def dot(q, col):
        x, y = P(q)
        d.ellipse((x - lw * 1.8, y - lw * 1.8, x + lw * 1.8, y + lw * 1.8), fill=col)

    def seg(a, b, col, k=1.15):
        ax, ay = P(a)
        bx, by = P(b)
        sx, sy = bx + (ax - bx) * k, by + (ay - by) * k
        ex, ey = ax + (bx - ax) * k, ay + (by - ay) * k
        d.line((sx, sy, ex, ey), fill=col, width=lw)

    h = c.get("heel_spur")
    if h:
        a, b, cc, dd = [v * s for v in h["box"]]
        d.ellipse((a, b, cc, dd), outline=SPUR if h["label"] == "spur" else (200, 200, 200), width=lw)
    f = c.get("flat_foot")
    if f and f.get("category") != "Not measured" and f.get("points", {}).get("L10"):
        p10, p11 = f["points"]["L10"], f["points"]["L11"]
        seg(p10, p11, ARCH, 1.25)
        dot(p10, ARCH)
        dot(p11, ARCH)
    for foot in (c.get("bunion") or {}).get("feet", []):
        p = foot["points"]
        for i, col in ((0, TOE), (2, MT1), (4, MT2)):
            seg(p[i], p[i + 1], col)
            dot(p[i], col)
            dot(p[i + 1], col)
    return img


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(18, 16, 18)
        self.set_auto_page_break(True, 18)
        self.font = "helvetica"
        for regular, bold in FONT_CHOICES:
            if os.path.exists(regular) and os.path.exists(bold):
                self.add_font("body", "", regular)
                self.add_font("body", "B", bold)
                self.font = "body"
                break

    def text_safe(self, s):
        s = str(s if s is not None else "")
        if self.font == "helvetica":  # core fonts are Latin-1 only
            s = s.replace("—", "-").replace("–", "-").replace("’", "'")
            s = s.encode("latin-1", "replace").decode("latin-1")
        return s

    def use(self, size, bold=False, color=INK):
        self.set_font(self.font, "B" if bold else "", size)
        self.set_text_color(*color)

    def line_text(self, text, size=10, bold=False, color=INK, h=5.2):
        self.use(size, bold, color)
        self.multi_cell(0, h, self.text_safe(text), new_x="LMARGIN", new_y="NEXT")

    def heading(self, text):
        self.ln(4)
        self.use(9, True, MUTED)
        self.cell(0, 6, self.text_safe(text.upper()), new_x="LMARGIN", new_y="NEXT")
        y = self.get_y()
        self.set_draw_color(*LINE)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(2.5)

    def table(self, rows, key_w=48):
        for k, v in rows:
            y0 = self.get_y()
            self.use(9.5, False, MUTED)
            self.multi_cell(key_w, 5.6, self.text_safe(k), new_x="RIGHT", new_y="TOP")
            self.use(9.5)
            self.multi_cell(0, 5.6, self.text_safe(v or "—"), new_x="LMARGIN", new_y="NEXT")
            self.set_y(max(self.get_y(), y0 + 5.6))

    def footer(self):
        self.set_y(-12)
        self.use(7.5, False, MUTED)
        self.cell(0, 5, self.text_safe("Remi AI measurements support the doctor's review. They are not a diagnosis."),
                  align="L")
        self.cell(0, 5, f"{self.page_no()}", align="R")


def write_pdf(folder, p, hist, visit, xrays, doctor, signer=None, previous=()):
    """Write the report for one visit. `signer` is the doctor who signed it off, or None for a draft."""
    pdf = Report()
    pdf.add_page()
    if os.path.exists(LOGO):
        pdf.image(LOGO, x=pdf.l_margin, y=14, h=11)
    pdf.set_xy(pdf.l_margin, 14)
    pdf.use(9, False, MUTED)
    status = "SIGNED OFF" if signer else "DRAFT - NOT SIGNED OFF"
    pdf.cell(0, 5, pdf.text_safe(status), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, pdf.text_safe(dt.date.today().strftime("%d %b %Y")), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(32)

    pdf.line_text("Foot X-ray report", 18, True, h=8)
    pdf.line_text(f"{p['name']} · {p['patient_id']} · visit of {nice_date(visit['date'])}", 10, color=MUTED)

    pdf.heading("Patient")
    rows = [("Patient ID", p["patient_id"]), ("Name", p["name"]),
            ("Age / sex", " · ".join(str(v) for v in (p.get("age"), p.get("sex")) if v not in (None, ""))),
            ("Phone", p.get("phone")), ("Referred by", p.get("referred_by")),
            ("Main complaint", p.get("complaint")), ("First seen", nice_date(p.get("created")))]
    pdf.table(rows)
    if hist:
        pdf.heading("Medical history")
        pdf.table(list(hist.items()))

    pdf.heading(f"X-rays · {visit.get('label') or 'Visit'}")
    if not xrays:
        pdf.line_text("No X-rays in this visit.", 10, color=MUTED)
    width = pdf.w - pdf.l_margin - pdf.r_margin
    img_w = 62
    for x in xrays:
        img = annotated(x["path"], x["results"])
        img_h = img_w * img.height / img.width
        if pdf.get_y() + min(img_h, 90) > pdf.h - 24:
            pdf.add_page()
        y0 = pdf.get_y()
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        buf.seek(0)
        shown_h = min(img_h, 90)
        pdf.image(buf, x=pdf.l_margin, y=y0, w=img_w * shown_h / img_h, h=shown_h)
        left = pdf.l_margin + img_w + 6
        pdf.set_xy(left, y0)
        pdf.use(10.5, True)
        pdf.multi_cell(width - img_w - 6, 5.6, pdf.text_safe(VIEWS.get(x["view"], x["view"])),
                       new_x="LEFT", new_y="NEXT")
        items = findings(x["results"])
        if not items:
            pdf.use(9.5, False, MUTED)
            pdf.multi_cell(width - img_w - 6, 5.2, "No checks ran on this X-ray.", new_x="LEFT", new_y="NEXT")
        for title, detail, sev in items:
            pdf.use(10, True, MUTED if title == "Not measured" else SEV.get(sev, INK))
            pdf.multi_cell(width - img_w - 6, 5.4, pdf.text_safe(title), new_x="LEFT", new_y="NEXT")
            pdf.use(9.5, False, MUTED)
            pdf.multi_cell(width - img_w - 6, 5.0, pdf.text_safe(detail), new_x="LEFT", new_y="NEXT")
        edited = [k for k, v in x["results"].get("checks", {}).items()
                  if v.get("edited") or any(f.get("edited") for f in v.get("feet", []))]
        notes = []
        if edited:
            notes.append("Measuring points adjusted by the doctor: " + ", ".join(CHECKS[k]["name"] for k in edited) + ".")
        for k, why in x["results"].get("skipped", {}).items():
            notes.append(f"{CHECKS[k]['name']}: {why.replace('Not run: ', 'not run, ')}.")
        notes.append("Reviewed by the doctor." if x["reviewed"] else "Not marked as reviewed.")
        pdf.set_xy(left, pdf.get_y() + 1)
        pdf.use(8.5, False, MUTED)
        pdf.multi_cell(width - img_w - 6, 4.4, pdf.text_safe(" ".join(notes)), align="L", new_x="LEFT", new_y="NEXT")
        pdf.set_y(max(pdf.get_y(), y0 + shown_h) + 5)

    if visit.get("note"):
        pdf.heading("Doctor's notes")
        pdf.line_text(visit["note"], 10)

    prev = [v for v in previous if v["xrays"]]
    if prev:
        pdf.heading("Earlier visits")
        for v in prev[:5]:
            found = "; ".join(f"{t} ({d})" for x in v["xrays"] for t, d, _ in findings(x["results"])) or "No findings"
            pdf.line_text(f"{nice_date(v['date'])}: {found}", 9.5)

    pdf.heading("Sign-off")
    if signer:
        pdf.line_text(f"Signed off by {signer['name']} ({signer['doctor_id']})"
                      f"{', ' + signer['clinic'] if signer.get('clinic') else ''}", 10.5, True)
        when = visit.get("signed_at") or dt.datetime.now().isoformat(timespec="seconds")
        pdf.line_text(dt.datetime.fromisoformat(when).strftime("%d %b %Y, %H:%M"), 9.5, color=MUTED)
    else:
        pdf.line_text(f"Draft prepared by {doctor['name']} ({doctor['doctor_id']}). "
                      "Not valid until a doctor signs it off.", 10.5, True, color=SEV["warn"])

    os.makedirs(folder, exist_ok=True)
    kind = "report" if signer else "draft_report"
    name = f"{kind}_{visit['date']}_{dt.datetime.now():%H%M%S}.pdf"
    path = os.path.join(folder, name)
    pdf.output(path)
    return path
