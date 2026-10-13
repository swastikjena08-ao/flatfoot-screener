"""Patient report and referral letter, written as HTML files in the patient's folder.

They open in the default browser, where the doctor can print them or save them as PDF.
"""
import base64
import datetime as dt
import html
import io
import os

from PIL import Image, ImageOps

from .engine import CHECKS, VIEWS, findings

CSS = """
body { font-family: "Segoe UI", system-ui, sans-serif; color: #17181a; max-width: 820px; margin: 32px auto;
       padding: 0 24px; line-height: 1.5; font-size: 14px; }
h1 { font-size: 24px; font-weight: 600; margin: 0; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: #6b6d72; margin: 28px 0 8px;
     border-bottom: 1px solid #e3e3e0; padding-bottom: 6px; font-weight: 500; }
.sub { color: #6b6d72; }
table { border-collapse: collapse; width: 100%; }
td, th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #ececea; vertical-align: top; }
th { color: #6b6d72; font-weight: 500; width: 170px; }
.xray { display: grid; grid-template-columns: 220px 1fr; gap: 16px; margin: 14px 0; page-break-inside: avoid; }
.xray img { width: 220px; border-radius: 6px; background: #000; }
.alert { color: #c4282d; font-weight: 600; }
.warn { color: #a5620b; font-weight: 600; }
.ok { color: #1f7a4d; font-weight: 600; }
.small { font-size: 12px; color: #6b6d72; }
.letter p { margin: 0 0 12px; }
"""


def esc(s):
    return html.escape(str(s if s is not None else ""))


def nice_date(iso):
    try:
        return dt.date.fromisoformat(iso[:10]).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return iso or ""


def thumb(path, size=440):
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("L")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _page(title, body):
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def _patient_table(p, hist):
    rows = [("Patient ID", p["patient_id"]), ("Name", p["name"]),
            ("Age / sex", " · ".join(str(v) for v in (p["age"], p["sex"]) if v not in (None, ""))),
            ("Phone", p["phone"]), ("Referred by", p["referred_by"]), ("Main complaint", p["complaint"]),
            ("First seen", nice_date(p["created"]))]
    rows += list(hist.items())
    return "<table>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v) or '—'}</td></tr>" for k, v in rows) + "</table>"


def _xray_block(x):
    items = findings(x["results"])
    lines = "".join(f"<div><span class='{sev}'>{esc(t)}</span> · {esc(d)}</div>" for t, d, sev in items)
    if not items:
        lines = "<div class='sub'>No checks ran on this X-ray.</div>"
    skipped = "".join(f"<div class='small'>{esc(CHECKS[k]['name'])}: {esc(v)}</div>"
                      for k, v in x["results"].get("skipped", {}).items())
    scores = "".join(f"<div class='small'>{esc(CHECKS[k]['name'])} model in testing: {esc(CHECKS[k]['score'])}</div>"
                     for k in x["results"].get("checks", {}))
    src = thumb(x["path"])
    reviewed = "Reviewed by the doctor" if x["reviewed"] else "Not yet reviewed"
    return (f"<div class='xray'><img src='{src}' alt='X-ray'><div><b>{esc(VIEWS.get(x['view'], x['view']))}</b>"
            f" <span class='small'>· {esc(reviewed)}</span>{lines}{skipped}{scores}</div></div>")


def write_report(folder, p, hist, visits, doctor):
    body = [f"<h1>{esc(p['name'])}</h1><div class='sub'>Remi AI foot X-ray report · "
            f"{dt.date.today().strftime('%d %b %Y')} · {esc(doctor['name'])} ({esc(doctor['doctor_id'])})</div>",
            "<h2>Patient and medical history</h2>", _patient_table(p, hist), "<h2>Visits</h2>"]
    for v in visits:
        body.append(f"<h3>{esc(nice_date(v['date']))} · {esc(v['label'])}</h3>")
        if v["note"]:
            body.append("<p>" + esc(v["note"]).replace("\n", "<br>") + "</p>")
        body += [_xray_block(x) for x in v["xrays"]]
    body.append("<p class='small'>Remi AI measurements are suggestions to support the doctor's review. "
                "They are not a diagnosis.</p>")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"report_{dt.datetime.now():%Y%m%d-%H%M%S}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"Report · {p['name']}", "".join(body)))
    return path


def write_referral(folder, p, hist, visits, doctor, to, reason):
    latest = next((v for v in visits if v["xrays"]), None)
    found = []
    if latest:
        for x in latest["xrays"]:
            found += [f"{t} ({d})" for t, d, _ in findings(x["results"])]
    body = [f"<div class='letter'><h1>Referral</h1><div class='sub'>{dt.date.today().strftime('%d %b %Y')}</div>",
            f"<h2>To</h2><p>{esc(to) or 'Specialist'}</p>",
            f"<h2>Patient</h2>{_patient_table(p, hist)}",
            f"<h2>Reason</h2><p>{esc(reason) or 'For further assessment.'}</p>"]
    if found:
        body.append("<h2>X-ray findings</h2><ul>" + "".join(f"<li>{esc(f)}</li>" for f in found) + "</ul>")
    if latest:
        body += [_xray_block(x) for x in latest["xrays"]]
    body.append(f"<p>Kind regards,<br>{esc(doctor['name'])}<br><span class='small'>{esc(doctor.get('clinic', ''))}"
                f" · Doctor ID {esc(doctor['doctor_id'])}</span></p></div>")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"referral_{dt.datetime.now():%Y%m%d-%H%M%S}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"Referral · {p['name']}", "".join(body)))
    return path
