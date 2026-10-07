"""Draft report text from a patient's latest results. Plain text so the doctor can edit it."""
import datetime as dt

from . import engine
from .store import loads


def _date(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%d %b %Y %H:%M") if ts else ""


def latest_by_view(xrays):
    """Newest analysed X-ray of each view."""
    out = {}
    for x in xrays:
        if x["results"] and x["view"] not in out:
            out[x["view"]] = x
    return out


def draft(patient, xrays, author, threshold=17.7):
    from .pages import age_of
    lines = ["FOOT X-RAY REPORT (DRAFT)", ""]
    facts = [f"Patient: {patient['name']}", f"Patient ID: {patient['code']}"]
    a = age_of(patient["birth_date"])
    if a is not None:
        facts.append(f"Age: {a}")
    if patient["sex"]:
        facts.append(f"Sex: {patient['sex']}")
    lines += facts + [f"Prepared by: {author['name']}", f"Date: {dt.date.today():%d %b %Y}", ""]
    picked = latest_by_view(xrays)
    if not picked:
        lines.append("No analysed X-rays yet.")
        return "\n".join(lines)
    impression = []
    side = picked.get("side")
    if side:
        res, ed = loads(side["results"], {}), loads(side["edits"], {})
        foot = (side["side"] or res.get("side") or "").capitalize()
        lines += [f"SIDE VIEW ({foot} foot), taken into the app {_date(side['added_at'])}", ""]
        ang = engine.arch_angle(res, ed)
        if ang is not None:
            t, _ = engine.arch_class(ang, threshold)
            moved = " (points adjusted by hand)" if ed.get("arch") else ""
            lines.append(f"Arch: calcaneal pitch {ang:.1f}°{moved}. {t}. Flat foot cut-off {threshold}°.")
            if t != "Normal arch":
                impression.append(f"{t.lower()} ({ang:.1f}°)")
        spur = res.get("spur")
        if spur:
            t, _ = engine.spur_class(spur)
            pr = spur["probs"]
            lines.append(f"Heel: {t}. Model scores: spur {pr['spur']:.0%}, normal {pr['normal']:.0%},"
                         f" child's heel (Sever's) {pr['sever']:.0%}.")
            if t != "No heel spur":
                impression.append(t.lower())
        lines.append("")
    top = picked.get("top")
    if top:
        res, ed = loads(top["results"], {}), loads(top["edits"], {})
        lines += [f"TOP-DOWN VIEW, taken into the app {_date(top['added_at'])}", ""]
        feet = engine.bunion_feet(res, ed)
        if not feet:
            lines.append("Bunion angles: no foot measured.")
        for f in feet:
            t, _ = engine.bunion_class(f["hva"], f["ima"])
            moved = " (points adjusted by hand)" if f["edited"] else ""
            lines.append(f"{f['side'].capitalize()} foot: hallux valgus angle {f['hva']:.1f}°, "
                         f"intermetatarsal angle {f['ima']:.1f}°{moved}. {t}.")
            if t != "No bunion":
                impression.append(f"{t.lower()} on the {f['side']} (HVA {f['hva']:.0f}°, IMA {f['ima']:.0f}°)")
        lines.append("")
    lines.append("IMPRESSION")
    lines.append(("Findings: " + "; ".join(impression) + ".") if impression else "No flagged findings on the measured views.")
    lines += ["", "Measurements are drafted by software and must be checked and signed by the treating doctor."]
    return "\n".join(lines)
