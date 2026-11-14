"""Turns the per-image CSVs from full_test.py into scores (summary.json).

Every score is given separately for X-rays the model learned from and X-rays it never saw.
Only the never-seen numbers are honest accuracy; the learned-from numbers show how well the
model remembers its training X-rays.

  python summarise.py --runs <full_test.py output folder> [--out summary.json]
"""
import argparse
import json
import os

import pandas as pd

VIEWS = ["side", "top_down", "oblique", "not_foot"]
GRADES = ["No bunion", "Mild bunion", "Moderate bunion", "Severe bunion"]


def grade(hva, ima):
    """Same grading as remi_ai/engine.py: the worse of HVA and IMA."""
    h = 0 if hva < 15 else 1 if hva <= 20 else 2 if hva <= 40 else 3
    i = 0 if ima < 9 else 1 if ima <= 11 else 2 if ima <= 16 else 3
    return max(h, i)


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def routing_scores(df):
    """Did the X-ray check send each X-ray to the right checks?
    right: correct view; asked: unsure, so the app asks the doctor (no wrong check runs);
    wrong: another view, so the wrong checks would run (or none)."""
    out = {}
    df = df[df.error.isna()] if "error" in df else df
    groups = {"learned_from": df[df.split == "train"], "tuning": df[df.split == "val"],
              "never_seen": df[df.split == "test"], "new_hospital": df[df.split == "outside_test"]}
    for g, d in groups.items():
        per_view = {}
        for v in VIEWS:
            dv = d[d.true_view == v]
            if not len(dv):
                continue
            per_view[v] = {"n": len(dv), "right": pct((dv.pred_view == v).sum(), len(dv)),
                           "asked_doctor": pct((dv.pred_view == "unsure").sum(), len(dv)),
                           "wrong": pct(((dv.pred_view != v) & (dv.pred_view != "unsure")).sum(), len(dv)),
                           "mistaken_for": {k: int(c) for k, c in dv[(dv.pred_view != v)].pred_view.value_counts().items()}}
        right = (d.pred_view == d.true_view).sum()
        asked = (d.pred_view == "unsure").sum()
        # The mistake m cares about: a foot check running on an X-ray it doesn't fit.
        wrong_check = d[(d.pred_view.isin(["side", "top_down"])) & (d.pred_view != d.true_view)]
        out[g] = {"n": len(d), "right": pct(right, len(d)), "asked_doctor": pct(asked, len(d)),
                  "wrong": pct(len(d) - right - asked, len(d)),
                  "wrong_check_ran": pct(len(wrong_check), len(d)), "wrong_check_ran_n": len(wrong_check),
                  "per_view": per_view}
    return out


def route_ok(d, view):
    return d.pred_view.isin([view, "unsure"])


def flat_foot_scores(df):
    out = {}
    names = {"labeled": "learned_from", "other_foot_of_labeled_patient": "other_foot_of_learned_patient",
             "never_labeled_patient": "never_seen"}
    for g, label in names.items():
        d = df[df.group == g]
        measured = d[d.ff_category.isin(["Flat foot", "Normal arch", "High arch"])]
        said_flat = measured.ff_category == "Flat foot"
        is_flat = measured.truth == "flat"
        right = (said_flat == is_flat)
        e2e = right & route_ok(measured, "side")
        out[label] = {
            "n": len(d), "measured": len(measured), "not_measured": len(d) - len(measured),
            "accuracy": pct(right.sum(), len(measured)), "right": int(right.sum()),
            "flat_caught": [int((said_flat & is_flat).sum()), int(is_flat.sum())],
            "normal_cleared": [int((~said_flat & ~is_flat).sum()), int((~is_flat).sum())],
            "end_to_end_right": pct(e2e.sum(), len(d)),
            "routed_right_or_asked": pct(route_ok(d, "side").sum(), len(d)),
        }
    out["heel_spur_found_on_flat_foot_xrays"] = {k: int(v) for k, v in df.hs_label.value_counts().items()}
    return out


def heel_spur_scores(df):
    out = {}
    for split, label in (("train", "learned_from"), ("val", "tuning"), ("test", "never_seen")):
        d = df[df.split == split]
        ok = d.hs_label == d.truth
        spur_yes = d.hs_label != "normal"
        truth_yes = d.truth != "normal"
        conf = pd.crosstab(d.truth, d.hs_label)
        out[label] = {
            "n": len(d), "accuracy_3_way": pct(ok.sum(), len(d)), "right": int(ok.sum()),
            "problem_vs_no_problem": pct((spur_yes == truth_yes).sum(), len(d)),
            "per_class": {c: [int(((d.truth == c) & ok).sum()), int((d.truth == c).sum())]
                          for c in ("normal", "spur", "sever")},
            "confusion": {t: {p: int(conf.loc[t, p]) for p in conf.columns} for t in conf.index},
            "end_to_end_right": pct((ok & route_ok(d, "side")).sum(), len(d)),
            "routed_right_or_asked": pct(route_ok(d, "side").sum(), len(d)),
        }
    out["flat_foot_answer_on_heel_spur_xrays"] = {k: int(v) for k, v in df.ff_category.value_counts().items()}
    return out


def bunion_scores(df):
    rows = []
    for _, r in df.iterrows():
        truth = json.loads(r.truth_feet)
        try:
            pred = json.loads(r.bn_feet) if isinstance(r.bn_feet, str) and r.bn_feet.startswith("[") else []
        except ValueError:
            pred = []
        for t in truth:
            p = next((f for f in pred if f["side"] == t["side"]), None)
            row = {"file": r.file, "split": r.split, "side": t["side"], "found": p is not None,
                   "route_ok": r.pred_view in ("top_down", "unsure")}
            if p:
                row.update(hva_err=abs(p["hva"] - t["hva"]), ima_err=abs(p["ima"] - t["ima"]),
                           grade_true=grade(t["hva"], t["ima"]), grade_pred=GRADES.index(p["grade"]))
            rows.append(row)
        # feet the app reported that the doctors did not mark (e.g. a second foot)
    feet = pd.DataFrame(rows)
    out = {}
    for split, label in (("train", "learned_from"), ("val", "tuning"), ("test", "never_seen"),
                         ("none", "never_seen_extra")):
        d = feet[feet.split == split]
        if not len(d):
            continue
        f = d[d.found]
        ok3 = (f.hva_err <= 3) & (f.ima_err <= 3)
        out[label] = {
            "images": int(d.file.nunique()), "feet": len(d), "feet_found": len(f),
            "hva_within_3": pct((f.hva_err <= 3).sum(), len(d)), "ima_within_3": pct((f.ima_err <= 3).sum(), len(d)),
            "both_within_3": pct(ok3.sum(), len(d)),
            "hva_avg_error": round(f.hva_err.mean(), 2), "ima_avg_error": round(f.ima_err.mean(), 2),
            "same_grade": pct((f.grade_true == f.grade_pred).sum(), len(d)),
            "grade_within_one": pct(((f.grade_true - f.grade_pred).abs() <= 1).sum(), len(d)),
            "end_to_end_both_within_3": pct((ok3 & f.route_ok).sum(), len(d)),
            "routed_right_or_asked": pct(df[df.split == split].pred_view.isin(["top_down", "unsure"]).sum(),
                                         (df.split == split).sum()),
        }
    return out


def speed(dfs):
    s = pd.concat([d[["seconds"]] for d in dfs])
    return {"median_seconds_per_xray": round(s.seconds.median(), 2),
            "p90_seconds_per_xray": round(s.seconds.quantile(0.9), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    load = lambda k: pd.read_csv(os.path.join(a.runs, f"{k}.csv"))
    r, ff, hs, bn = load("routing"), load("flat_foot"), load("heel_spur"), load("bunion")
    summary = {
        "images_run": {"routing": len(r), "flat_foot": len(ff), "heel_spur": len(hs), "bunion": len(bn)},
        "errors": {k: int(d.error.notna().sum()) if "error" in d else 0
                   for k, d in (("routing", r), ("flat_foot", ff), ("heel_spur", hs), ("bunion", bn))},
        "routing": routing_scores(r),
        "flat_foot": flat_foot_scores(ff),
        "heel_spur": heel_spur_scores(hs),
        "bunion": bunion_scores(bn),
        "speed_full_checks": speed([ff, hs, bn]),
    }
    text = json.dumps(summary, indent=1)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
