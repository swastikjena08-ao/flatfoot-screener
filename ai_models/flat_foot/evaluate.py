"""Scores the heel bone angle on Pesplanus X-rays from patients never used for labels.

The split CSV (one row per X-ray: file, folder, group, ...) marks which patients had their
points hand-labelled; only group == never_labeled_patient is scored, so the result is on
X-rays the landmark model never saw. The X-rays and the CSV are not in this repo.

  python evaluate.py --data "D:/Pesplanus Dataset" --split pesplanus_split.csv
  (--data holds the pesplanus/ and notpesplanus/ folders)

Result on 2026-10-04: 198/211 = 93.8% (flat 190/202, normal 8/9).
"""
import argparse
import csv
import json
import os

from PIL import Image

from predict import FlatFoot

FOLDERS = {"flat": "pesplanus", "normal": "notpesplanus"}


def find(folder, name):
    """File names differ in case between copies of the dataset (1R.JPG / 1R.jpg)."""
    for f in os.listdir(folder):
        if f.lower() == name.lower():
            return os.path.join(folder, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--models", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--group", default="never_labeled_patient")
    ap.add_argument("--out", default=None, help="optional JSON file for the scores")
    args = ap.parse_args()

    ff = FlatFoot(args.models)
    rows = [r for r in csv.DictReader(open(args.split, encoding="utf-8")) if r["group"] == args.group]
    hits = {"flat": [0, 0], "normal": [0, 0]}
    missing = 0
    for r in rows:
        path = find(os.path.join(args.data, FOLDERS[r["folder"]]), r["file"])
        if path is None:
            missing += 1
            continue
        res = ff.predict(Image.open(path))
        hits[r["folder"]][0] += res["flat_by_angle"] == (r["folder"] == "flat")
        hits[r["folder"]][1] += 1
    right = sum(h[0] for h in hits.values())
    total = sum(h[1] for h in hits.values())
    scores = {"group": args.group, "accuracy": right / max(total, 1), "right": right, "total": total,
              "flat_caught": hits["flat"], "normal_cleared": hits["normal"], "missing_files": missing}
    print(json.dumps(scores, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
