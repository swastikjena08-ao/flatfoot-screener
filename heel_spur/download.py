"""Download the Kaggle heel dataset (no Kaggle login needed) and unzip it.

Usage:
  python download.py --out D:/heel-spur
"""
import argparse
import os
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/osamahtaher/heel-dataset"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = os.path.join(a.out, "heel-dataset.zip")
    if not os.path.exists(z):
        print("downloading ~160 MB ...")
        urllib.request.urlretrieve(URL, z)
    zipfile.ZipFile(z).extractall(a.out)
    print("done:", a.out)
