"""Download and unzip FracAtlas (figshare article 22363012, CC BY 4.0, 338 MB).

Usage:
  python download.py --out D:/fracatlas      # -> D:/fracatlas/FracAtlas/dataset.csv ...
"""
import argparse
import os
import urllib.request
import zipfile

URL = "https://ndownloader.figshare.com/files/65518038"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = os.path.join(a.out, "FracAtlas.zip")
    if not os.path.exists(z):
        urllib.request.urlretrieve(URL, z)
    zipfile.ZipFile(z).extractall(a.out)
    os.remove(z)
    print(os.path.join(a.out, "FracAtlas"))


if __name__ == "__main__":
    main()
