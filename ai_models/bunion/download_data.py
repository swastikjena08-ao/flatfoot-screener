"""Download the HVAngleEst dataset (version V5) from scidb.cn, file by file.

The site has no single "download all" link, so this asks its file-list API for
every file URL and fetches them in parallel. About 5,600 files, 215 MB.
License CC BY-NC-ND 4.0: use it, but don't re-share it or modified copies.

Usage: python download_data.py --out data/hvangleest
"""
import argparse
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATASET_ID = "85e73c3b2f9141138d9885732030f624"
LIST_URL = f"https://www.scidb.cn/api/sdb-filetree-service/getAllUrl?dataSetId={DATASET_ID}&version=V5"


def fetch(url, dest, tries=8):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url.replace(" ", "%20"), timeout=300) as r, open(dest + ".part", "wb") as f:
                f.write(r.read())
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:
            # the server answers 429 when hit too hard; back off longer then
            time.sleep((30 if getattr(e, "code", None) == 429 else 2) * (i + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hvangleest")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    with urllib.request.urlopen(LIST_URL, timeout=120) as r:
        urls = [u.strip() for u in r.read().decode().splitlines() if u.strip()]
    jobs = []
    for u in urls:
        path = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)["path"][0]
        jobs.append((u, os.path.join(a.out, path.replace("/V5/", "", 1))))
    print(len(jobs), "files")
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(lambda j: fetch(*j), jobs))
    failed = [d for (u, d), o in zip(jobs, ok) if not o]
    print("done,", len(failed), "failed", failed[:5])


if __name__ == "__main__":
    main()
