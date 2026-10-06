"""Make one-time activation keys for new doctor accounts.

  python tools/make_keys.py 10 --out keys.txt

Prints the new keys and writes them to --out (keep that file private: hand each doctor
one key). Their SHA-256 hashes are added to assistant/activation_keys.json, which ships
with the app; the app never stores the keys themselves. Rebuild the .exe after adding keys.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assistant import auth  # noqa: E402

HASHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assistant", "activation_keys.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=10)
    ap.add_argument("--out", required=True, help="text file for the plain keys (never commit it)")
    ap.add_argument("--hashes", default=HASHES)
    a = ap.parse_args()
    data = {"sha256": []}
    if os.path.exists(a.hashes):
        with open(a.hashes, encoding="utf-8") as f:
            data = json.load(f)
    keys = [auth.new_key() for _ in range(a.n)]
    data["sha256"] = sorted(set(data["sha256"]) | {auth.key_hash(k) for k in keys})
    data["note"] = "SHA-256 of each activation key (normalised, as auth.normalise_key). Made with tools/make_keys.py."
    with open(a.hashes, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write("\n".join(keys) + "\n")
    print("\n".join(keys))
    print(f"{len(keys)} keys added; {len(data['sha256'])} valid keys in {a.hashes}")


if __name__ == "__main__":
    main()
