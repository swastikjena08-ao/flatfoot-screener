"""Foot X-ray assistant: doctor accounts, patient profiles and the three foot X-ray models.

  python foot_assistant.py [--data-dir FOLDER]

Patients, X-ray copies and results are kept in FOLDER (default: %APPDATA%\\Foot X-ray Assistant).
The single-X-ray screener is still flatfoot_app.py.
"""
import argparse
import sys

from assistant.app import run


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", help="where patients and X-rays are stored")
    ap.add_argument("--keys", help="activation key hashes (default: the bundled activation_keys.json)")
    ap.add_argument("--bunion-models", help="folder with the bunion .onnx files (default: models/bunion)")
    a, rest = ap.parse_known_args()
    sys.exit(run([sys.argv[0], *rest], a.data_dir, a.keys, a.bunion_models))


if __name__ == "__main__":
    main()
