"""Passwords and activation keys. Standard library only.

Passwords are stored as PBKDF2-SHA256 hashes with a random salt, never in plain text.

Activation keys look like FXA-7K3M-QW9P-2HTC-X4RD. The app ships only the SHA-256 hashes
of valid keys (activation_keys.json), so the keys themselves can't be read out of the app
or the repo. tools/make_keys.py makes new keys and adds their hashes to that file.
"""
import hashlib
import hmac
import json
import os
import secrets

PBKDF2_ITERATIONS = 240_000
MIN_PASSWORD = 8
KEY_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # no 0/O or 1/I to misread
KEY_PREFIX = "FXA"


def hash_password(password, salt=None, iterations=PBKDF2_ITERATIONS):
    """-> (hash hex, salt hex, iterations)."""
    salt = bytes.fromhex(salt) if salt else secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return h.hex(), salt.hex(), iterations


def check_password(password, pw_hash, salt, iterations):
    h, _, _ = hash_password(password, salt, iterations)
    return hmac.compare_digest(h, pw_hash)


def password_problem(password):
    """A plain-language reason the password is too weak, or None."""
    if len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters."
    if password.isdigit() or password.isalpha():
        return "Mix letters with numbers or symbols."
    return None


def normalise_key(text):
    """Accepts any case, spaces or missing dashes: 'fxa 7k3m qw9p 2htc x4rd' -> 'FXA-7K3M-QW9P-2HTC-X4RD'."""
    raw = "".join(c for c in text.upper() if c.isalnum())
    if raw.startswith(KEY_PREFIX):
        raw = raw[len(KEY_PREFIX):]
    if len(raw) != 16:
        return None
    return KEY_PREFIX + "-" + "-".join(raw[i:i + 4] for i in range(0, 16, 4))


def key_hash(key):
    return hashlib.sha256(key.encode("ascii")).hexdigest()


def new_key():
    raw = "".join(secrets.choice(KEY_ALPHABET) for _ in range(16))
    return normalise_key(raw)


def load_key_hashes(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f).get("sha256", []))
