"""Input hardening: mark colours, patient IDs, report copies, 16-bit X-rays and the speech model download.

No model weights and no network needed.
"""
import hashlib
import io
import os
import pathlib
import zipfile

import numpy as np
import pytest
from PIL import Image

from remi_ai import api as api_mod
from remi_ai import report, voice


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod.Voice, "prepare", lambda self: self.status())  # no speech model download
    a = api_mod.Api(data_dir=str(tmp_path / "data"), open_files=False)
    a.create_doctor("DR-1", "Dr One", "secret1")
    a.add_patient({"patient_id": "RM-0001", "name": "Pat One"})
    yield a
    a._store.db.close()


@pytest.fixture
def xray_id(app, tmp_path):
    path = tmp_path / "side.png"
    Image.new("L", (64, 48), 120).save(path)
    vid = app._store.todays_visit("RM-0001", "DR-1")
    return app._store.add_xray(vid, str(path), "side", "doctor")


@pytest.fixture
def xray_16bit(tmp_path):
    path = tmp_path / "x16.tif"
    Image.fromarray(np.tile(np.linspace(0, 4000, 300), (200, 1)).astype(np.uint16)).save(path)
    return str(path)


# ---- mark colours go into SVG attributes, so only #rrggbb is stored ---------------------

@pytest.mark.parametrize("sent", ['"/onload=alert(1)', "red", "#fff", "#12345g", None, '#ff5a5f" x'])
def test_unsafe_mark_colour_is_stored_as_the_default(app, xray_id, sent):
    app.save_marks(xray_id, [{"type": "pen", "color": sent, "points": [[1, 2], [3, 4]]}])
    assert app._store.xray(xray_id)["marks"][0]["color"] == "#ff5a5f"


def test_palette_colours_are_kept_on_pen_and_flag_marks(app, xray_id):
    app.save_marks(xray_id, [{"type": "pen", "color": "#2ec4b6", "points": [[1, 2], [3, 4]]},
                             {"type": "flag", "color": "#7AA7FF", "text": "Spur", "pin": [5, 5], "at": [9, 9]}])
    assert [m["color"] for m in app._store.xray(xray_id)["marks"]] == ["#2ec4b6", "#7AA7FF"]


# ---- patient IDs are folder names ------------------------------------------------------

@pytest.mark.parametrize("pid", ["RM/01", "RM 01", "RM.01", "../X", "RM\\01"])
def test_patient_id_with_folder_unsafe_characters_is_refused(app, pid):
    r = app.add_patient({"patient_id": pid, "name": "P"})
    assert "letters, numbers, - and _" in r.get("error", "")
    assert app._store.patients() and all(p["patient_id"] == "RM-0001" for p in app._store.patients())


@pytest.mark.parametrize("pid", ["RM-0002", "rm_02", "A1"])
def test_patient_id_with_letters_numbers_dash_underscore_is_accepted(app, pid):
    assert app.add_patient({"patient_id": pid, "name": "P"})["patient_id"] == pid.upper()


# ---- copying report PDFs to Downloads is a setting ---------------------------------------

@pytest.fixture
def downloads(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)
    return home / "Downloads"


def test_report_is_copied_to_downloads_by_default(app, xray_id, downloads):
    assert app.account()["copy_to_downloads"] is True
    out = app.export_report("RM-0001")
    assert out["download"] == str(downloads / out["file"]) and os.path.exists(out["download"])


def test_report_stays_in_patient_files_when_copying_is_off(app, xray_id, downloads):
    assert app.set_copy_to_downloads(False) == {"copy_to_downloads": False}
    out = app.export_report("RM-0001")
    assert out["download"] is None and os.path.exists(out["path"]) and not list(downloads.iterdir())
    assert app.account()["copy_to_downloads"] is False


def test_copy_setting_needs_sign_in(app):
    app.logout()
    assert "sign in" in app.set_copy_to_downloads(False).get("error", "")


# ---- 16-bit X-rays are stretched to 8-bit everywhere they are shown ---------------------
# The test X-ray is a smooth left-to-right ramp from 0 to 4000. Stretched, it stays a ramp: dark on the
# left, mid-grey in the middle, light on the right. Converted without stretching, everything above 255
# clips to white, so the middle comes out white too.

def grey_ramp(data_url):
    import base64
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("L"), float)
    w = a.shape[1]
    return a[:, :w // 10].mean(), a[:, w // 2 - 5:w // 2 + 5].mean(), a[:, -w // 10:].mean()


def test_16bit_xray_shown_in_window_keeps_its_grey_ramp(xray_16bit):
    url, w, h = api_mod.image_data_url(xray_16bit)
    left, middle, right = grey_ramp(url)
    assert (w, h) == (300, 200) and left < 40 and 100 < middle < 160 and right > 215


def test_16bit_xray_thumbnail_in_referral_keeps_its_grey_ramp(xray_16bit):
    left, middle, right = grey_ramp(report.thumb(xray_16bit))
    assert left < 40 and 100 < middle < 160 and right > 215


# ---- the speech model is checked before it is unpacked ----------------------------------

def tiny_model_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{voice.MODEL_NAME}/conf/model.conf", "--sample-frequency=16000\n")
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def test_speech_model_with_matching_checksum_is_installed(tmp_path, monkeypatch):
    digest = tiny_model_zip(tmp_path / "m.zip")
    monkeypatch.setattr(voice, "MODEL_URL", (tmp_path / "m.zip").as_uri())
    monkeypatch.setattr(voice, "MODEL_SHA256", digest)
    v = voice.Voice(str(tmp_path / "data"))
    v._fetch_model()
    assert v._error == "" and os.path.isfile(os.path.join(v.model_path(), "conf", "model.conf"))


def test_speech_model_with_wrong_checksum_is_thrown_away(tmp_path, monkeypatch):
    tiny_model_zip(tmp_path / "m.zip")
    monkeypatch.setattr(voice, "MODEL_URL", (tmp_path / "m.zip").as_uri())
    monkeypatch.setattr(voice, "MODEL_SHA256", "0" * 64)
    v = voice.Voice(str(tmp_path / "data"))
    v._fetch_model()
    assert "damaged or changed" in v._error
    assert not os.path.exists(v.model_path()) and not os.listdir(v.dir)  # nothing unpacked, no .part left


def test_real_model_checksum_is_pinned():
    assert len(voice.MODEL_SHA256) == 64 and int(voice.MODEL_SHA256, 16)


# ---- spoken phrases reach the window once -----------------------------------------------

def test_poll_hands_over_each_phrase_once(tmp_path):
    v = voice.Voice(str(tmp_path))
    v._finals.append("remi open patient")
    assert v.poll()["finals"] == ["remi open patient"]
    assert v.poll()["finals"] == []
