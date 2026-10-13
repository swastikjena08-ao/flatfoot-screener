# Remi AI

A Windows desktop assistant that reads foot X-rays for doctors. A doctor signs in, opens a patient's record and adds an X-ray. Remi first checks what kind of X-ray it is, then runs only the AI models that fit that view:

| X-ray | Checks |
|---|---|
| Side view | Flat foot (heel bone angle), heel spur |
| Top-down view | Bunion angles (HVA, IMA) |
| Anything else | None; the X-ray is kept but not measured |

Each patient has a record with their first-visit history, medical history, visits, notes and every X-ray, so results can be compared over time. An assistant panel (Ask Remi) guides the doctor through each step by text or voice.

## Run it

On Windows 10/11 with Python 3.10 or newer, double-click `run_remi.bat`. The first run installs what it needs into a private `.venv` folder.

Patient files are stored on the computer in `%APPDATA%\Remi AI` (one folder per patient plus a small database), never in this repository.

To build a standalone `Remi AI.exe`, run `build_exe.bat`; the result is in `dist\Remi AI\`.

Developers can open the same app in a browser with `python -m remi_ai --browser`.

## Model files

The app loads each model from its folder in [`ai_models/`](ai_models/). The flat foot and heel spur weights are in the repository. The X-ray check and bunion weights are not (their training data licences don't allow sharing them) and must be copied in by hand:

- `ai_models/view_check/view_check.onnx`
- `ai_models/bunion/bunion_points.onnx` and `ai_models/bunion/bunion_zoom.onnx` (optional: `footfinder.onnx` for X-rays showing both feet)

Without the X-ray check, Remi asks the doctor what kind of X-ray each upload is. Each model folder's README covers its dataset, training and test scores.

## Layout

- `remi_ai/`: the app (`api.py` for everything the window can do, `engine.py` for model routing, `store.py` for patient records, `web/` for the interface)
- `ai_models/`: the trained models, one folder each
