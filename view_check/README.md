# X-ray check

Looks at an uploaded X-ray and says what it is, so the app can send it to the right tool:

| View | Sent to |
| --- | --- |
| **side** (lateral foot / ankle) | flat foot (heel bone angle) and heel spur |
| **top_down** (dorsoplantar / AP foot) | bunion angles |
| **oblique** (angled foot view) | nothing yet, shown as "angled view, no measurement" |
| **not_foot** (any other body part) | rejected |

Below `min_confidence` (0.6 by default) the answer is **unsure** and the app should ask the user.

One ResNet-18 started from the flat foot classifier in `../flatfoot_app/model.onnx`, reading the whole X-ray letterboxed to 224x224.

## Data

A single label table, `labels.csv` (kept outside the repo with the images), built from five public sets:

| Source | Images used | Views | Licence |
| --- | --- | --- | --- |
| Heel spur set (Kaggle, Taher) | 3,948 | side | Apache 2.0 |
| Pesplanus (Kaggle copy of Mendeley sss89tjddy) | 538 | side | see Mendeley page |
| HVAngleEst (bunion) | 1,392 | top_down | CC BY-NC-ND 4.0 |
| FracAtlas | 3,992 | side / top_down / oblique / not_foot | CC BY 4.0 |
| UNIFESP body-part set (Kaggle) | 1,731 + 743 unlabeled | not_foot, plus 78 feet sorted by eye | CC BY-NC-SA 4.0 |

- Copies are removed with a 16x16 image hash. The Kaggle Pesplanus copy has 842 files but only 538 distinct images.
- Splits are 70/15/15 by patient (Pesplanus left and right foot, HVAngleEst patient id, UNIFESP patient id), so one person never sits on both sides.
- FracAtlas has no foot label and its "frontal" views mix foot and ankle shots, so its 395 foot/ankle front views were sorted by eye (359 top-down, 13 ankle-only, 21 end-on, 2 leg). The 78 UNIFESP foot X-rays were sorted the same way. Nothing here was checked by a doctor.
- **Outside test:** UNIFESP's 743 unlabeled test X-rays, sorted by eye (21 side, 11 top-down, 8 angled, 653 not a foot after removing copies). None of those patients appear in training. This is the score that says how it does on a hospital it has not seen.

Because nearly every top-down training image comes from one bunion dataset, the test also reports FracAtlas and UNIFESP X-rays on their own.

## Run

```
pip install -r ../heel_spur/requirements.txt
python prep.py --labels D:/datasets/xray-check/labels.csv --root D:/datasets --out work/cache.npz
python train.py --cache work/cache.npz --init ../flatfoot_app/model.onnx --out work/view
python evaluate.py --cache work/cache.npz --model work/view/best.pt --out work/eval
python export.py --model work/view/best.pt --out ../flatfoot_app
```

`train.py` draws the same number of X-rays from every class each round (`--per-class`, default 1,200), so the small angled class is not drowned out. It uses the GPU when there is one (`--device cpu` to force CPU); on a 4-core cloud CPU a round takes about 4 minutes.

## Use in the app

`predict.py` needs only numpy, Pillow and onnxruntime:

```python
from view_check.predict import ViewCheck
vc = ViewCheck("flatfoot_app")          # view_check.onnx, view_check.json
r = vc.predict(Image.open("xray.jpg"))
r["view"], r["probs"], r["tools"]       # "side", {...}, ["flat_foot", "heel_spur"]
```

`view_check.onnx` is trained partly on CC BY-NC-ND and CC BY-NC-SA images, so like the bunion models it is not committed; run `export.py` to make it.

## Results

Scores are recall per view (share of each view sorted correctly); "balanced" is their average, so the 41 angled test X-rays count as much as the 720 non-foot ones. `results/` holds the per-round logs and full test scores.

| | Held-out test (1,738) | FracAtlas only (581) | Outside test, new hospital (693) |
| --- | --- | --- | --- |
| **v2** (shipped) | 99.1% correct, 97.6% balanced | 98.5% correct | 98.1% correct, **83.1% balanced** |
| v3 (20 rounds instead of 8, not used) | 98.8% correct, 96.6% balanced | 98.5% correct | 97.8% correct, 72.5% balanced |
| v1 (no UNIFESP feet in training) | 99.0% correct, 96.4% balanced | 98.3% correct | 97.4% correct, 60.9% balanced |

Outside test, v2, per view: not a foot 649/653, side 16/21, top-down 9/11, angled 6/8.

v3 trained the same way for 20 rounds. It scored best on the check set (98.7% balanced at round 12, v2 about 98.6%) but worse on both tests, mostly side and angled views from the new hospital (side 13/21, angled 3/8). The extra rounds fit the training hospitals more closely, so v2 stays.

v1 learned that anything that looked like a UNIFESP X-ray was "not a foot" (every UNIFESP image it trained on was a non-foot) and turned away 10 of the 40 outside feet. v2 adds the 78 sorted UNIFESP feet to training; it turns away 2.

Known gaps:
- Side and angled views get mixed up on new hospitals (4 of 21 outside side views called angled).
- There is no class for ankle-only or end-on (heel axial) shots. The model calls most end-on shots top-down (22 of 26) and most ankle-only fronts "not a foot", so the app should not trust top-down blindly; the bunion foot finder is the second check.
- The outside test is small (40 feet) and was sorted by eye, not by a doctor.
