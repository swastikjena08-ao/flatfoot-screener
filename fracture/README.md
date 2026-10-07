# Fracture flag

Looks at a foot or ankle X-ray and gives **a fracture probability** plus **a 12x12 map of where the break most likely is**, so the doctor's eye goes straight to it. The doctor still decides.

One ResNet-18, started from the flat foot classifier in `../flatfoot_app/model.onnx`. Its last feature map (12x12 at 384 px) goes through a 1x1 conv that scores every cell for "fracture here"; the image score is a soft maximum over the cells. Training uses both the yes/no label and the fracture boxes drawn in FracAtlas.

## Data

FracAtlas (Abedeen et al., *Scientific Data* 10, 521, 2023; figshare 22363012, CC BY 4.0): 4,083 X-rays from three hospitals in Bangladesh, 717 fractured, with a box around every fracture. Its body regions are only hand / leg / hip / shoulder, so `leg_region_labels.csv` sorts the 2,273 leg images by hand (by eye from thumbnails, not checked by a doctor):

| | fractured | not fractured |
|---|---|---|
| foot (toes, metatarsals, midfoot) | 27 | |
| ankle (at the joint) | 42 | |
| lower shin, just above the ankle | 46 | |
| foot or ankle view | | 870 |
| rest of the leg | 148 | 1,140 |

All leg images are used for training (a shin break still teaches what a break looks like); the number that matters is the foot and ankle part of the test split. Only 69 foot/ankle fractures exist in total, so test scores rest on a handful of X-rays. The images are not in this repo.

## Run

```
pip install -r requirements.txt
python download.py --out D:/fracatlas
python prep.py --data D:/fracatlas/FracAtlas --split work/split.csv --out work/cache.npz
python train.py --cache work/cache.npz --init ../flatfoot_app/model.onnx --out work/run
python evaluate.py --cache work/cache.npz --model work/run/best.pt --out work/run --overlays work/overlays
```

`prep.py` splits 65/15/20, except groups holding a foot or ankle fracture, which split 75/10/15 so more of these scarce breaks reach training. FracAtlas has no patient ids, so it keeps blocks of 10 consecutive image ids (views of one patient sit next to each other) and near-identical copies (same perceptual hash) in one split. `train.py` uses the GPU when one is available. `evaluate.py` picks the flag threshold on the validation split (catch 90% of its fractures) and reports the test split; "points at break" means the hottest cell overlaps a drawn fracture box.

`train_colab.ipynb` reruns everything on a Colab GPU at 512 px for 40 rounds.

## Results

CPU run (`results/cpu_384/`): 384 px, 20 rounds, best round 14 by validation score. Flag threshold set on validation to catch 90% of its fractures.

| test split | X-rays | fractured | AUC | fractures flagged | normals cleared | points at break |
|---|---|---|---|---|---|---|
| all leg | 470 | 50 | 0.86 | 80% | 66% | 48% |
| foot and ankle | 192 | 16 | 0.55 | 50% | 59% | 19% |

Found: shin and thigh 25/27, lower shin 7/7, ankle 3/4, **foot 5/12**. The model learned long-bone breaks but not foot breaks: only 11 foot fractures landed in the training split (12 in test), and small toe and metatarsal breaks shrink to a few pixels at 384 px. Not usable for feet yet.

GPU run (`results/gpu_512/`, GTX 1650): 512 px, 40 rounds (about 1.2 min each), best round 39 by validation score (0.97). Uses the re-split that puts 18 foot fractures in training, so its test split differs from the CPU run's. `log.json` holds per-round scores rounded to 4 decimals.

| test split | X-rays | fractured | AUC | fractures flagged | normals cleared | points at break |
|---|---|---|---|---|---|---|
| all leg | 449 | 56 | 0.92 | 82% | 94% | 64% |
| foot and ankle | 177 | 13 | 0.67 | 46% | 90% | 46% |

Found: shin and thigh 36/38, lower shin 4/5, ankle 4/8, **foot 2/5**. Long-bone breaks are now reliable. Foot and ankle stay weak: validation scored them at 0.99, test at 0.67, and with only 12-13 such fractures in each split neither number is solid. Treat the foot and ankle flag as a demo, not a measured result.
