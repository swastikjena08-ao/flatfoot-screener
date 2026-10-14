# Full-dataset test

Runs every X-ray collected for the project through the Remi AI engine (`remi_ai/engine.py`)
and scores the outcomes against each dataset's own labels.

| Step | What is checked | Data |
|------|-----------------|------|
| Routing | Does the X-ray check send each X-ray to the right checks? | X-ray check table (side, top-down, angled, not a foot) |
| Flat foot | Arch verdict vs the dataset folder (flat / normal) | Pesplanus (Mendeley), 540 distinct X-rays |
| Heel spur | Normal / spur / Sever's vs the dataset label | Heel spur dataset (Kaggle), 3,956 X-rays |
| Bunion | HVA and IMA vs doctors' angles, per foot | HVAngleEst, 1,392 X-rays |

Each score is split into X-rays the model learned from and X-rays it never saw. Only the
never-seen numbers are accuracy claims.

```
python full_test.py --app <checkout with remi_ai/ and ai_models/ incl. weights> \
    --data <datasets folder> --out runs \
    --pesplanus-split pesplanus_split.csv --heel-spur-split split.csv
python summarise.py --runs runs --out summary.json
```

Needs numpy, Pillow, opencv-python-headless and onnxruntime. X-rays, labels and per-image
outputs are not part of the repository; `summary.json` holds only aggregate scores.
