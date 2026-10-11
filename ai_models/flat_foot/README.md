# Flat foot model

Looks at a side (lateral) foot X-ray and says whether the arch is flat. Two ResNet-18 models, both in this folder:

1. **Classifier** (`model.onnx`): 224x224 image -> flat foot probability plus a 7x7 heatmap of what it looked at.
2. **Heel bone angle** (`landmarks.onnx`, `landmarks.json`): finds two points on the heel bone, L10 and L11 (names from Lee et al., Yonsei Med J 2025). The L10->L11 angle against the horizontal is the heel bone angle (calcaneal pitch). **Under 17.7° = flat foot.**

Both were trained in Google Colab on right feet only (toes pointing left). Left feet are found automatically, mirrored, and the points mapped back.

## Test score

**93.8%** (198 of 211) on Pesplanus X-rays from patients whose points were never hand-labelled: flat feet caught 190/202, normal arches cleared 8/9. All 12 misses read 18.4°–19.7°, just above the cut-off. Only 9 unseen normal feet, so the normal-arch score needs a second dataset.

Data: Pesplanus dataset (Mendeley, https://data.mendeley.com/datasets/sss89tjddy). The X-rays and labels are not in this repo.

## Run

```
pip install numpy pillow onnxruntime
python predict.py xray.jpg                 # models load from this folder
python evaluate.py --data "D:/Pesplanus Dataset" --split pesplanus_split.csv
```

```python
from predict import FlatFoot
ff = FlatFoot(".")
r = ff.predict(Image.open("xray.jpg"))
r["angle"], r["flat_by_angle"], r["p_flat"], r["points"], r["left"], r["heatmap"]
```

`predict.py` runs both models with ONNX Runtime only (no PyTorch). The original Colab training notebooks are not in this repo.
