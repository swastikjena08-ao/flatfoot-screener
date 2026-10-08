# Heel spur check

Finds the heel bone on a lateral foot X-ray, zooms in, and scores it as **normal**, **heel spur**, or **child's heel growth plate (Sever)**.

Two ResNet-18 models, both started from the flat foot classifier in `../../flatfoot_app/model.onnx`:

1. **Heel finder** (`train.py --task locate`): whole X-ray letterboxed to 256x256 -> box around the heel bone.
2. **Spur classifier** (`train.py --task classify`): square crop around that box (1.6x its longer side), 224x224 -> 3 class probabilities plus a 7x7 heatmap.

## Data

Kaggle "Heel Dataset" by Osamah Taher (Apache 2.0): 3,956 lateral foot X-rays from Kirkuk General Hospital, labelled Normal 1,842 / Heel Spur 1,316 / Sever 798, with a box around the heel bone in every image. Paper: Taher & Özacar, *HeCapsNet*, Int J Imaging Syst Technol 34(3), 2024.

The "Sever" images all show an open heel growth plate (children), so that class most likely means Sever's disease rather than a severe spur. A model can tell it apart just by spotting a child's heel, so spur vs normal is the main result. The images are not in this repo.

## Run

```
pip install -r requirements.txt
python download.py --out D:/heel-spur
python prep.py --data D:/heel-spur --split work/split.csv --out work/cache.npz
python train.py --task classify --cache work/cache.npz --init ../../flatfoot_app/model.onnx --out work/classify
python train.py --task locate   --cache work/cache.npz --init ../../flatfoot_app/model.onnx --out work/locate --epochs 30
python evaluate.py --data D:/heel-spur --split work/split.csv --locate work/locate/best.pt --classify work/classify/best.pt --out work/eval
python export.py --locate work/locate/best.pt --classify work/classify/best.pt --out ../../flatfoot_app
```

`prep.py` splits 70/15/15 by class with a fixed seed and keeps near-identical images (same perceptual hash) in one split. `train.py` uses the GPU when one is available (`--device cpu` to force CPU). `evaluate.py` crops with the heel finder's box, as the app would, and also reports the score with the hand-drawn box for comparison.

## Use in the app

`predict.py` runs the exported models with onnxruntime only (no PyTorch), the same way `flatfoot_app.py` does:

```python
from heel_spur.predict import HeelSpur
hs = HeelSpur("flatfoot_app")      # heel_finder.onnx, heel_spur.onnx, heel_spur.json
r = hs.predict(Image.open("xray.jpg"))
r["label"], r["probs"], r["box"], r["heatmap"], r["crop_box"]
```

`predict_check.py` confirms the ONNX pipeline matches the PyTorch evaluation on test X-rays.
