# Bunion angle model

Measures the two bunion (hallux valgus) angles on a front-view (AP) foot X-ray:

- **HVA**, hallux valgus angle: big toe line vs 1st metatarsal line.
- **IMA**, intermetatarsal angle: 1st vs 2nd metatarsal line.

The model finds 6 points (two ends of each of the three bone lines) and the
angles are computed from them, so a doctor can see and correct the lines.

## Data

[HVAngleEst](https://www.scidb.cn/en/s/FVFFnq) (Sci Data 2025,
doi 10.1038/s41597-025-05261-9): 1,392 AP foot X-rays, one label row per foot.
License CC BY-NC-ND 4.0, so the images and labels are **not** in this repo.
Download them separately.

## Steps

```
pip install -r requirements.txt
python prep.py --data /path/to/hvangleest --out work/cache.npz
python train.py --cache work/cache.npz --weights work/resnet18_imagenet.pth --out work/run1
python evaluate.py --cache work/cache.npz --ckpt work/run1/best.pt --out work/run1
```

- `prep.py` cuts each labeled foot out of its X-ray (keeping the aspect ratio so
  angles don't change), mirrors left feet to look like right feet, and caches
  the crops plus the 6 points.
- `model.py` is a ResNet-18 with a small upsampling head: one heatmap per point,
  read out with a soft-argmax.
- `train.py` trains on the official train split, keeps the checkpoint with the
  lowest validation angle error, and reports HVA/IMA error and the share
  within 3°.
- ImageNet ResNet-18 starter weights: timm `resnet18_a1_0-d63eafa0.pth` from the
  pytorch-image-models GitHub releases.

## GPU training (Colab)

`train_colab.ipynb` downloads the data (`download_data.py`), then trains both
passes with a ResNet-34 at higher resolution and gives back the ONNX models
and test scores. Open it in Colab with a GPU runtime and run all cells.

Second pass: `zoom_prep.py` crops the forefoot around the first-pass points
from the original X-ray, and the same `train.py` / `evaluate.py` run on it.

`train.py --angle_w 0.3` adds a loss on the HVA/IMA angles and each bone line's
direction (fades in over epochs 11-20). On a local GTX 1650 (ResNet-34,
320x640, 60 epochs, `--bs 8`) it lifted HVA within 3° from 81.9% to 83.8% on
the official test set (`results/pass1_angle_gpu`).

## Foot finder

`footfinder.py` finds the left and/or right foot in a whole X-ray (ResNet-18,
input letterboxed to 320x512), so new X-rays don't need hand-drawn boxes.

```
python footfinder.py prep  --data /path/to/hvangleest --out work/feet.npz
python footfinder.py train --cache work/feet.npz --out work/footfinder
python footfinder.py eval  --cache work/feet.npz --ckpt work/footfinder/best.pt --data /path/to/hvangleest \
                           --bunion-cache work/cache.npz --bunion-ckpt work/run/best.pt --out work/footfinder
```

`eval` scores the boxes, then runs the bunion model on foot-finder crops and
exports `footfinder.onnx` (outputs: boxes `(1, 2, 4)` for left/right in input
pixels, and presence probabilities). Test results (`results/footfinder`):
left/right correct on all 140 test X-rays, mean box IoU 0.92, and end to end
HVA 84.4% / IMA 94.4% within 3°, the same as with the dataset's boxes.

## Notes

- The official split puts 42 patients in both train and test, so we also score
  test feet whose patients are not in train.
- The bunion model expects a foot crop (left feet mirrored); for new X-rays,
  take the crop from the foot finder's box.
