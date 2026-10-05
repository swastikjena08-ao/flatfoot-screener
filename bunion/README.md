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

## Notes

- The official split puts 42 patients in both train and test, so we also score
  test feet whose patients are not in train.
- The model expects a foot crop. The dataset's foot boxes are used for now; a
  foot finder is still needed for new X-rays.
