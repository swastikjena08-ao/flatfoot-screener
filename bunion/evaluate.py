"""Score a trained checkpoint on the test split and export it to ONNX.

Reports two test scores:
  - official: every foot in the official test split (as in the paper)
  - clean:    only test feet whose patient never appears in train or val

Usage: python evaluate.py --cache work/cache.npz --ckpt work/run1/best.pt --out work/run1
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from model import PointNet, soft_argmax
from train import predict, score


class Exported(torch.nn.Module):
    """Model + soft-argmax, so the ONNX file outputs points directly."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return soft_argmax(self.net(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18", help="must match the checkpoint")
    a = ap.parse_args()

    z = np.load(a.cache)
    imgs = z["imgs"]
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    model = PointNet(arch=a.arch)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    if torch.cuda.is_available():
        model = model.cuda()

    seen = set(meta[meta.split.isin(["train", "val"])].patient_id)
    sets = {
        "val": meta.split == "val",
        "test_official": meta.split == "test",
        "test_clean": (meta.split == "test") & ~meta.patient_id.isin(seen),
    }
    results, rows = {}, []
    for name, mask in sets.items():
        idx = np.where(mask)[0]
        s = score(predict(model, imgs[idx]), meta.iloc[idx])
        results[name] = {k: round(v, 4) for k, v in s.items() if not k.endswith("_pred")}
        results[name]["patients"] = int(meta.iloc[idx].patient_id.nunique())
        if name == "test_official":
            for j, h, i in zip(idx, s["hva_pred"], s["ima_pred"]):
                r = meta.iloc[j]
                rows.append(dict(filename=r.filename, side=r.side, HVA=r.HVA, IMA=r.IMA,
                                 HVA_pred=round(h, 2), IMA_pred=round(i, 2)))
        print(name, json.dumps(results[name]))
    with open(os.path.join(a.out, "test_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)

    model = model.cpu().eval()
    torch.onnx.export(Exported(model), torch.zeros(1, 3, *imgs.shape[1:]), os.path.join(a.out, "bunion_points.onnx"),
                      input_names=["image"], output_names=["points"], dynamic_axes={"image": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print("exported", os.path.join(a.out, "bunion_points.onnx"))


if __name__ == "__main__":
    main()
