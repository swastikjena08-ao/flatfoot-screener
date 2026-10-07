"""Export a trained fracture model to ONNX for the app (flatfoot_app/fracture.onnx).

The ONNX model takes the letterboxed X-ray (1 x 3 x S x S, ImageNet-normalised gray) and
returns the fracture probability and the (S/32) x (S/32) map of per-cell probabilities.
fracture.json next to it holds the image size and the flag threshold from evaluate.py.

Usage:
  python export.py --model work/run/best.pt --metrics work/run/metrics.json --size 512 --out ../flatfoot_app
"""
import argparse
import json
import os

import torch

from common import FractureNet


class Probs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logit, cells = self.net(x)
        return torch.sigmoid(logit), torch.sigmoid(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best.pt from train.py")
    ap.add_argument("--metrics", required=True, help="metrics.json from evaluate.py (holds the threshold)")
    ap.add_argument("--size", type=int, required=True, help="image size the model was trained at")
    ap.add_argument("--out", required=True, help="folder for fracture.onnx and fracture.json")
    a = ap.parse_args()

    net = FractureNet()
    net.load_state_dict(torch.load(a.model, map_location="cpu"))
    model = Probs(net).eval()
    os.makedirs(a.out, exist_ok=True)
    x = torch.zeros(1, 3, a.size, a.size)
    torch.onnx.export(model, x, os.path.join(a.out, "fracture.onnx"), input_names=["image"],
                      output_names=["prob", "cells"], dynamic_axes={"image": {0: "n"}, "prob": {0: "n"},
                                                                     "cells": {0: "n"}},
                      opset_version=17, dynamo=False)
    m = json.load(open(a.metrics))
    meta = {"size": a.size, "threshold": m["threshold"],
            "test": {k: m[k] for k in ("test_all_leg", "test_foot_ankle", "found_foot_fractures",
                                       "found_ankle_fractures")}}
    json.dump(meta, open(os.path.join(a.out, "fracture.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, "fracture.onnx"), "threshold", round(m["threshold"], 4))


if __name__ == "__main__":
    main()
