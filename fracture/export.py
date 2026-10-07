"""Export the trained fracture model to ONNX for the app.

  fracture.onnx: image 1x3xSxS (letterboxed X-ray) -> prob 1 (fracture, 0..1), cells 1 x S/32 x S/32 (0..1)
  fracture.json: input size and the flag threshold picked on the validation split by evaluate.py

Usage:
  python export.py --model work/run/best.pt --metrics work/run/metrics.json --size 512 --out ../flatfoot_app
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from common import FractureNet


class Probs(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y, cells = self.m(x)
        return y.sigmoid(), cells.sigmoid()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--metrics", required=True, help="evaluate.py metrics.json (holds the flag threshold)")
    ap.add_argument("--size", type=int, required=True, help="letterbox side the model was trained at")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = FractureNet()
    m.load_state_dict(torch.load(a.model, map_location="cpu"))
    torch.onnx.export(Probs(m).eval(), torch.zeros(1, 3, a.size, a.size), os.path.join(a.out, "fracture.onnx"),
                      input_names=["image"], output_names=["prob", "cells"], opset_version=17, dynamo=False)
    metrics = json.load(open(a.metrics))
    json.dump({"size": a.size, "threshold": metrics["threshold"]},
              open(os.path.join(a.out, "fracture.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
