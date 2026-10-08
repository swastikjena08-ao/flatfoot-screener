"""Export the view classifier to ONNX for the app.

  view_check.onnx:  image 1x3x224x224 (letterboxed X-ray) -> probs 1x4 (side, top_down, oblique, not_foot)
  view_check.json:  class names, input size and the confidence cut-off used by predict.py

Usage:
  python export.py --model work/view/best.pt --out ../../flatfoot_app
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from common import CLASSES, SIZE, resnet18


class Probs(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).softmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.6)
    a = ap.parse_args()
    m = resnet18(len(CLASSES))
    m.load_state_dict(torch.load(a.model, map_location="cpu"))
    m.eval()
    torch.onnx.export(Probs(m), torch.zeros(1, 3, SIZE, SIZE), os.path.join(a.out, "view_check.onnx"),
                      input_names=["image"], output_names=["probs"], opset_version=17,
                      dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, dynamo=False)
    json.dump({"classes": CLASSES, "size": SIZE, "min_confidence": a.min_confidence},
              open(os.path.join(a.out, "view_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
