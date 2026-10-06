"""Export the trained models to ONNX for the app.

  heel_finder.onnx: image 1x3x256x256 (letterboxed X-ray) -> box 1x4 (x0, y0, x1, y1, 0..1 of the square)
  heel_spur.onnx:   image 1x3x224x224 (heel crop) -> probs 1x3 (normal, spur, sever), cam 1x3x7x7

Usage:
  python export.py --locate work/locate/best.pt --classify work/classify/best.pt --out ../flatfoot_app
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from common import CLASSES, CROP_PAD, CROP_SIZE, LOC_SIZE, resnet18


class Finder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        p = self.m(x).sigmoid()
        return torch.cat([p[:, :2] - p[:, 2:] / 2, p[:, :2] + p[:, 2:] / 2], 1)


class SpurWithCam(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.body = nn.Sequential(*list(m.children())[:-2])
        self.fc = m.fc

    def forward(self, x):
        f = self.body(x)                                   # N x 512 x 7 x 7
        probs = self.fc(f.mean((2, 3))).softmax(1)
        cam = torch.einsum("kc,nchw->nkhw", self.fc.weight, f)
        return probs, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locate", required=True)
    ap.add_argument("--classify", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    loc = resnet18(4)
    loc.load_state_dict(torch.load(a.locate, map_location="cpu"))
    torch.onnx.export(Finder(loc).eval(), torch.zeros(1, 3, LOC_SIZE, LOC_SIZE),
                      os.path.join(a.out, "heel_finder.onnx"), input_names=["image"],
                      output_names=["box"], opset_version=17, dynamo=False)
    clf = resnet18(len(CLASSES))
    clf.load_state_dict(torch.load(a.classify, map_location="cpu"))
    torch.onnx.export(SpurWithCam(clf).eval(), torch.zeros(1, 3, CROP_SIZE, CROP_SIZE),
                      os.path.join(a.out, "heel_spur.onnx"), input_names=["image"],
                      output_names=["probs", "cam"], opset_version=17, dynamo=False)
    json.dump({"classes": CLASSES, "finder_size": LOC_SIZE, "crop_size": CROP_SIZE, "crop_pad": CROP_PAD},
              open(os.path.join(a.out, "heel_spur.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
