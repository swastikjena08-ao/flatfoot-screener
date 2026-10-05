"""ResNet-18 backbone with a small upsampling head that outputs one heatmap per point.

Points are read out with a soft-argmax, so training can use a plain coordinate loss
and predictions are sub-pixel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from common import N_PTS, STRIDE


class PointNet(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        r = torchvision.models.resnet18()
        if pretrained_path:
            sd = torch.load(pretrained_path, map_location="cpu")
            r.load_state_dict(sd, strict=False)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4

        def up(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        # stride 32 -> 16 -> 8 -> 4, with skip connections from the backbone
        self.u3 = up(512 + 256, 256)
        self.u2 = up(256 + 128, 128)
        self.u1 = up(128 + 64, 64)
        self.out = nn.Conv2d(64, N_PTS, 1)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        y = self.u3(torch.cat([F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False), f3], 1))
        y = self.u2(torch.cat([F.interpolate(y, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
        y = self.u1(torch.cat([F.interpolate(y, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
        return self.out(y)  # (B, N_PTS, H/4, W/4) logits


def soft_argmax(logits):
    """Heatmap logits -> (B, N, 2) points in input-pixel coordinates (x, y)."""
    b, n, h, w = logits.shape
    p = F.softmax(logits.reshape(b, n, -1), -1).reshape(b, n, h, w)
    xs = (torch.arange(w, dtype=p.dtype) + 0.5) * STRIDE
    ys = (torch.arange(h, dtype=p.dtype) + 0.5) * STRIDE
    x = (p.sum(2) * xs).sum(-1)
    y = (p.sum(3) * ys).sum(-1)
    return torch.stack([x, y], -1)
