"""Foreground-VesselNet (FVNet): 2D / 3D blood-vessel segmentation network.

Encoder (4 stages): Conv Block -> FAM -> MSFB                      -> F1..F4
Bottleneck        : dilated convs (r=1,2,4) + context aggregation + channel & spatial attention
Decoder (Up3..1)  : up-sampling -> Attention Gate on skip -> concat -> Conv Block
Dual-branch heads : vessel segmentation head + foreground-confidence head (1x1 conv each)
Adaptive fusion   : P = alpha * P_v + (1 - alpha) * P_f,  alpha = sigmoid(learnable scalar)

``ndim=2`` builds the 2D network (images), ``ndim=3`` the 3D network (volumes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (Bottleneck, ConvBlock, DecoderStage, EncoderStage, conv_cls, convT_cls, norm, act)

DIV = 16  # stem (x2) + 3 encoder downsamplings (x2 each) -> input is padded to a multiple of 16


class ForegroundVesselNet(nn.Module):
    def __init__(self, ndim=3, in_channels=3, filters=(32, 64, 128, 256)):
        super().__init__()
        assert ndim in (2, 3)
        self.nd = nd = ndim
        C, CT = conv_cls(nd), convT_cls(nd)
        f1, f2, f3, f4 = filters
        self.stem = nn.Sequential(C(in_channels, f1, 3, 2, 1, bias=False), norm(nd, f1), act())  # H/2
        self.enc1 = EncoderStage(nd, f1, f1, 1)   # H/2
        self.enc2 = EncoderStage(nd, f1, f2, 2)   # H/4
        self.enc3 = EncoderStage(nd, f2, f3, 2)   # H/8
        self.enc4 = EncoderStage(nd, f3, f4, 2)   # H/16
        self.bott = Bottleneck(nd, f4)
        self.up3 = DecoderStage(nd, f4, f3, f3)   # H/8
        self.up2 = DecoderStage(nd, f3, f2, f2)   # H/4
        self.up1 = DecoderStage(nd, f2, f1, f1)   # H/2
        c0 = max(f1 // 2, 16)
        self.final_up = CT(f1, c0, 2, 2)                       # H/2 -> full resolution
        self.final_conv = ConvBlock(nd, c0 + in_channels, c0)  # re-inject raw input: keeps thin-vessel detail
        self.vessel_head = C(c0, 1, 1)                         # Vessel Segmentation Head (1x1)
        self.fg_head = C(c0, 1, 1)                             # Foreground Confidence Head (1x1)
        self.alpha = nn.Parameter(torch.tensor(1.0))           # adaptive fusion weight: alpha = sigmoid(.)
        self.aux3 = C(f3, 1, 1)                                # deep-supervision heads
        self.aux2 = C(f2, 1, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")

    def _pad(self, x):
        sp = x.shape[2:]
        pads = []
        for s in reversed(sp):
            pads += [0, (-s) % DIV]
        return F.pad(x, pads) if any(pads) else x

    def forward(self, x):
        orig = x.shape[2:]
        x = self._pad(x)
        size = x.shape[2:]
        crop = (slice(None), slice(None)) + tuple(slice(0, s) for s in orig)
        F1 = self.enc1(self.stem(x))
        F2 = self.enc2(F1)
        F3 = self.enc3(F2)
        F4 = self.enc4(F3)
        b = self.bott(F4)
        d3 = self.up3(b, F3)
        d2 = self.up2(d3, F2)
        d1 = self.up1(d2, F1)
        h = self.final_conv(torch.cat([self.final_up(d1), x], 1))
        lv, lf = self.vessel_head(h).float(), self.fg_head(h).float()
        a = torch.sigmoid(self.alpha)
        P = (a * torch.sigmoid(lv) + (1 - a) * torch.sigmoid(lf)).clamp(1e-6, 1 - 1e-6)
        fused = torch.log(P / (1 - P))  # logit of the fused probability (BCE/sigmoid compatible)
        if self.training:
            mode = "bilinear" if self.nd == 2 else "trilinear"
            up = lambda t: F.interpolate(t.float(), size=size, mode=mode, align_corners=False)
            outs = [fused, lv, lf, up(self.aux3(d3)), up(self.aux2(d2))]
            return [o[crop] for o in outs]
        return fused[crop]


def build_model(cfg):
    m = cfg["model"]
    return ForegroundVesselNet(ndim=cfg["dim"], in_channels=m.get("in_channels", 3), filters=tuple(m["filters"]))
