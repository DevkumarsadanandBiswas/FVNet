"""Dimension-generic building blocks (2D or 3D) for Foreground-VesselNet."""
import torch
import torch.nn as nn


def conv_cls(nd):
    return nn.Conv2d if nd == 2 else nn.Conv3d


def convT_cls(nd):
    return nn.ConvTranspose2d if nd == 2 else nn.ConvTranspose3d


def norm(nd, c):
    return (nn.InstanceNorm2d if nd == 2 else nn.InstanceNorm3d)(c, affine=True)


def act():
    return nn.LeakyReLU(0.01, inplace=True)


class ConvBlock(nn.Module):
    """Conv Block (3x3[x3]): two conv-norm-act; the first conv may be strided (downsampling)."""

    def __init__(self, nd, cin, cout, stride=1):
        super().__init__()
        C = conv_cls(nd)
        self.body = nn.Sequential(
            C(cin, cout, 3, stride, 1, bias=False), norm(nd, cout), act(),
            C(cout, cout, 3, 1, 1, bias=False), norm(nd, cout), act())

    def forward(self, x):
        return self.body(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, nd, c, r=8):
        super().__init__()
        C = conv_cls(nd)
        h = max(c // r, 4)
        self.dims = tuple(range(2, 2 + nd))
        self.fc = nn.Sequential(C(c, h, 1), nn.ReLU(inplace=True), C(h, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(self.dims, keepdim=True))


class SpatialAttention(nn.Module):
    def __init__(self, nd, k=7):
        super().__init__()
        self.conv = conv_cls(nd)(2, 1, k, padding=k // 2)

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(a))


class FAM(nn.Module):
    """Foreground Attention Module: Channel Attention (SE) -> Spatial Attention -> feature reweighting."""

    def __init__(self, nd, c):
        super().__init__()
        self.ca = ChannelAttention(nd, c)
        self.sa = SpatialAttention(nd)

    def forward(self, x):
        return self.sa(self.ca(x))


class MSFB(nn.Module):
    """Multi-Scale Feature Block: parallel 3x3 | 5x5 | dilated 3x3 (r=2) -> concat -> 1x1 conv."""

    def __init__(self, nd, c):
        super().__init__()
        C = conv_cls(nd)
        self.b3 = nn.Sequential(C(c, c, 3, 1, 1, bias=False), norm(nd, c), act())
        self.b5 = nn.Sequential(C(c, c, 5, 1, 2, bias=False), norm(nd, c), act())
        self.bd = nn.Sequential(C(c, c, 3, 1, 2, dilation=2, bias=False), norm(nd, c), act())
        self.fuse = nn.Sequential(C(3 * c, c, 1, bias=False), norm(nd, c), act())

    def forward(self, x):
        return self.fuse(torch.cat([self.b3(x), self.b5(x), self.bd(x)], 1))


class EncoderStage(nn.Module):
    """Conv Block -> FAM -> MSFB  (F_i)."""

    def __init__(self, nd, cin, cout, stride):
        super().__init__()
        self.conv = ConvBlock(nd, cin, cout, stride)
        self.fam = FAM(nd, cout)
        self.msfb = MSFB(nd, cout)

    def forward(self, x):
        return self.msfb(self.fam(self.conv(x)))


class Bottleneck(nn.Module):
    """Dilated convs (r=1,2,4) -> context aggregation -> channel & spatial attention."""

    def __init__(self, nd, c):
        super().__init__()
        C = conv_cls(nd)
        self.dims = tuple(range(2, 2 + nd))
        self.dil = nn.ModuleList([
            nn.Sequential(C(c, c, 3, 1, r, dilation=r, bias=False), norm(nd, c), act()) for r in (1, 2, 4)])
        self.fuse = nn.Sequential(C(3 * c, c, 1, bias=False), norm(nd, c), act())
        self.gctx = nn.Sequential(C(c, c, 1), act())  # global context vector
        self.agg = ConvBlock(nd, c, c)
        self.att = FAM(nd, c)

    def forward(self, x):
        x = self.fuse(torch.cat([d(x) for d in self.dil], 1))
        x = self.agg(x + self.gctx(x.mean(self.dims, keepdim=True)))
        return self.att(x)


class AttentionGate(nn.Module):
    """Skip F_s + gating F_g -> 1x1 convs -> concat -> ReLU -> 1x1 conv -> sigmoid -> F_s' = A * F_s."""

    def __init__(self, nd, cs, cg, ci):
        super().__init__()
        C = conv_cls(nd)
        self.ws = C(cs, ci, 1)
        self.wg = C(cg, ci, 1)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), C(2 * ci, 1, 1), nn.Sigmoid())

    def forward(self, fs, fg):
        return fs * self.psi(torch.cat([self.ws(fs), self.wg(fg)], 1))


class DecoderStage(nn.Module):
    """Upsampling -> Attention Gate on skip -> concat -> Conv Block."""

    def __init__(self, nd, cin, cskip, cout):
        super().__init__()
        self.up = convT_cls(nd)(cin, cskip, 2, 2)
        self.ag = AttentionGate(nd, cskip, cskip, max(cskip // 2, 8))
        self.conv = ConvBlock(nd, 2 * cskip, cout)

    def forward(self, x, skip):
        g = self.up(x)
        return self.conv(torch.cat([g, self.ag(skip, g)], 1))
