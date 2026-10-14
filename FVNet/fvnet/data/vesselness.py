"""GPU/CPU multi-scale Frangi-style vesselness (Hessian eigen-analysis) for 2D images and 3D volumes."""
import numpy as np
import torch
import torch.nn.functional as F


def gauss_nd(x, sigma):
    """Separable Gaussian blur of a (1,1,*spatial) tensor (replicate padding)."""
    nd = x.ndim - 2
    conv = F.conv2d if nd == 2 else F.conv3d
    r = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    for d in range(nd):
        pads = [0] * (2 * nd)
        pads[2 * (nd - 1 - d)] = r
        pads[2 * (nd - 1 - d) + 1] = r
        x = F.pad(x, pads, mode="replicate")
        shape = [1] * (2 + nd)
        shape[2 + d] = -1
        x = conv(x, k.view(shape))
    return x


def eig3(a11, a22, a33, a12, a13, a23):
    """Closed-form eigenvalues of a batch of symmetric 3x3 matrices."""
    q = (a11 + a22 + a33) / 3
    p1 = a12 ** 2 + a13 ** 2 + a23 ** 2
    p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2 * p1
    p = torch.sqrt(p2 / 6 + 1e-12)
    b11, b22, b33 = (a11 - q) / p, (a22 - q) / p, (a33 - q) / p
    b12, b13, b23 = a12 / p, a13 / p, a23 / p
    det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13)
    phi = torch.acos((det / 2).clamp(-1, 1)) / 3
    e1 = q + 2 * p * torch.cos(phi)
    e3 = q + 2 * p * torch.cos(phi + 2 * np.pi / 3)
    return e1, 3 * q - e1 - e3, e3


def eig2(a, b, c):
    """Eigenvalues of symmetric 2x2 matrices [[a,b],[b,c]]."""
    m = (a + c) / 2
    d = torch.sqrt(((a - c) / 2) ** 2 + b ** 2)
    return m + d, m - d


def _stride_sample(t, target=1_500_000):
    f = t.flatten()
    return f[:: max(1, f.numel() // target)]


def vesselness(vol, sigmas=(0.8, 1.2, 1.8, 2.5), alpha=0.5, beta=0.5, bright=True):
    """Multi-scale vesselness of ``vol`` shaped (1,1,*spatial), values in [0,1]; returns (*spatial) tensor.

    bright=False inverts the image first (dark vessels, e.g. green-channel fundus).
    """
    nd = vol.ndim - 2
    if not bright:
        vol = 1 - vol
    best = torch.zeros_like(vol[0, 0])
    for s in sigmas:
        g = gauss_nd(vol, s)[0, 0]
        d = [torch.gradient(g, dim=i)[0] for i in range(nd)]
        H = {(i, j): torch.gradient(d[i], dim=j)[0] * s ** 2 for i in range(nd) for j in range(i, nd)}
        if nd == 3:
            e = eig3(H[(0, 0)], H[(1, 1)], H[(2, 2)], H[(0, 1)], H[(0, 2)], H[(1, 2)])
        else:
            e = eig2(H[(0, 0)], H[(0, 1)], H[(1, 1)])
        del H, d, g
        v = torch.stack(e)
        v = torch.gather(v, 0, v.abs().argsort(0))  # sorted by |lambda| ascending
        if nd == 3:
            l1, l2, l3 = v[0], v[1], v[2]
            a1, a2, a3 = l1.abs(), l2.abs(), l3.abs()
            Ra = a2 / (a3 + 1e-10)
            Rb = a1 / (torch.sqrt(a2 * a3) + 1e-10)
            S = torch.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
            c = 0.5 * torch.quantile(_stride_sample(S), 0.995) + 1e-10
            V = (1 - torch.exp(-Ra ** 2 / (2 * alpha ** 2))) * torch.exp(-Rb ** 2 / (2 * beta ** 2)) \
                * (1 - torch.exp(-S ** 2 / (2 * c ** 2)))
            V[(l2 > 0) | (l3 > 0)] = 0
        else:
            l1, l2 = v[0], v[1]
            Rb = l1.abs() / (l2.abs() + 1e-10)
            S = torch.sqrt(l1 ** 2 + l2 ** 2)
            c = 0.5 * torch.quantile(_stride_sample(S), 0.995) + 1e-10
            V = torch.exp(-Rb ** 2 / (2 * beta ** 2)) * (1 - torch.exp(-S ** 2 / (2 * c ** 2)))
            V[l2 > 0] = 0
        best = torch.maximum(best, V)
    return best


def normalize_vesselness(ves, q=0.99):
    samp = _stride_sample(ves, 3_000_000)
    samp = samp[samp > 0]
    if samp.numel() == 0:
        return torch.zeros_like(ves)
    return (ves / (torch.quantile(samp[:16_000_000], q) + 1e-10)).clamp(0, 1)


def vessel_guided_enhancement(ct, vn):
    """Background-suppressed image: ct * (0.05 + 0.95 * sqrt(clip(4*vn)))."""
    gate = (vn * 4).clamp(0, 1).sqrt()
    return ct * (0.05 + 0.95 * gate)
