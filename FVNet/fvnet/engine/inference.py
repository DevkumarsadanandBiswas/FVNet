"""Sliding-window inference (2D / 3D) with Gaussian blending, zero-padding and flip TTA."""
import itertools

import numpy as np
import torch
import torch.nn.functional as F


def make_weight(roi, dev):
    nd = len(roi)
    w = 1.0
    for d, r in enumerate(roi):
        t = torch.arange(r, device=dev).float()
        g = torch.exp(-0.5 * ((t - (r - 1) / 2) / (0.3 * r)) ** 2) + 1e-3
        shape = [1] * nd
        shape[d] = r
        w = w * g.view(shape)
    return w


@torch.no_grad()
def predict_volume(model, x, roi, dev, use_cuda, tta=False, overlap=0.5):
    """x: array (C,*spatial). Returns probability array (*spatial) float32 in [0,1]."""
    model.eval()
    xt = torch.from_numpy(np.array(x)).to(dev).float()
    nd = len(roi)
    sp = tuple(xt.shape[1:])
    pads = []
    for s, r in zip(reversed(sp), reversed(roi)):
        pads += [0, max(r - s, 0)]
    if any(pads):
        xt = F.pad(xt, pads)
    psp = tuple(xt.shape[1:])
    starts = [sorted(set(list(range(0, max(d - r, 0) + 1, max(int(r * (1 - overlap)), 1))) + [max(d - r, 0)]))
              for d, r in zip(psp, roi)]
    wt = make_weight(roi, dev)
    prob = torch.zeros(psp, device=dev)
    cnt = torch.zeros_like(prob)
    flips = [[]] + ([[d] for d in range(2, 2 + nd)] if tta else [])
    for st in itertools.product(*starts):
        sl = tuple(slice(a, a + r) for a, r in zip(st, roi))
        patch = xt[(slice(None),) + sl][None]
        acc = 0
        for f in flips:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                o = model(patch.flip(f) if f else patch)
            o = torch.sigmoid(o.float())
            acc = acc + (o.flip(f) if f else o)
        prob[sl] += (acc / len(flips))[0, 0] * wt
        cnt[sl] += wt
    prob = prob / cnt
    return prob[tuple(slice(0, s) for s in sp)].cpu().numpy()
