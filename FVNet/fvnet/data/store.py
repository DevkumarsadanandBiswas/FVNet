"""Cached-case store (memory-mapped) + patch sampling / augmentation for training."""
import json
import random

import numpy as np
import torch


class CaseStore:
    def __init__(self, cache_dir, ids):
        self.ids = list(ids)
        self.X, self.Y, self.FG, self.SP = {}, {}, {}, {}
        for c in self.ids:
            self.X[c] = np.load(f"{cache_dir}/{c}_x.npy", mmap_mode="r")
            self.Y[c] = np.load(f"{cache_dir}/{c}_y.npy", mmap_mode="r")
            self.FG[c] = np.load(f"{cache_dir}/{c}_fg.npy")
            self.SP[c] = json.load(open(f"{cache_dir}/{c}_sp.json"))

    def x(self, c):
        return np.array(self.X[c])

    def y(self, c):
        return np.array(self.Y[c]) > 0


def sample_one(store, cid, patch, fg_prob, rng):
    shp = store.Y[cid].shape
    fg = store.FG[cid]
    if rng.random() < fg_prob and len(fg):
        c = fg[rng.randrange(len(fg))]
        st = [int(np.clip(c[i] - patch[i] // 2 + rng.randint(-patch[i] // 4, patch[i] // 4), 0, shp[i] - patch[i]))
              for i in range(len(patch))]
    else:
        st = [rng.randint(0, shp[i] - patch[i]) for i in range(len(patch))]
    sl = tuple(slice(s, s + p) for s, p in zip(st, patch))
    return np.array(store.X[cid][(slice(None),) + sl]), np.array(store.Y[cid][sl])[None]


def train_batch(store, ids, patch, batch, fg_prob, rng, dev):
    nd = len(patch)
    s = [sample_one(store, rng.choice(ids), patch, fg_prob, rng) for _ in range(batch)]
    x = torch.from_numpy(np.stack([a for a, _ in s]).astype(np.float32)).to(dev)
    y = torch.from_numpy(np.stack([b for _, b in s]).astype(np.float32)).to(dev)
    for ax in range(2, 2 + nd):
        if rng.random() < .5:
            x, y = x.flip(ax), y.flip(ax)
    if patch[0] == patch[1] and rng.random() < .5:
        x, y = x.transpose(2, 3), y.transpose(2, 3)
    x = x * rng.uniform(0.9, 1.1) + 0.02 * torch.randn_like(x)
    return x.contiguous(), y.contiguous()


def make_val_patches(store, ids, patch, n, dev):
    vr = random.Random(123)
    vp = [sample_one(store, vr.choice(ids), patch, 0.5, vr) for _ in range(n)]
    vx = torch.from_numpy(np.stack([a for a, _ in vp]).astype(np.float32)).to(dev)
    vy = torch.from_numpy(np.stack([b for _, b in vp]).astype(np.float32)).to(dev)
    return vx, vy
