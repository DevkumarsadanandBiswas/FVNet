"""Preprocessing for 3D volumes (CTA / MRA / MRI) and 2D images (fundus ...).

3D : RAS orientation -> intensity window + [0,1] normalisation -> resample to target grid
     -> multi-scale 3D Frangi vesselness -> vessel-guided enhancement
2D : channel select (green) -> optional CLAHE -> [0,1] normalisation -> optional resize
     -> multi-scale 2D Frangi vesselness -> vessel-guided enhancement
Both produce a 3-channel tensor [enhanced, vesselness, normalised image] (+ label, voxel spacing).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .vesselness import normalize_vesselness, vessel_guided_enhancement, vesselness


# ----------------------------------------------------------------------------- helpers
def _finish(ct, ves_args, label, sp, dev, fov=None):
    """ct: (1,1,*spatial) float tensor in [0,1].  label: float tensor (1,1,*spatial) or None."""
    ves = vesselness(ct, **ves_args)
    if fov is not None:
        ves = ves * fov
    vn = normalize_vesselness(ves)
    ct0 = ct[0, 0]
    enh = vessel_guided_enhancement(ct0, vn)
    x = torch.stack([enh, vn, ct0]).half().cpu().numpy()
    y = None if label is None else (label[0, 0] > 0.3).cpu().numpy().astype(np.uint8)
    return x, y, sp


def _resize(t, size, mode="area"):
    if size is None or tuple(t.shape[2:]) == tuple(size):
        return t
    return F.interpolate(t, size=tuple(size), mode=mode)


def pad_to_min(x, y, min_shape):
    """Zero-pad (C,*sp) / (*sp) arrays so every spatial dim >= min_shape."""
    sp = x.shape[1:]
    pads = [(0, max(m - s, 0)) for s, m in zip(sp, min_shape)]
    if any(p[1] for p in pads):
        x = np.pad(x, [(0, 0)] + pads)
        y = None if y is None else np.pad(y, pads)
    return x, y


# ----------------------------------------------------------------------------- 3D
def preprocess_3d(img_path, label_path, cfg, dev):
    import nibabel as nib
    pc = cfg["preprocess"]
    nii = nib.load(img_path)
    lnii = nib.load(label_path) if label_path else None
    if pc["ras"]:
        nii = nib.as_closest_canonical(nii)
        lnii = nib.as_closest_canonical(lnii) if lnii is not None else None
    vol = np.asarray(nii.dataobj).astype(np.float32)
    vol = vol.reshape(vol.shape[:3])
    lab = None if lnii is None else (np.asarray(lnii.dataobj).reshape(vol.shape) > 0)
    zooms = nii.header.get_zooms()[:3]
    target = pc["target"] or vol.shape
    sp = [float(zooms[i] * vol.shape[i] / target[i]) for i in range(3)]
    lo, hi = pc["window"]
    v = torch.from_numpy(vol).to(dev)[None, None]
    ct = (v.clamp(lo, hi) - lo) / (hi - lo)
    ct = _resize(ct, target)
    del v
    lt = None if lab is None else _resize(torch.from_numpy(lab.astype(np.float32)).to(dev)[None, None], target)
    args = dict(sigmas=tuple(pc["sigmas"]), alpha=pc["alpha"], beta=pc["beta"], bright=pc["bright_vessels"])
    x, y, sp = _finish(ct, args, lt, sp, dev)
    meta = dict(orig_shape=list(vol.shape), affine=nii.affine.tolist(), spacing_target=sp)  # shape/affine of the (RAS) array
    return x, y, sp, meta


# ----------------------------------------------------------------------------- 2D
def load_image_2d(path):
    """Returns float32 array (H,W,C) scaled to [0,1]."""
    from PIL import Image
    im = Image.open(path)
    if im.mode.startswith("I") or im.mode == "F":
        a = np.asarray(im).astype(np.float32)
        a = a / max(a.max(), 1e-6)
        return a[..., None]
    if im.mode in ("RGB", "RGBA", "P", "CMYK", "LA"):
        a = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        return a
    return np.asarray(im.convert("L")).astype(np.float32)[..., None] / 255.0


def load_mask_2d(path):
    from PIL import Image
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., :3].max(-1)
    return (a > 127) if a.max() > 1 else (a > 0)


def preprocess_2d(img_path, label_path, cfg, dev, fov_path=None):
    pc = cfg["preprocess"]
    rgb = load_image_2d(img_path)
    C = rgb.shape[-1]
    ch = pc["channel"]
    g = rgb[..., 1] if (ch == "green" and C >= 3) else (rgb.mean(-1) if ch == "gray" or C > 1 else rgb[..., 0])
    if fov_path:
        fov = load_mask_2d(fov_path)
    elif pc.get("auto_fov", True) and C >= 3:
        fov = ndi.binary_fill_holes(rgb.max(-1) > 0.1)
        fov = ndi.binary_erosion(fov, iterations=4)
    else:
        fov = np.ones(g.shape, bool)
    if fov.sum() < 0.05 * fov.size:  # degenerate FOV -> use everything
        fov = np.ones(g.shape, bool)
    if pc.get("clahe", False):
        from skimage import exposure
        g = exposure.equalize_adapthist(np.clip(g, 0, 1), clip_limit=0.01).astype(np.float32)
    inside = g[fov]
    lo, hi = np.percentile(inside, [1, 99])
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    g = np.where(fov, g, g[fov].mean()).astype(np.float32)  # no artificial edge at the FOV border
    lab = None if label_path is None else load_mask_2d(label_path)
    sp = [1.0, 1.0]
    ct = torch.from_numpy(g).to(dev)[None, None]
    fv = torch.from_numpy(fov.astype(np.float32)).to(dev)
    lt = None if lab is None else torch.from_numpy(lab.astype(np.float32)).to(dev)[None, None]
    target = pc["target"]
    if target:
        ct = _resize(ct, target, "bilinear" if target[0] >= g.shape[0] else "area")
        fv = _resize(fv[None, None], target)[0, 0] > 0.5
        lt = None if lt is None else _resize(lt, target)
        sp = [g.shape[0] / target[0], g.shape[1] / target[1]]
    else:
        fv = fv > 0.5
    args = dict(sigmas=tuple(pc["sigmas"]), alpha=pc["alpha"], beta=pc["beta"], bright=pc["bright_vessels"])
    x, y, sp = _finish(ct, args, lt, sp, dev, fov=fv.float())
    meta = dict(orig_shape=list(g.shape), fov=fov)
    return x, y, sp, meta


# ----------------------------------------------------------------------------- dispatch + caching
def preprocess_case(paths, cfg, dev, with_label=True):
    lab = paths.get("label") if with_label else None
    if cfg["dim"] == 3:
        return preprocess_3d(paths["img"], lab, cfg, dev)
    return preprocess_2d(paths["img"], lab, cfg, dev, paths.get("fov"))


def min_shape(cfg):
    return [max(a, b) for a, b in zip(cfg["train"]["patch"], cfg["infer"]["roi"])]


def cache_case(cid, paths, cfg, dev, cache_dir):
    """Write {cid}_x.npy / _y.npy / _sp.json / _fg.npy (resumable: skipped if all exist)."""
    fx, fy, fs, ff = (f"{cache_dir}/{cid}_{s}" for s in ("x.npy", "y.npy", "sp.json", "fg.npy"))
    if all(os.path.exists(f) for f in (fx, fy, fs, ff)):
        return
    x, y, sp, _ = preprocess_case(paths, cfg, dev)
    x, y = pad_to_min(x, y, min_shape(cfg))
    idx = np.flatnonzero(y)
    rs = np.random.RandomState(cfg["seed"])
    idx = idx[rs.permutation(len(idx))[: cfg["preprocess"]["fg_samples"]]]
    fg = np.stack(np.unravel_index(idx, y.shape), 1).astype(np.int32)
    np.save(fx, x)
    np.save(fy, y)
    np.save(ff, fg)
    json.dump(sp, open(fs, "w"))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
