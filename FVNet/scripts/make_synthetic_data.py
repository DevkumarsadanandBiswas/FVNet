"""Create tiny synthetic vessel datasets for smoke-testing the pipeline (no real data needed).

  python scripts/make_synthetic_data.py --dim 2 --out data/synth2d --n 8
  python scripts/make_synthetic_data.py --dim 3 --out data/synth3d --n 6
Layouts written:  2D -> images/<id>.png + labels/<id>.png   3D -> images/<id>.nii.gz + labels/<id>.nii.gz
"""
import argparse
import os

import numpy as np
from scipy import ndimage as ndi


def _walk(rng, shape, steps, step_len=1.5):
    p = rng.uniform(0.2, 0.8, len(shape)) * np.array(shape)
    d = rng.normal(size=len(shape)); d /= np.linalg.norm(d)
    pts = [p.copy()]
    for _ in range(steps):
        d = d + 0.35 * rng.normal(size=len(shape)); d /= np.linalg.norm(d)
        p = np.clip(p + d * step_len, 1, np.array(shape) - 2)
        pts.append(p.copy())
    return np.array(pts)


def make_case(rng, shape, n_vessels=4, radius=1.6):
    line = np.zeros(shape, bool)
    for _ in range(n_vessels):
        for p in _walk(rng, shape, steps=int(max(shape) * 1.2)):
            line[tuple(np.round(p).astype(int))] = True
    dist = ndi.distance_transform_edt(~line)
    mask = dist <= radius
    img = ndi.gaussian_filter(mask.astype(np.float32), 0.7)
    return mask, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(f"{a.out}/images", exist_ok=True)
    os.makedirs(f"{a.out}/labels", exist_ok=True)
    if a.dim == 2:
        from PIL import Image
        shape = tuple(a.size or (96, 96))
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        fov = ((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2) < (0.48 * min(shape)) ** 2
        for i in range(a.n):
            mask, v = make_case(rng, shape, n_vessels=5, radius=1.5)
            bg = 0.55 + 0.1 * ndi.gaussian_filter(rng.normal(size=shape), 6) * 6
            g = np.clip(bg - 0.35 * v + 0.03 * rng.normal(size=shape), 0, 1) * fov  # dark vessels
            rgb = np.stack([np.clip(g * 1.3, 0, 1), g, g * 0.5], -1)
            Image.fromarray((rgb * 255).astype(np.uint8)).save(f"{a.out}/images/{i:02d}_img.png")
            Image.fromarray(((mask & fov) * 255).astype(np.uint8)).save(f"{a.out}/labels/{i:02d}_manual.png")
    else:
        import nibabel as nib
        shape = tuple(a.size or (48, 48, 32))
        for i in range(a.n):
            mask, v = make_case(rng, shape, n_vessels=3, radius=1.5)
            hu = -50 + 40 * ndi.gaussian_filter(rng.normal(size=shape), 2) * 3 + 450 * v + 15 * rng.normal(size=shape)
            aff = np.diag([0.5, 0.5, 0.7, 1.0])
            nib.save(nib.Nifti1Image(hu.astype(np.float32), aff), f"{a.out}/images/{i:02d}.nii.gz")
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), f"{a.out}/labels/{i:02d}.nii.gz")
    print(f"wrote {a.n} synthetic {a.dim}D cases to {a.out}")


if __name__ == "__main__":
    main()
