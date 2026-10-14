"""YAML config loading with defaults and simple CLI overrides (``--set a.b=value``)."""
import copy
import yaml

DEFAULTS = {
    "dim": 3,
    "seed": 42,
    "out": "outputs_fvnet",
    "data": {"root": "", "layout": "imagecas", "image_dir": "images", "label_dir": "labels",
             "cache": "cache", "fov_dir": None, "split_file": None, "image_exts": [".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"]},
    "preprocess": {"ras": True, "target": [256, 256, 160], "window": [-100, 700], "sigmas": [0.8, 1.2, 1.8, 2.5],
                   "bright_vessels": True, "channel": "green", "clahe": False, "auto_fov": True, "alpha": 0.5, "beta": 0.5, "fg_samples": 30000},
    "split": [0.70, 0.15, 0.15],
    "model": {"in_channels": 3, "filters": [32, 64, 128, 256]},
    "train": {"patch": [96, 96, 96], "batch": 2, "epochs": 100, "iters": 80, "lr": 1.0e-3, "weight_decay": 1.0e-5,
              "fg_prob": 0.7, "full_val_every": 3, "n_val_patches": 16, "amp": True, "num_val_overlap": 0.5},
    "loss": {"lambdas": {"dice": 1.0, "bce": 0.5, "tversky": 0.5, "cl": 0.3}, "tv_alpha": 0.3, "tv_beta": 0.7,
             "skel_iters": 4, "ds_weights": {"vessel": 0.4, "fg": 0.3, "aux3": 0.2, "aux2": 0.1}},
    "infer": {"roi": [128, 128, 96], "overlap": 0.5},
    "test": {"tta": True, "min_cc": 30, "n_slices": 5, "thr_range": [0.1, 0.9, 0.05]},
}


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    if user.get("dim", 3) == 2:  # 2D-friendly defaults (overridable by the YAML)
        cfg["preprocess"].update(target=None, sigmas=[1.0, 1.5, 2.0, 3.0], bright_vessels=False)
        cfg["train"].update(patch=[256, 256], batch=8)
        cfg["infer"]["roi"] = [256, 256]
        cfg["test"]["min_cc"] = 20
    _merge(cfg, user)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    nd = cfg["dim"]
    for k in (("preprocess", "target"), ("train", "patch"), ("infer", "roi")):
        v = cfg[k[0]][k[1]]
        if v is not None and len(v) != nd:
            raise ValueError(f"{'.'.join(k)} must have {nd} entries for dim={nd}, got {v}")
    return cfg
