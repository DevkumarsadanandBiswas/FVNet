import json
import os
import random
import re

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("WARNING: no GPU found - running on CPU will be very slow.")
    else:
        torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if use_cuda else "cpu"), use_cuda


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def paths(cfg):
    out = cfg["out"]
    d = dict(out=out, fig=f"{out}/figures", ind=f"{out}/figures/individual", cmp=f"{out}/comparisons",
             ck=f"{out}/checkpoint_latest.pth", hist=f"{out}/training_history.csv", last=f"{out}/last_model.pth",
             best=f"{out}/best_model.pth", split=f"{out}/split.json", thr=f"{out}/threshold.json",
             tcache=f"{out}/test_cache", cache=cfg["data"]["cache"])
    for k in ("out", "fig", "ind", "cmp", "tcache", "cache"):
        os.makedirs(d[k], exist_ok=True)
    return d
