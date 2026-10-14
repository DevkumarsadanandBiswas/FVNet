"""Command-line entry points: fvnet-preprocess / fvnet-train / fvnet-test / fvnet-predict / fvnet-all."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import visualize as viz
from .config import load_config
from .data.preprocess import preprocess_case
from .engine.inference import predict_volume
from .engine.metrics import remove_small
from .engine.setup import prepare
from .engine.tester import test as run_test
from .engine.trainer import train as run_train
from .models import build_model
from .utils import get_device, load_json, paths, seed_everything


def _parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", "-c", required=True, help="YAML config (see configs/)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override config values, e.g. --set train.epochs=50 data.root=/path out=runs/a")
    return ap


def _setup(args):
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    dev, use_cuda = get_device()
    return cfg, paths(cfg), dev, use_cuda


def preprocess_main(argv=None):
    cfg, P, dev, _ = _setup(_parser("Preprocess + cache a dataset").parse_args(argv))
    _, split, store = prepare(cfg, P, dev)
    viz.draw_dataset(store, split, P)
    viz.draw_preproc(store, split["train"][0], P)
    print(f"Cached to {P['cache']}")


def train_main(argv=None):
    cfg, P, dev, use_cuda = _setup(_parser("Train FVNet").parse_args(argv))
    _, split, store = prepare(cfg, P, dev)
    viz.draw_dataset(store, split, P)
    viz.draw_preproc(store, split["train"][0], P)
    H = run_train(cfg, P, split, store, dev, use_cuda)
    viz.draw_train(H, P)
    print(f"Done. best val dice = {float(H['val_dice_full'].max()):.4f}  (weights: {P['best']})")


def test_main(argv=None):
    cfg, P, dev, use_cuda = _setup(_parser("Evaluate a trained FVNet on the test split").parse_args(argv))
    _, split, store = prepare(cfg, P, dev, verbose=False)
    H = pd.read_csv(P["hist"]) if os.path.exists(P["hist"]) else None
    best = float(H["val_dice_full"].max()) if H is not None else float("nan")
    if H is not None:
        viz.draw_train(H, P)
    run_test(cfg, P, split, store, dev, use_cuda, best)
    print(f"Done. Results in {P['out']}")


def all_main(argv=None):
    cfg, P, dev, use_cuda = _setup(_parser("Preprocess -> train -> test (the whole pipeline)").parse_args(argv))
    _, split, store = prepare(cfg, P, dev)
    viz.draw_dataset(store, split, P)
    viz.draw_preproc(store, split["train"][0], P)
    H = run_train(cfg, P, split, store, dev, use_cuda)
    viz.draw_train(H, P)
    run_test(cfg, P, split, store, dev, use_cuda, float(H["val_dice_full"].max()))
    print(f"Done. Results in {P['out']}")


# ----------------------------------------------------------------------------- prediction on new data
def _list_inputs(path, cfg):
    if os.path.isdir(path):
        exts = tuple(cfg["data"]["image_exts"])
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(exts)]
    return [path]


def predict_main(argv=None):
    ap = _parser("Segment new images / volumes with a trained FVNet")
    ap.add_argument("--input", "-i", required=True, help="image/volume file or a folder of them")
    ap.add_argument("--output", "-o", required=True, help="output folder")
    ap.add_argument("--weights", default=None, help="default: <out>/best_model.pth")
    ap.add_argument("--threshold", type=float, default=None, help="default: tuned threshold in <out>/threshold.json, else 0.5")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args(argv)
    cfg, P, dev, use_cuda = _setup(args)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(args.weights or P["best"], map_location=dev))
    thr = args.threshold if args.threshold is not None else (load_json(P["thr"])["threshold"] if os.path.exists(P["thr"]) else 0.5)
    os.makedirs(args.output, exist_ok=True)
    for f in _list_inputs(args.input, cfg):
        x, _, _, meta = preprocess_case({"img": f}, cfg, dev, with_label=False)
        prob = predict_volume(model, x, cfg["infer"]["roi"], dev, use_cuda, not args.no_tta and cfg["test"]["tta"], cfg["infer"]["overlap"])
        shape = tuple(meta["orig_shape"])
        if prob.shape != shape:  # back to the native grid
            t = torch.from_numpy(prob)[None, None]
            prob = F.interpolate(t, size=shape, mode="trilinear" if prob.ndim == 3 else "bilinear", align_corners=False)[0, 0].numpy()
        mask = remove_small(prob > thr, cfg["test"]["min_cc"])
        name = os.path.basename(f).replace(".nii.gz", "").replace(".nii", "")
        name = os.path.splitext(name)[0] if cfg["dim"] == 2 else name
        if cfg["dim"] == 3:
            import nibabel as nib
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.array(meta["affine"])), f"{args.output}/{name}_pred.nii.gz")
        else:
            from PIL import Image
            mask = mask & meta["fov"]
            Image.fromarray((mask * 255).astype(np.uint8)).save(f"{args.output}/{name}_pred.png")
            Image.fromarray((prob * 255).astype(np.uint8)).save(f"{args.output}/{name}_prob.png")
        print(f"{f} -> {args.output}/{name}_pred (thr {thr:.2f}, {int(mask.sum())} vessel voxels)")
