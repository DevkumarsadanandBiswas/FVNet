"""Threshold tuning on validation, test-set inference (resumable), full metrics, plots and comparison images."""
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from .. import visualize as viz
from ..models import build_model
from ..utils import save_json
from .inference import predict_volume
from .metrics import compute_metrics, remove_small
from .trainer import evaluate_full


def tune_threshold(model, store, split, cfg, P, dev, use_cuda):
    lo, hi, step = cfg["test"]["thr_range"]
    _, _, VP = evaluate_full(model, store, split["val"], cfg, dev, use_cuda, keep=True)
    ths = np.arange(lo, hi + 1e-9, step)
    G = {c: store.y(c) for c in split["val"]}
    vdice = [np.mean([2 * ((VP[c] > t) & G[c]).sum() / ((VP[c] > t).sum() + G[c].sum() + 1e-9)
                      for c in split["val"]]) for t in ths]
    thr = float(ths[int(np.argmax(vdice))])
    print(f"Best threshold on val: {thr:.2f} (Dice {max(vdice):.4f})")
    viz.draw_threshold(ths, vdice, thr, P)
    save_json(dict(threshold=thr, val_dice=float(max(vdice))), P["thr"])
    return thr


def test(cfg, P, split, store, dev, use_cuda, best_dice=float("nan")):
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(P["best"], map_location=dev))
    model.eval()
    thr = tune_threshold(model, store, split, cfg, P, dev, use_cuda)
    tc, roi = cfg["test"], cfg["infer"]["roi"]
    rows, PROBS, PREDS, SAMPLES = [], {}, {}, {}
    mrng = np.random.default_rng(cfg["seed"])
    print("\nRunning test inference...")
    for c in split["test"]:
        fpk = f"{P['tcache']}/case_{c}_thr{thr:.2f}.pkl"
        if os.path.exists(fpk):
            try:
                m, smp, prob16 = pickle.load(open(fpk, "rb"))
                rows.append(m); PROBS[c] = prob16; SAMPLES[c] = smp
                PREDS[c] = remove_small(prob16.astype(np.float32) > thr, tc["min_cc"])
                print(f"  case {c:>6}: [cached] Dice {m['Dice']:.4f} IoU {m['IoU']:.4f} clDice {m['clDice']:.4f}")
                continue
            except Exception:
                pass
        if use_cuda:
            torch.cuda.synchronize()
        t = time.time()
        prob = predict_volume(model, store.X[c], roi, dev, use_cuda, tc["tta"], cfg["infer"]["overlap"])
        if use_cuda:
            torch.cuda.synchronize()
        it = time.time() - t
        pred = remove_small(prob > thr, tc["min_cc"])
        gt = store.y(c)
        m, smp = compute_metrics(pred, gt, prob, store.SP[c], mrng)
        m["case"] = c; m["Infer_time_s"] = it
        rows.append(m); PROBS[c] = prob.astype(np.float16); PREDS[c] = pred; SAMPLES[c] = smp
        pickle.dump((m, smp, PROBS[c]), open(fpk, "wb"))
        print(f"  case {c:>6}: Dice {m['Dice']:.4f} IoU {m['IoU']:.4f} clDice {m['clDice']:.4f} "
              f"HD95 {m['HD95_mm']:.2f}mm AUC {m['AUC_ROC']:.4f} ({it:.1f}s)")
    R = pd.DataFrame(rows).set_index("case")
    R.to_csv(f"{P['out']}/test_metrics_per_case.csv")
    num = R.drop(columns=["TP", "FP", "FN", "TN"]).select_dtypes(include=[np.number])
    S = pd.DataFrame({"mean": num.mean(), "std": num.std(), "median": num.median(), "min": num.min(), "max": num.max()})
    S.to_csv(f"{P['out']}/test_metrics_summary.csv")
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print("\n================ TEST SUMMARY (mean / std / median / min / max) ================")
    print(S.round(4))
    viz.draw_test(R, SAMPLES, split["test"], thr, P)
    for c in split["test"]:
        viz.make_comparison(c, store, PREDS[c], PROBS[c], R, P, tc["n_slices"])
    save_json(dict(config=cfg, threshold=thr, best_val_dice=best_dice,
                   final_fusion_alpha=float(torch.sigmoid(model.alpha).item())), f"{P['out']}/config_used.json")
    return R, S
