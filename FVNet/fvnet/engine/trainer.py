"""Resumable training loop (AMP, deep supervision, compound loss, periodic full-volume validation)."""
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.store import make_val_patches, train_batch
from ..losses import FVNetLoss, batch_dice
from ..models import build_model
from .inference import predict_volume


def make_loss(cfg):
    L = cfg["loss"]
    return FVNetLoss(L["lambdas"], L["tv_alpha"], L["tv_beta"], L["skel_iters"], L["ds_weights"])


def evaluate_full(model, store, ids, cfg, dev, use_cuda, keep=False, tta=False, thr=0.5):
    ds, ious, P = [], [], {}
    for c in ids:
        p = predict_volume(model, store.X[c], cfg["infer"]["roi"], dev, use_cuda, tta, cfg["infer"]["overlap"])
        pr, g = p > thr, store.y(c)
        i = (pr & g).sum()
        ds.append(2 * i / (pr.sum() + g.sum() + 1e-9))
        ious.append(i / ((pr | g).sum() + 1e-9))
        if keep:
            P[c] = p.astype(np.float16)
    return float(np.mean(ds)), float(np.mean(ious)), P


@torch.no_grad()
def eval_val_patches(model, loss_fn, vx, vy, use_cuda):
    model.eval()
    L, D = [], []
    for i in range(0, len(vx), 2):
        x, y = vx[i:i + 2], vy[i:i + 2]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
            o = model(x)
        L.append(loss_fn(o.float(), y).item())
        D.append(batch_dice(o.float(), y))
    return float(np.mean(L)), float(np.mean(D))


def train(cfg, P, split, store, dev, use_cuda):
    tc = cfg["train"]
    E = tc["epochs"]
    model = build_model(cfg).to(dev)
    print(f"Foreground-VesselNet ({cfg['dim']}D) params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    loss_fn = make_loss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E, eta_min=1e-5)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    hist, best_dice, start_ep, rng = [], -1.0, 1, random.Random(cfg["seed"])

    if os.path.exists(P["last"]) and os.path.exists(P["best"]) and os.path.exists(P["hist"]) \
            and len(pd.read_csv(P["hist"])) >= E:
        print(f"Training already complete ({E} epochs) - skipping.")
        return pd.read_csv(P["hist"])

    if os.path.exists(P["ck"]):
        ck = torch.load(P["ck"], map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        hist, best_dice, start_ep = ck["hist"], ck["best_dice"], ck["epoch"] + 1
        rng.setstate(ck["rng"])
        print(f">>> RESUMING training from epoch {start_ep} (best val dice so far {best_dice:.4f})")

    tr, va = split["train"], split["val"]
    vx, vy = make_val_patches(store, va, tc["patch"], tc["n_val_patches"], dev)
    t_start = time.time()
    for ep in range(start_ep, E + 1):
        model.train()
        tl, td = [], []
        for _ in range(tc["iters"]):
            x, y = train_batch(store, tr, tc["patch"], tc["batch"], tc["fg_prob"], rng, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                o = model(x)
            loss, main = loss_fn([t.float() for t in o], y)  # total (deep supervision) for backward
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
            tl.append(main.item())
            td.append(batch_dice(o[0].detach().float(), y))
        vl, vd = eval_val_patches(model, loss_fn, vx, vy, use_cuda)
        fd = fi = np.nan
        if ep % tc["full_val_every"] == 0 or ep == E:
            fd, fi, _ = evaluate_full(model, store, va, cfg, dev, use_cuda)
            if fd > best_dice:
                best_dice = fd
                torch.save(model.state_dict(), P["best"])
        hist.append(dict(epoch=ep, lr=opt.param_groups[0]["lr"], train_loss=np.mean(tl), train_dice=np.mean(td),
                         val_loss=vl, val_dice_patch=vd, val_dice_full=fd, val_iou_full=fi,
                         fusion_alpha=float(torch.sigmoid(model.alpha).item())))
        sched.step()
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                        hist=hist, best_dice=best_dice, epoch=ep, rng=rng.getstate()), P["ck"] + ".tmp")
        os.replace(P["ck"] + ".tmp", P["ck"])
        pd.DataFrame(hist).to_csv(P["hist"], index=False)
        print(f"Ep {ep:3d}/{E} | lr {hist[-1]['lr']:.1e} | train loss {np.mean(tl):.4f} dice {np.mean(td):.4f} | "
              f"val loss {vl:.4f} dice(patch) {vd:.4f} | alpha {hist[-1]['fusion_alpha']:.3f}"
              + (f" | VAL FULL dice {fd:.4f} iou {fi:.4f} *best {best_dice:.4f}" if not np.isnan(fd) else "")
              + f" | {(time.time() - t_start) / 60:.1f} min")
    torch.save(model.state_dict(), P["last"])
    if os.path.exists(P["ck"]):
        os.remove(P["ck"])
    H = pd.DataFrame(hist)
    H.to_csv(P["hist"], index=False)
    return H
