"""Figures: dataset / preprocessing / training curves / test evaluation plots and 2D comparison images."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from .utils import slug  # noqa: E402

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass


def render(plots, prefix, P, ncols=3):
    """Save a combined grid (<prefix>_ALL.png) and every plot individually."""
    nr = int(np.ceil(len(plots) / ncols))
    fig = plt.figure(figsize=(6 * ncols, 4.8 * nr))
    for i, (name, fn, polar) in enumerate(plots):
        ax = fig.add_subplot(nr, ncols, i + 1, projection="polar" if polar else None)
        fn(ax)
        ax.set_title(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{P['fig']}/{prefix}_ALL.png", dpi=130)
    plt.close(fig)
    for name, fn, polar in plots:
        f2 = plt.figure(figsize=(6.5, 5))
        a = f2.add_subplot(111, projection="polar" if polar else None)
        fn(a)
        a.set_title(name)
        f2.tight_layout()
        f2.savefig(f"{P['ind']}/{prefix}_{slug(name)}.png", dpi=150)
        plt.close(f2)


# ------------------------------------------------------------------ dataset / preprocessing
def draw_dataset(store, split, P):
    ids = [c for k in ("train", "val", "test") for c in split[k]]
    unit = "mL" if len(store.Y[ids[0]].shape) == 3 else "mm^2"
    scale = 1000 if unit == "mL" else 1

    def a1(ax):
        ax.bar(["Train", "Val", "Test"], [len(split[k]) for k in ("train", "val", "test")], color=["#4c72b0", "#dd8452", "#55a868"])
        ax.set_ylabel("cases")

    def a2(ax):
        ax.hist([float(np.mean(store.Y[c])) * 100 for c in ids], bins=15, color="#8172b3")
        ax.set_xlabel("vessel pixels (% of image)"); ax.set_ylabel("cases")

    def a3(ax):
        ax.bar(range(len(ids)), [float(np.sum(store.Y[c])) * np.prod(store.SP[c]) / scale for c in ids], color="#c44e52")
        ax.set_xlabel("case index"); ax.set_ylabel(f"vessel size ({unit})")

    render([("Split sizes", a1, False), ("Vessel fraction", a2, False), ("Vessel size per case", a3, False)], "00_dataset", P)


def draw_preproc(store, cid, P):
    X, Y = store.X[cid], store.Y[cid]
    names = ["Raw (windowed)", "Vesselness", "Enhanced (bg suppressed)", "GT mask"]
    if Y.ndim == 3:
        z = np.where(np.asarray(Y).sum((0, 1)) > 0)[0]
        zs = z[np.linspace(0, len(z) - 1, 5)[1:-1].astype(int)]
        rows = [[np.asarray(X[2][..., k]), np.asarray(X[1][..., k]), np.asarray(X[0][..., k]), np.asarray(Y[..., k])] for k in zs]
        tags = [f"z{k}" for k in zs]
    else:
        rows = [[np.asarray(X[2]), np.asarray(X[1]), np.asarray(X[0]), np.asarray(Y)]]
        tags = ["img"]
    fig, axs = plt.subplots(len(rows), 4, figsize=(14, 3.4 * len(rows)), squeeze=False)
    for r, ims in enumerate(rows):
        for c, im in enumerate(ims):
            axs[r, c].imshow(im.astype(np.float32).T if Y.ndim == 3 else im.astype(np.float32), cmap="gray", origin="lower" if Y.ndim == 3 else "upper")
            axs[r, c].axis("off")
            if r == 0:
                axs[r, c].set_title(names[c])
            plt.imsave(f"{P['ind']}/preproc_case{cid}_{tags[r]}_{slug(names[c])}.png",
                       im.astype(np.float32).T if Y.ndim == 3 else im.astype(np.float32), cmap="gray")
    fig.tight_layout()
    fig.savefig(f"{P['fig']}/01_preprocessing_case{cid}.png", dpi=130)
    plt.close(fig)


def draw_train(H, P):
    e = H.epoch
    fv = H.dropna(subset=["val_dice_full"])

    def p1(ax):
        ax.plot(e, H.train_loss, label="train"); ax.plot(e, H.val_loss, label="val"); ax.set_xlabel("epoch"); ax.set_ylabel("compound loss"); ax.legend()

    def p2(ax):
        ax.plot(e, H.train_dice, label="train"); ax.plot(e, H.val_dice_patch, label="val (patches)"); ax.set_xlabel("epoch"); ax.set_ylabel("Dice"); ax.legend()

    def p3(ax):
        ax.plot(fv.epoch, fv.val_dice_full, "o-", label="Dice"); ax.plot(fv.epoch, fv.val_iou_full, "s-", label="IoU")
        ax.set_xlabel("epoch"); ax.set_title("Val full-volume"); ax.legend()

    def p4(ax):
        ax.plot(e, H.lr, color="k"); ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")

    def p5(ax):
        ax.plot(e, H.val_loss - H.train_loss, color="crimson"); ax.axhline(0, color="k", lw=.7); ax.set_xlabel("epoch"); ax.set_ylabel("val - train loss")

    def p6(ax):
        if "fusion_alpha" in H:
            ax.plot(e, H.fusion_alpha, color="purple"); ax.set_ylabel("fusion alpha (vessel weight)"); ax.set_ylim(0, 1)
        ax.set_xlabel("epoch")

    render([("Loss curve (train vs val)", p1, False), ("Dice curve (train vs val)", p2, False), ("Full-volume validation metrics", p3, False),
            ("Learning rate", p4, False), ("Generalization gap", p5, False), ("Adaptive fusion alpha", p6, False)], "02_training", P)


def draw_threshold(ths, vdice, thr, P):
    fig = plt.figure(figsize=(6, 4.5))
    plt.plot(ths, vdice, "o-"); plt.axvline(thr, color="r", ls="--"); plt.xlabel("threshold"); plt.ylabel("val Dice")
    plt.title("Threshold sweep (validation)"); plt.tight_layout()
    plt.savefig(f"{P['fig']}/03_threshold_sweep_val.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ test evaluation
def draw_test(R, SAMPLES, TEST, thr, P):
    Ys = np.concatenate([SAMPLES[c][0] for c in TEST]); Ps = np.concatenate([SAMPLES[c][1] for c in TEST]); Ws = np.concatenate([SAMPLES[c][2] for c in TEST])
    cs = [str(c) for c in R.index]
    core = ["Dice", "IoU", "Precision", "Recall(Sensitivity)", "Specificity", "clDice", "Balanced_Acc", "VolSimilarity"]

    def t_box(ax):
        ax.boxplot([R[k].dropna() for k in core], showmeans=True)
        ax.set_xticks(range(1, len(core) + 1)); ax.set_xticklabels([k.split("(")[0] for k in core])
        for i, k in enumerate(core):
            ax.scatter(np.random.normal(i + 1, .05, R[k].notna().sum()), R[k].dropna(), s=14, alpha=.6, c="k")
        ax.tick_params(axis="x", rotation=45)

    def t_bar(ax):
        w = .27; xs = np.arange(len(cs))
        for j, k in enumerate(["Dice", "IoU", "clDice"]):
            ax.bar(xs + (j - 1) * w, R[k], w, label=k)
        ax.set_xticks(xs); ax.set_xticklabels(cs); ax.set_xlabel("test case"); ax.legend(); ax.set_ylim(0, 1)

    def t_dist(ax):
        xs = np.arange(len(cs))
        ax.bar(xs - .2, R["HD95_mm"], .4, label="HD95 (mm)"); ax.bar(xs + .2, R["ASSD_mm"], .4, label="ASSD (mm)")
        ax.set_xticks(xs); ax.set_xticklabels(cs); ax.set_xlabel("test case"); ax.legend()

    def t_hist(ax):
        ax.hist(R["Dice"], bins=10, color="#4c72b0", alpha=.8)
        ax.axvline(R["Dice"].mean(), color="r", ls="--", label=f"mean {R['Dice'].mean():.3f}"); ax.set_xlabel("Dice"); ax.set_ylabel("cases"); ax.legend()

    def t_cdf(ax):
        for k in ["Dice", "IoU", "clDice"]:
            v = np.sort(R[k].dropna()); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=k)
        ax.set_xlabel("score"); ax.set_ylabel("cumulative fraction of test cases"); ax.legend()

    def t_case(ax):
        o = R["Dice"].sort_values()
        ax.plot(range(len(o)), o.values, "o-"); ax.set_xticks(range(len(o))); ax.set_xticklabels(o.index.astype(str))
        ax.set_xlabel("test case (sorted)"); ax.set_ylabel("Dice")

    def t_roc(ax):
        for c in TEST:
            f, t, _ = roc_curve(SAMPLES[c][0], SAMPLES[c][1], sample_weight=SAMPLES[c][2]); ax.plot(f, t, alpha=.25, color="gray")
        f, t, _ = roc_curve(Ys, Ps, sample_weight=Ws)
        ax.plot(f, t, "r", lw=2, label=f"pooled AUC={roc_auc_score(Ys, Ps, sample_weight=Ws):.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=.7); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()

    def t_pr(ax):
        for c in TEST:
            p, r, _ = precision_recall_curve(SAMPLES[c][0], SAMPLES[c][1], sample_weight=SAMPLES[c][2]); ax.plot(r, p, alpha=.25, color="gray")
        p, r, _ = precision_recall_curve(Ys, Ps, sample_weight=Ws)
        ax.plot(r, p, "r", lw=2, label=f"pooled AP={average_precision_score(Ys, Ps, sample_weight=Ws):.4f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()

    def t_cm(ax):
        tp, fp, fn, tn = [float(R[k].sum()) for k in ("TP", "FP", "FN", "TN")]
        cm = np.array([[tn, fp], [fn, tp]], float); cn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cn[i, j] * 100:.2f}%\n({int(cm[i, j]):,})", ha="center", va="center", color="w" if cn[i, j] > .5 else "k")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred BG", "Pred Vessel"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["GT BG", "GT Vessel"]); ax.grid(False)

    def t_vol(ax):
        ax.scatter(R.GT_vol_mL, R.Pred_vol_mL, c="C0")
        lim = [0, max(R.GT_vol_mL.max(), R.Pred_vol_mL.max()) * 1.1]; ax.plot(lim, lim, "k--")
        r = np.corrcoef(R.GT_vol_mL, R.Pred_vol_mL)[0, 1] if len(R) > 1 else np.nan
        ax.set_title(f"Volume agreement (r={r:.3f})"); ax.set_xlabel("GT size"); ax.set_ylabel("Pred size")

    def t_ba(ax):
        mn = (R.GT_vol_mL + R.Pred_vol_mL) / 2; df = R.Pred_vol_mL - R.GT_vol_mL
        ax.scatter(mn, df); ax.axhline(df.mean(), color="r")
        ax.axhline(df.mean() + 1.96 * df.std(), color="r", ls="--"); ax.axhline(df.mean() - 1.96 * df.std(), color="r", ls="--")
        ax.set_xlabel("mean size"); ax.set_ylabel("Pred - GT")

    def t_dv(ax):
        ax.scatter(R.GT_vol_mL, R.Dice); ax.set_xlabel("GT vessel size"); ax.set_ylabel("Dice")

    def t_radar(ax):
        ks = ["Dice", "IoU", "Precision", "Recall(Sensitivity)", "Specificity", "clDice", "Balanced_Acc", "VolSimilarity", "MCC"]
        v = [R[k].mean() for k in ks]
        a = np.linspace(0, 2 * np.pi, len(ks), endpoint=False); a = np.r_[a, a[0]]; v = v + v[:1]
        ax.plot(a, v, "o-"); ax.fill(a, v, alpha=.25)
        ax.set_xticks(a[:-1]); ax.set_xticklabels([k.split("(")[0] for k in ks], fontsize=8); ax.set_ylim(0, 1)

    def t_rel(ax):
        b = np.linspace(0, 1, 11); ix = np.clip(np.digitize(Ps, b) - 1, 0, 9); mp, my = [], []
        for i in range(10):
            s = ix == i
            if Ws[s].sum() > 0:
                mp.append(np.average(Ps[s], weights=Ws[s])); my.append(np.average(Ys[s], weights=Ws[s]))
        ax.plot(mp, my, "o-"); ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("mean predicted prob"); ax.set_ylabel("observed vessel fraction")

    def t_time(ax):
        ax.bar(cs, R.Infer_time_s, color="#55a868"); ax.set_xlabel("test case"); ax.set_ylabel("inference time (s)")

    def t_corr(ax):
        ks = ["Dice", "IoU", "Precision", "Recall(Sensitivity)", "Specificity", "clDice", "HD95_mm", "ASSD_mm", "AUC_ROC", "GT_vol_mL"]
        cm_ = R[ks].corr().values
        im = ax.imshow(cm_, cmap="coolwarm", vmin=-1, vmax=1); ax.set_xticks(range(len(ks))); ax.set_yticks(range(len(ks)))
        ax.set_xticklabels([k.split("(")[0] for k in ks], rotation=90, fontsize=7); ax.set_yticklabels([k.split("(")[0] for k in ks], fontsize=7)
        ax.grid(False); plt.colorbar(im, ax=ax, fraction=.046)

    def t_phist(ax):
        ax.hist(Ps[Ys == 1], bins=40, weights=Ws[Ys == 1], alpha=.6, label="vessel", density=True)
        ax.hist(Ps[Ys == 0], bins=40, weights=Ws[Ys == 0], alpha=.6, label="background", density=True)
        ax.set_yscale("log"); ax.set_xlabel("predicted prob"); ax.legend()

    def t_thr(ax):
        tt = np.linspace(.05, .95, 19); d, p, r = [], [], []
        for t in tt:
            pr = Ps >= t
            tp = (Ws * pr * (Ys == 1)).sum(); fp = (Ws * pr * (Ys == 0)).sum(); fn = (Ws * (~pr) * (Ys == 1)).sum()
            d.append(2 * tp / (2 * tp + fp + fn + 1e-9)); p.append(tp / (tp + fp + 1e-9)); r.append(tp / (tp + fn + 1e-9))
        ax.plot(tt, d, label="Dice"); ax.plot(tt, p, label="Precision"); ax.plot(tt, r, label="Recall")
        ax.axvline(thr, color="k", ls="--"); ax.set_xlabel("threshold"); ax.legend()

    def t_rvd(ax):
        ax.bar(cs, R["RVD_%"], color=np.where(R["RVD_%"] > 0, "#c44e52", "#4c72b0")); ax.axhline(0, color="k")
        ax.set_xlabel("test case"); ax.set_ylabel("relative volume diff (%)")

    render([("Overlap metrics (boxplot)", t_box, False), ("Per-case Dice-IoU-clDice", t_bar, False), ("Surface distances per case", t_dist, False),
            ("Dice histogram", t_hist, False), ("Cumulative distribution of scores", t_cdf, False), ("Inference Dice per case", t_case, False),
            ("ROC curve", t_roc, False), ("Precision-Recall curve", t_pr, False), ("Confusion matrix (pooled)", t_cm, False),
            ("Predicted vs GT volume", t_vol, False), ("Bland-Altman volume", t_ba, False), ("Dice vs vessel volume", t_dv, False),
            ("Mean metric radar", t_radar, True), ("Reliability diagram", t_rel, False), ("Inference time", t_time, False),
            ("Metric correlation", t_corr, False), ("Probability distribution", t_phist, False), ("Test threshold sweep", t_thr, False),
            ("Relative volume difference", t_rvd, False)], "04_test_evaluation", P, ncols=4)


# ------------------------------------------------------------------ comparison images
def overlay(base, layers, a=0.6):
    rgb = np.stack([base] * 3, -1).astype(np.float32)
    for m, col in layers:
        rgb[m] = (1 - a) * rgb[m] + a * np.array(col)
    return rgb


def _save_img(arr, path, cmap=None, up=3, origin="lower"):
    arr = np.repeat(np.repeat(arr, up, 0), up, 1)
    plt.imsave(path, arr, cmap=cmap, origin=origin, vmin=0 if cmap else None, vmax=1 if cmap else None)


COLS = ["Raw", "Enhanced", "Ground Truth", "Prediction", "Probability", "GT overlay", "Pred overlay", "TP/FP/FN overlap"]


def _panels(rw, enh, g, p_, prob):
    return [rw, enh, g.astype(np.float32), p_.astype(np.float32), prob,
            overlay(rw, [(g, (0, 1, 0))]), overlay(rw, [(p_, (1, 0, 0))]),
            overlay(rw, [(g & p_, (1, 1, 0)), (p_ & ~g, (1, 0, 0)), (g & ~p_, (0, .5, 1))], .75)]


def _legend(fig):
    fig.legend(handles=[Patch(color="yellow", label="TP"), Patch(color="red", label="FP"), Patch(color="dodgerblue", label="FN")],
               loc="lower center", ncol=3)


def make_comparison(cid, store, pred, prob, R, P, n_slices=5):
    """3D: n_slices axial slices x 8 panels + axial/coronal MIPs.  2D: one row of 8 panels."""
    d = f"{P['cmp']}/case_{cid}"
    os.makedirs(d, exist_ok=True)
    gt = store.y(cid)
    prob = prob.astype(np.float32)
    raw, enh = np.asarray(store.X[cid][2]).astype(np.float32), np.asarray(store.X[cid][0]).astype(np.float32)
    title = (f"Test case {cid} | Dice {R.loc[cid, 'Dice']:.3f}  IoU {R.loc[cid, 'IoU']:.3f}  "
             f"clDice {R.loc[cid, 'clDice']:.3f}  HD95 {R.loc[cid, 'HD95_mm']:.1f}mm")
    if gt.ndim == 2:
        panels = _panels(raw, enh, gt, pred, prob)
        fig, axs = plt.subplots(1, len(COLS), figsize=(2.9 * len(COLS), 3.4))
        for j, im in enumerate(panels):
            rgb = im.ndim == 3
            axs[j].imshow(im, **({} if rgb else dict(cmap="magma" if j == 4 else "gray", vmin=0, vmax=1)))
            axs[j].axis("off"); axs[j].set_title(COLS[j], fontsize=9)
            _save_img(im, f"{d}/{slug(COLS[j])}.png", None if rgb else ("magma" if j == 4 else "gray"), up=1, origin="upper")
        _legend(fig); fig.suptitle(title)
        fig.tight_layout(rect=[0, .06, 1, .94]); fig.savefig(f"{P['cmp']}/case_{cid}_comparison.png", dpi=130); plt.close(fig)
        return
    zs = np.where(gt.sum((0, 1)) > 0)[0]
    sel = zs[np.linspace(0, len(zs) - 1, n_slices + 2)[1:-1].astype(int)]
    u = (gt | pred).any(2)
    r, c = np.where(u.any(1))[0], np.where(u.any(0))[0]
    x0, x1, y0, y1 = max(r[0] - 15, 0), r[-1] + 15, max(c[0] - 15, 0), c[-1] + 15
    T = lambda a, k: a[x0:x1, y0:y1, k].T
    fig, axs = plt.subplots(len(sel), len(COLS), figsize=(2.6 * len(COLS), 2.6 * len(sel)))
    for i, k in enumerate(sel):
        panels = _panels(T(raw, k), T(enh, k), T(gt, k), T(pred, k), T(prob, k))
        for j, im in enumerate(panels):
            a = axs[i, j]; rgb = im.ndim == 3
            a.imshow(im, origin="lower", **({} if rgb else dict(cmap="magma" if j == 4 else "gray", vmin=0, vmax=1)))
            a.axis("off")
            if i == 0:
                a.set_title(COLS[j], fontsize=9)
            _save_img(im, f"{d}/z{k:03d}_{slug(COLS[j])}.png", None if rgb else ("magma" if j == 4 else "gray"))
        axs[i, 0].text(-0.05, .5, f"z={k}", transform=axs[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    _legend(fig); fig.suptitle(title, y=.995)
    fig.tight_layout(rect=[0, .03, 1, .98]); fig.savefig(f"{P['cmp']}/case_{cid}_comparison.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    for i, ax_ in enumerate((2, 1)):
        g, p_ = gt.any(ax_), pred.any(ax_)
        ov = overlay(np.zeros(g.shape, np.float32), [(g & p_, (1, 1, 0)), (p_ & ~g, (1, 0, 0)), (g & ~p_, (0, .5, 1))], 1.0)
        for j, (im, nm) in enumerate([(g.astype(np.float32), "GT MIP"), (p_.astype(np.float32), "Pred MIP"), (ov, "Overlap MIP")]):
            im2 = im.T if im.ndim == 2 else im.transpose(1, 0, 2)
            axs[i, j].imshow(im2, origin="lower", **({} if im.ndim == 3 else dict(cmap="gray"))); axs[i, j].axis("off")
            axs[i, j].set_title(f"{nm} ({'axial' if ax_ == 2 else 'coronal'})")
            plt.imsave(f"{d}/MIP_{slug(nm)}_{'axial' if ax_ == 2 else 'coronal'}.png", np.repeat(np.repeat(im2, 2, 0), 2, 1),
                       origin="lower", cmap=None if im.ndim == 3 else "gray")
    fig.tight_layout(); fig.savefig(f"{P['cmp']}/case_{cid}_MIP.png", dpi=130); plt.close(fig)
