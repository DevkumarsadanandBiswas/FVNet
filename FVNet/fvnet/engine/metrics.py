"""Segmentation metrics for 2D / 3D binary masks (overlap, surface distance, topology, ROC/PR)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.metrics import average_precision_score, roc_auc_score


def structure(nd):
    return np.ones((3,) * nd)


def bbox(mask, margin=1):
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return None
    return tuple(slice(max(i.min() - margin, 0), i.max() + margin + 1) for i in idx)


def remove_small(pred, min_size):
    """Post-processing: drop connected components smaller than min_size."""
    lab, n = ndi.label(pred, structure=structure(pred.ndim))
    if n == 0:
        return pred
    sizes = ndi.sum(pred, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_size
    return keep[lab]


def cl_dice(pred, gt):
    bb = bbox(pred | gt, 2)
    if bb is None or pred.sum() == 0:
        return np.nan, np.nan, np.nan
    p, g = pred[bb], gt[bb]
    sp_, sg = skeletonize(p) > 0, skeletonize(g) > 0
    tp = (sp_ & g).sum() / (sp_.sum() + 1e-9)
    ts = (sg & p).sum() / (sg.sum() + 1e-9)
    return 2 * tp * ts / (tp + ts + 1e-9), tp, ts


def surf_dist(a, b, sp):
    bb = bbox(a | b, 2)
    if bb is None or a.sum() == 0 or b.sum() == 0:
        return None, None
    a, b = a[bb], b[bb]
    sa = a ^ ndi.binary_erosion(a)
    sb = b ^ ndi.binary_erosion(b)
    return ndi.distance_transform_edt(~sb, sampling=sp)[sa], ndi.distance_transform_edt(~sa, sampling=sp)[sb]


def compute_metrics(pred, gt, prob, sp, rng):
    TP = int((pred & gt).sum()); FP = int((pred & ~gt).sum()); FN = int((~pred & gt).sum())
    TN = int(pred.size - TP - FP - FN)
    e = 1e-9; N = float(pred.size); vox = float(np.prod(sp)); m = {}
    div = 1000.0 if pred.ndim == 3 else 1.0  # *_vol_mL: mL for 3D volumes, mm^2 for 2D images
    fTP, fFP, fFN, fTN = float(TP), float(FP), float(FN), float(TN)  # floats: avoid int overflow
    m["Dice"] = 2 * fTP / (2 * fTP + fFP + fFN + e)
    m["IoU"] = fTP / (fTP + fFP + fFN + e)
    m["Precision"] = fTP / (fTP + fFP + e)
    m["Recall(Sensitivity)"] = fTP / (fTP + fFN + e)
    m["Specificity"] = fTN / (fTN + fFP + e)
    m["NPV"] = fTN / (fTN + fFN + e)
    m["Accuracy"] = (fTP + fTN) / N
    m["Balanced_Acc"] = (m["Recall(Sensitivity)"] + m["Specificity"]) / 2
    m["F2"] = 5 * fTP / (5 * fTP + 4 * fFN + fFP + e)
    m["FPR"] = fFP / (fFP + fTN + e)
    m["FNR"] = fFN / (fFN + fTP + e)
    den = np.sqrt((fTP + fFP) * (fTP + fFN) * (fTN + fFP) * (fTN + fFN))
    m["MCC"] = float((fTP * fTN - fFP * fFN) / (den + e))
    po = (fTP + fTN) / N
    pe = ((fTP + fFP) * (fTP + fFN) + (fFN + fTN) * (fFP + fTN)) / N ** 2
    m["Kappa"] = float((po - pe) / (1 - pe + e))
    m["VolSimilarity"] = 1 - abs(fFN - fFP) / (2 * fTP + fFP + fFN + e)
    m["RVD_%"] = (float(pred.sum()) - float(gt.sum())) / (float(gt.sum()) + e) * 100
    d1, d2 = surf_dist(pred, gt, sp)
    if d1 is None:
        m.update({"HD_mm": np.nan, "HD95_mm": np.nan, "ASSD_mm": np.nan, "ASD_pred2gt_mm": np.nan})
    else:
        m["HD_mm"] = max(d1.max(), d2.max())
        m["HD95_mm"] = max(np.percentile(d1, 95), np.percentile(d2, 95))
        m["ASSD_mm"] = np.concatenate([d1, d2]).mean()
        m["ASD_pred2gt_mm"] = d1.mean()
    m["clDice"], m["clPrecision"], m["clRecall"] = cl_dice(pred, gt)
    st = structure(pred.ndim)
    m["Pred_components"] = ndi.label(pred, structure=st)[1]
    m["GT_components"] = ndi.label(gt, structure=st)[1]
    pos = np.flatnonzero(gt.ravel())
    ri = rng.integers(0, gt.size, 600000)
    neg = ri[~gt.ravel()[ri]]
    ys = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ps = np.concatenate([prob.ravel()[pos], prob.ravel()[neg]]).astype(np.float32)
    ws = np.concatenate([np.ones(len(pos)), np.full(len(neg), (gt.size - len(pos)) / max(len(neg), 1))])
    m["AUC_ROC"] = roc_auc_score(ys, ps, sample_weight=ws)
    m["AvgPrecision(AP)"] = average_precision_score(ys, ps, sample_weight=ws)
    m.update(TP=TP, FP=FP, FN=FN, TN=TN, GT_vol_mL=float(gt.sum()) * vox / div, Pred_vol_mL=float(pred.sum()) * vox / div)
    return m, (ys, ps, ws)
