"""Find (image, label) pairs on disk for 3D (NIfTI) and 2D (PNG/TIF/GIF/JPG ...) datasets."""
import glob
import os
import re


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _stem(fn):
    fn = os.path.basename(fn)
    for e in (".nii.gz", ".nii"):
        if fn.lower().endswith(e):
            return fn[: -len(e)]
    return os.path.splitext(fn)[0]


def _token(stem):
    return re.split(r"[_.\-]", stem)[0]


def _imagecas(root):
    cases = {}
    for d in os.listdir(root):
        m = re.match(r"(\d+)\.(img|label)\.nii$", d)
        if m and os.path.isdir(os.path.join(root, d)):
            fs = glob.glob(f"{root}/{d}/*.nii") + glob.glob(f"{root}/{d}/*.nii.gz")
            if fs:
                cases.setdefault(m.group(1), {})[m.group(2)] = fs[0]
    return {k: {"img": v["img"], "label": v["label"]} for k, v in cases.items() if "img" in v and "label" in v}


def _match(stem, labels):
    """labels: {stem: path}. exact -> label starts with image stem -> same leading token."""
    if stem in labels:
        return labels[stem]
    c = sorted(s for s in labels if s.startswith(stem))
    if c:
        return labels[c[0]]
    c = sorted(s for s in labels if _token(s) == _token(stem))
    return labels[c[0]] if c else None


def _flat(cfg):
    d, root = cfg["data"], cfg["data"]["root"]
    exts = tuple(e.lower() for e in d["image_exts"])
    lst = lambda sub: sorted((os.path.join(root, sub, f) for f in os.listdir(os.path.join(root, sub))
                              if f.lower().endswith(exts)), key=_natural)
    imgs, labs = lst(d["image_dir"]), {_stem(p): p for p in lst(d["label_dir"])}
    fovs = {_stem(p): p for p in lst(d["fov_dir"])} if d.get("fov_dir") else {}
    cases, missing = {}, []
    for p in imgs:
        st = _stem(p)
        lp = _match(st, labs)
        if lp is None:
            missing.append(st)
            continue
        cases[st] = {"img": p, "label": lp}
        if fovs:
            fp = _match(st, fovs)
            if fp:
                cases[st]["fov"] = fp
    if missing:
        print(f"WARNING: no label found for {len(missing)} images, e.g. {missing[:3]}")
    return cases


def find_cases(cfg):
    layout = cfg["data"]["layout"]
    cases = _imagecas(cfg["data"]["root"]) if layout == "imagecas" else _flat(cfg)
    if not cases:
        raise RuntimeError(f"No image/label pairs found under {cfg['data']['root']} (layout={layout}).")
    return {k: cases[k] for k in sorted(cases, key=_natural)}
