"""Shared pipeline setup: find cases -> preprocess/cache -> split -> store."""
import os
import random
import time

from ..data import CaseStore, cache_case, find_cases
from ..utils import load_json, save_json


def make_split(ids, frac, seed):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_te = max(1, round(frac[2] * n))
    n_va = max(1, round(frac[1] * n))
    return {"test": ids[:n_te], "val": ids[n_te:n_te + n_va], "train": ids[n_te + n_va:]}


def _cached(cache, c):
    return all(os.path.exists(f"{cache}/{c}_{s}") for s in ("x.npy", "y.npy", "sp.json", "fg.npy"))


def prepare(cfg, P, dev, verbose=True):
    sf = cfg["data"].get("split_file")
    if not sf and os.path.exists(P["split"]):  # resume / test-only: cache + split already there -> raw data not needed
        split = {k: [str(c) for c in v] for k, v in load_json(P["split"]).items()}
        ids = [c for k in ("train", "val", "test") for c in split[k]]
        if all(_cached(P["cache"], c) for c in ids):
            if verbose:
                print(f"Using cached data in {P['cache']} ({len(ids)} cases)")
            return None, split, CaseStore(P["cache"], ids)
    cases = find_cases(cfg)
    if verbose:
        print(f"Found {len(cases)} image/label pairs ({cfg['dim']}D)")
    t0 = time.time()
    for i, (cid, p) in enumerate(cases.items()):
        cache_case(cid, p, cfg, dev, P["cache"])
        if verbose and ((i + 1) % 5 == 0 or i == 0):
            print(f"  preprocessed {i + 1}/{len(cases)} ({time.time() - t0:.0f}s)")
    sf = cfg["data"].get("split_file")
    if sf:
        split = load_json(sf)
    elif os.path.exists(P["split"]):
        split = load_json(P["split"])
    else:
        split = make_split(cases.keys(), cfg["split"], cfg["seed"])
    split = {k: [str(c) for c in v] for k, v in split.items()}
    save_json(split, P["split"])
    if verbose:
        print(f"Split -> train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    store = CaseStore(P["cache"], [c for k in ("train", "val", "test") for c in split[k]])
    return cases, split, store
