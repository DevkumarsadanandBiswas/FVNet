"""Run with:  pytest -q   (CPU, a few seconds per test)."""
import subprocess
import sys

import numpy as np
import pytest
import torch

from fvnet.engine.metrics import compute_metrics, remove_small
from fvnet.losses import FVNetLoss
from fvnet.models import ForegroundVesselNet


@pytest.mark.parametrize("nd,shape", [(2, (64, 64)), (2, (50, 70)), (3, (32, 32, 32)), (3, (24, 40, 33))])
def test_forward_shapes(nd, shape):
    m = ForegroundVesselNet(nd, 3, (8, 16, 32, 64))
    x = torch.randn(1, 3, *shape)
    m.train()
    outs = m(x)
    assert len(outs) == 5 and all(o.shape == (1, 1, *shape) for o in outs)
    m.eval()
    assert m(x).shape == (1, 1, *shape)


@pytest.mark.parametrize("nd,shape", [(2, (64, 64)), (3, (32, 32, 32))])
def test_loss_backward(nd, shape):
    m = ForegroundVesselNet(nd, 3, (8, 16, 32, 64)).train()
    x = torch.randn(2, 3, *shape)
    y = (torch.rand(2, 1, *shape) > 0.9).float()
    L = FVNetLoss(dict(dice=1.0, bce=0.5, tversky=0.5, cl=0.3))
    tot, main = L(m(x), y)
    tot.backward()
    assert torch.isfinite(tot) and torch.isfinite(main)
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_metrics_and_cc():
    rng = np.random.default_rng(0)
    gt = np.zeros((40, 40), bool); gt[10:30, 20] = True
    pred = gt.copy(); pred[5, 5] = True
    assert remove_small(pred, 3).sum() == gt.sum()
    m, _ = compute_metrics(remove_small(pred, 3), gt, gt.astype(np.float32) * 0.8 + 0.1, [1.0, 1.0], rng)
    assert m["Dice"] > 0.99 and m["clDice"] > 0.99


@pytest.mark.parametrize("dim", [2, 3])
def test_end_to_end_smoke(dim, tmp_path):
    d = tmp_path / "data"
    subprocess.check_call([sys.executable, "scripts/make_synthetic_data.py", "--dim", str(dim), "--out", str(d), "--n", "8"])
    o = ["--set", f"data.root={d}", f"out={tmp_path}/out", f"data.cache={tmp_path}/cache"]
    subprocess.check_call([sys.executable, "scripts/run_all.py", "-c", f"configs/smoke_{dim}d.yaml", *o])
    assert (tmp_path / "out" / "test_metrics_summary.csv").exists()
