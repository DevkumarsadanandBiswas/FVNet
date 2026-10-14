"""Compound training loss  L = l1*Dice + l2*BCE + l3*Tversky + l4*Centerline(clDice)  (2D / 3D generic)."""
import torch
import torch.nn.functional as F


def _red(y):
    return tuple(range(1, y.ndim))


def dice_loss(lg, y, eps=1e-5):
    p = torch.sigmoid(lg)
    d = _red(y)
    return 1 - ((2 * (p * y).sum(d) + eps) / (p.sum(d) + y.sum(d) + eps)).mean()


def tversky_loss(lg, y, a=0.3, b=0.7, eps=1e-5):
    """a weights false positives, b weights false negatives (recall-favouring for thin vessels)."""
    p = torch.sigmoid(lg)
    d = _red(y)
    tp = (p * y).sum(d)
    fp = (p * (1 - y)).sum(d)
    fn = ((1 - p) * y).sum(d)
    return 1 - ((tp + eps) / (tp + a * fp + b * fn + eps)).mean()


def _pool(x, k, pad):
    return (F.max_pool2d if x.ndim == 4 else F.max_pool3d)(x, k, 1, pad)


def soft_erode(x):
    nd = x.ndim - 2
    outs = []
    for ax in range(nd):
        k = [1] * nd
        pd = [0] * nd
        k[ax], pd[ax] = 3, 1
        outs.append(-_pool(-x, tuple(k), tuple(pd)))
    out = outs[0]
    for o in outs[1:]:
        out = torch.min(out, o)
    return out


def soft_dilate(x):
    return _pool(x, 3, 1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skel(x, iters=4):
    sk = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        sk = sk + F.relu(delta - sk * delta)
    return sk


def cl_loss(lg, y, iters=4, eps=1e-5):
    """Soft clDice: centerline / topology preserving loss."""
    p = torch.sigmoid(lg)
    d = _red(y)
    sp_, sg = soft_skel(p, iters), soft_skel(y, iters)
    tprec = ((sp_ * y).sum(d) + eps) / (sp_.sum(d) + eps)
    tsens = ((sg * p).sum(d) + eps) / (sg.sum(d) + eps)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


class FVNetLoss:
    """Compound loss on the fused output + deep supervision on the auxiliary outputs.

    ``__call__(outputs, y)``: training (list of 5 logits) -> (total, main); eval (tensor) -> main compound loss.
    """

    def __init__(self, lambdas, tv_alpha=0.3, tv_beta=0.7, skel_iters=4, ds_weights=None):
        self.l = lambdas
        self.a, self.b, self.it = tv_alpha, tv_beta, skel_iters
        self.ds = ds_weights or dict(vessel=0.4, fg=0.3, aux3=0.2, aux2=0.1)

    def compound(self, lg, y):
        return (self.l["dice"] * dice_loss(lg, y) + self.l["bce"] * F.binary_cross_entropy_with_logits(lg, y)
                + self.l["tversky"] * tversky_loss(lg, y, self.a, self.b) + self.l["cl"] * cl_loss(lg, y, self.it))

    @staticmethod
    def simple(lg, y):
        return 0.5 * F.binary_cross_entropy_with_logits(lg, y) + dice_loss(lg, y)

    def __call__(self, o, y):
        if isinstance(o, (list, tuple)):
            main = self.compound(o[0], y)
            tot = (main + self.ds["vessel"] * self.simple(o[1], y) + self.ds["fg"] * self.simple(o[2], y)
                   + self.ds["aux3"] * self.simple(o[3], y) + self.ds["aux2"] * self.simple(o[4], y))
            return tot, main
        return self.compound(o, y)


def batch_dice(lg, y, eps=1e-5):
    p = (torch.sigmoid(lg) > 0.5).float()
    d = _red(y)
    return ((2 * (p * y).sum(d) + eps) / (p.sum(d) + y.sum(d) + eps)).mean().item()
