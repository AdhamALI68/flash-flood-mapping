#!/usr/bin/env python3
"""
The model. Deliberately small and boring.

NEXT_STEPS 6b, from measurement not preference:
  - trial 1's 961k-parameter U-Net BEAT a 24.5M ResNet-34 U-Net with ImageNet
    pretraining, 0.7736 against 0.7580. Bigger was worse
  - no transformer: without convolutional inductive bias it needs far more data
    to learn spatial structure, and at this size would underfit or memorise
  - no KAN: already published for flood mapping (GMKAT, FloodKAN) so not novel,
    and measured >100x slower to train on identical hardware
  - the novelty is the dataset and the experiments, not the architecture. A
    trendy layer invites "what is actually new here?"

Two carried-forward changes are built into the architecture:

  ELEVATION PRIOR IN THE OUTPUT  (+0.014 in trial 1)
      logit = unet(x) + softplus(w) * (-z)
  Water flows downhill. Rather than making the network rediscover that from
  scratch, it is written into the output as a learned-strength term. softplus
  keeps the weight positive, so the prior can only ever say "lower floods more",
  never invert. The network learns the residual.

  HYBRID MULTI-SCALE HEAD  (+0.037, the largest architectural gain)
      the head sees the finest decoder features AND upsampled coarse features
  Flood extent depends on both local micro-relief and basin-scale context, and a
  single-scale head has to trade them off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class FloodUNet(nn.Module):
    """Small U-Net with an elevation prior and a multi-scale head.

    elev_index points at the channel holding mean elevation. It is used twice:
    as an ordinary input channel, and in the output prior. That is not
    double-counting - the prior fixes the SIGN of the relationship a priori,
    which is information the channel alone does not carry.
    """

    def __init__(self, in_ch: int, elev_index: int | None = None,
                 base: int = 24, depth: int = 4, prior: str = "downhill"):
        """prior: how the elevation term is allowed to act.

        "downhill"  softplus(w) * (-z). Can only say LOWER ground floods more.
                    This is what trial 1 used, where it was worth +0.014 because
                    elevation there scored 0.7790.
        "free"      w * (-z), sign learned. Same term, no assumption baked in.
        "off"       no elevation term at all.

        WHY THIS IS NOT JUST A KNOB
            On this dataset elevation scores 0.4804 at pixel level - BELOW
            chance. A prior locked to "lower floods more" therefore pushes every
            pixel the wrong way and cannot switch itself off, because softplus
            is strictly positive. The assumption that held in trial 1 does not
            hold here, so it has to be testable rather than assumed.
        """
        super().__init__()
        if prior not in ("downhill", "free", "off"):
            raise ValueError(f"unknown prior {prior!r}")
        self.prior = prior if elev_index is not None else "off"
        self.elev_index = elev_index if self.prior != "off" else None
        chs = [base * (2 ** i) for i in range(depth)]

        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.enc.append(conv_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(chs[-1], chs[-1] * 2)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = chs[-1] * 2
        for ch in reversed(chs):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(conv_block(ch * 2, ch))
            c = ch

        # Multi-scale head: finest decoder features plus every coarser stage,
        # upsampled. Lets the head use local detail and basin context together.
        self.head = nn.Conv2d(sum(chs), 1, 1)

        if self.prior != "off":
            self.prior_w = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for blk in self.enc:
            h = blk(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)

        feats = []
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = dec(torch.cat([h, skip], dim=1))
            feats.append(h)

        target = feats[-1].shape[-2:]
        multi = [f if f.shape[-2:] == target
                 else F.interpolate(f, size=target, mode="bilinear", align_corners=False)
                 for f in feats]
        logit = self.head(torch.cat(multi, dim=1))

        if self.prior != "off":
            z = x[:, self.elev_index:self.elev_index + 1]
            w = F.softplus(self.prior_w) if self.prior == "downhill" else self.prior_w
            logit = logit + w * (-z)
        return logit


def masked_bce(logit, target, mask, pos_weight: float = 1.0):
    """BCE over valid pixels only. Fabricated pixels must not train the model.

    pos_weight multiplies the loss on flooded pixels. Only 5-10% of pixels flood,
    so with equal weighting the cheapest way to reduce the loss is to predict
    "dry" everywhere - which is exactly what the model does, outputting
    probabilities that never reach 0.5. Raising this trades precision for recall
    at the level of what the model LEARNS, not just where the cutoff is put.
    """
    pw = (logit.new_tensor(pos_weight) if pos_weight != 1.0 else None)
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none",
                                           pos_weight=pw)
    m = mask.float()
    return (l * m).sum() / m.sum().clamp(min=1)


def masked_focal(logit, target, mask, gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss: down-weights pixels the model already gets right.

    With 5% of pixels flooded, the easy dry majority dominates the gradient even
    under weighted BCE. Focal loss shrinks the contribution of confident correct
    pixels by (1-p)^gamma, so what is left is the hard boundary between wet and
    dry - which is exactly the part we are failing at.
    """
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    l = a_t * (1 - p_t).pow(gamma) * ce
    mk = mask.float()
    return (l * mk).sum() / mk.sum().clamp(min=1)


def masked_bce_weighted(logit, target, mask, weight):
    """BCE where each PIXEL carries its own confidence weight.

    Filtering whole events by label quality failed. This is the finer version:
    a pixel whose coherence drop is small relative to its own measurement noise
    contributes less to the loss than one with a large, unambiguous drop. The
    label is not changed, only how much each pixel is allowed to teach.
    """
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    w = mask.float() * weight
    return (l * w).sum() / w.sum().clamp(min=1)


def within_tile_ranking_loss(logit, target, mask, n_pairs: int = 512):
    """Soft surrogate for WITHIN-TILE AUC (+0.024 in trial 1).

    The reported metric is within-tile AUC, which depends only on the ORDER of
    pixels inside a tile. BCE optimises calibration instead, which is a different
    objective - a model can improve its BCE while getting the ordering worse.

    Sampling pairs strictly inside each tile is what makes this match the metric:
    pairs drawn across tiles would reward getting tile-level wetness right, the
    exact failure that makes pooled AUC misleading.
    """
    B = logit.shape[0]
    total, n = logit.new_zeros(()), 0
    for b in range(B):
        m = mask[b].reshape(-1) > 0.5
        if m.sum() < 20:
            continue
        s = logit[b].reshape(-1)[m]
        y = target[b].reshape(-1)[m] > 0.5
        if y.sum() < 2 or (~y).sum() < 2:
            continue          # no ordering to learn in this tile
        pi = torch.nonzero(y, as_tuple=True)[0]
        ni = torch.nonzero(~y, as_tuple=True)[0]
        k = min(n_pairs, pi.numel() * ni.numel())
        a = pi[torch.randint(pi.numel(), (k,), device=s.device)]
        c = ni[torch.randint(ni.numel(), (k,), device=s.device)]
        # softplus(neg - pos): zero when positives already rank above negatives
        total = total + F.softplus(s[c] - s[a]).mean()
        n += 1
    return total / max(n, 1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for base in (16, 24, 32):
        m = FloodUNet(in_ch=20, elev_index=0, base=base)
        x = torch.randn(2, 20, 112, 112)
        y = m(x)
        print(f"base={base:<3} params={count_params(m):>10,}  out={tuple(y.shape)}")
    print("\ntrial 1's U-Net was 961k parameters and beat a 24.5M ResNet-34.")
