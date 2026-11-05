#!/usr/bin/env python3
"""
Phase 6 - train the flood model and score it against the free baseline.

    ./.venv/bin/python -u phase6/train.py --tiles data/tiles_v4 \
        --neg data/tiles_neg_v4 --groups terrain,basin,rain,rain_routed,soil,landcover

WHAT COUNTS AS A RESULT
    Not "the model reached 0.62". The elevation-only rule is free, needs no
    training and no rain data, and on this dataset it scores 0.5442. A model
    earns its place only by beating that ON THE SAME TEST TILES, by more than
    the spread between tiles. So every run reports the paired difference and a
    Wilcoxon test over tiles, not two numbers side by side.

    PROJECT_PLAN's comparison set is elevation rule -> U-Net -> U-Net + rain +
    soil. --groups runs exactly that, one channel family at a time.

RULES CARRIED FROM TRIAL 1 AND FROM THE BUGS THIS PROJECT ALREADY HIT
    SPLIT BY REGION, NEVER RANDOMLY. Tiles from one event overlap heavily, so a
    random split puts near-identical tiles on both sides and every score becomes
    a memorisation score. Splitting on HUC4 also makes the test a transfer test,
    which is what NEXT_STEPS 4b asks for.

    NORMALISATION STATISTICS COME FROM TRAIN ONLY. Fitting them on everything
    leaks test distribution into training. Cheap to get right, invisible when
    wrong.

    THE QUIET CONTROLS ARE TRAINING DATA, NOT DECORATION. Every flood tile is
    from a day a flood was reported. Without the 174 quiet dates the model never
    sees "these same hills, no storm" and cannot learn to stay silent. They are
    also the only way to measure a false alarm.

    LABELS ARE THRESHOLDED HERE, NOT IN THE TILES. The tiles hold continuous
    coh_drop, so the threshold stays a reported choice rather than a baked-in one.

    THE LABEL IS CHECKED BEFORE ANY TRAINING. coh_drop is a difference of two
    coherences and cannot leave -1..+1. Tiling the wrong band cost this project
    a full pass of every downstream number, so it is asserted, not assumed.

Output: data/train_<tag>.json, data/model_<tag>.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (FloodUNet, count_params, masked_bce, masked_focal,
                   masked_bce_weighted, within_tile_ranking_loss)

ROOT = Path(__file__).resolve().parent.parent

# Channel families, for the ablation. Membership is by name prefix so a new
# band joins the right family automatically.
GROUPS = {
    "terrain": ("elev_", "microrelief_", "roughness_", "slope_",
                "terrain_roughness", "slope_variability", "local_depression"),
    "basin": ("basin_", "twi_spi_interaction"),
    "extra": ("sink_depth", "dist_stream", "dist_big_stream"),
    "rain": ("rain_1h_peak", "rain_24h", "rain_ante"),
    "rain_routed": ("catchment_area", "rain_1h_over_ksat", "rain_1h_upstream",
                    "rain_24h_upstream", "rain_cover", "runoff_1h_upstream"),
    "soil": ("soil_",),
    "landcover": ("lc_",),
}


def log(m: str) -> None:
    print(m, flush=True)


def auc_within(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    """Within-tile AUC, identical to phase6/baselines.py so numbers compare."""
    s, y = scores[mask], labels[mask]
    if s.size < 20:
        return np.nan
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(s).rank().values
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# Channels spanning many orders of magnitude, all strictly non-negative.
# A catchment volume runs from 0 to 1.6e7 m3 and flow accumulation from 1 cell
# to millions; z-scoring those raw leaves a handful of outlet pixels dominating
# every batch. log1p first is the standard handling and is physically natural,
# since contributing area is log-distributed. Applied identically at train and
# test time because it happens once, at load.
LOG_CHANNELS = ("basin_facc", "basin_spi", "catchment_area_km2",
                "rain_1h_upstream_vol_m3", "rain_24h_upstream_vol_m3",
                "runoff_1h_upstream_vol_m3")


def auc_pixel(P, Y, M, thr, cap=2_000_000, seed=0) -> float:
    """Pooled per-pixel AUC: every pixel of every tile, one ranking.

    THIS is the research question - for each 20 m pixel, flood or not - so it is
    what the best epoch is now chosen on. Selecting on within-tile AUC while
    reporting pixel AUC would be optimising one thing and grading another.
    """
    s = P[M]
    y = (Y > thr)[M]
    if y.sum() < 10 or (~y).sum() < 10:
        return float("nan")
    if s.size > cap:
        i = np.random.default_rng(seed).choice(s.size, cap, replace=False)
        s, y = s[i], y[i]
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_tiles(d: Path, rung, ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        pd.DataFrame, list[str]]:
    """Read every shard into memory.

    float32, NOT float16. The routed rainfall volumes reach 1.6e7 m3 and float16
    saturates at 65504, so half precision silently turned the strongest features
    in the dataset into infinity. Memory is the cheaper thing to spend.
    """
    # rung may be a single value or several. Several stacks the terrain from
    # each resolution side by side, so the model sees 1 m micro-relief and 30 m
    # context at once instead of being forced to pick one scale.
    rungs = [int(r) for r in (rung if isinstance(rung, (list, tuple))
                              else str(rung).split(","))]
    chan = json.loads((d / "channels.json").read_text())
    names = []
    for r_ in rungs:
        names += list(chan["rung_channels"][str(r_)])
    names += list(chan["shared_channels"])
    idx = pd.read_csv(d / "index.csv")
    sh = sorted((d / "shared").glob("shard-*.npz"))
    rgs = [sorted((d / f"rung{r_}m").glob("shard-*.npz")) for r_ in rungs]
    for r_, g in zip(rungs, rgs):
        if len(g) != len(sh):
            raise SystemExit(f"{d}: {len(sh)} shared shards but {len(g)} for "
                             f"rung {r_} m")
    X, Y, M = [], [], []
    for k, ps in enumerate(sh):
        zs = np.load(ps)
        parts = [np.load(g[k])["xt"] for g in rgs] + [zs["xs"]]
        X.append(np.concatenate(parts, axis=1).astype("float32"))
        Y.append(zs["y"].astype("float32"))
        M.append(zs["m"])
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    M = np.concatenate(M)
    if len(idx) != len(X):
        raise SystemExit(f"{d}: index has {len(idx)} rows but shards hold {len(X)}")

    for i, n in enumerate(names):
        if n in LOG_CHANNELS:
            X[:, i] = np.log1p(np.maximum(X[:, i], 0.0))
    # Cheap finiteness audit on a sample of tiles - a full boolean index here
    # would allocate a second copy of the whole array.
    s = np.linspace(0, len(X) - 1, min(200, len(X))).astype(int)
    bad = [names[c] for c in range(X.shape[1])
           if not np.isfinite(X[s, c][M[s]]).all()]
    if bad:
        log(f"  warning: {d.name} has non-finite values inside the mask: {bad}")

    # The check that was missing the first time round.
    v = Y[M]
    lo, hi = float(v.min()), float(v.max())
    if lo < -1.001 or hi > 1.001:
        raise SystemExit(f"{d}: label range {lo:.2f}..{hi:.2f} is impossible for "
                         f"a coherence difference - the wrong band was tiled")
    return X, Y, M, idx, names


def pick_channels(names: list[str], groups: list[str]) -> list[int]:
    keep = []
    for i, n in enumerate(names):
        for g in groups:
            if any(n.startswith(p) for p in GROUPS[g]):
                keep.append(i)
                break
    return keep


def random_split(idx: pd.DataFrame, seed: int = 0, val_frac: float = 0.15,
                 test_frac: float = 0.20):
    """A RANDOM tile split. Reported separately, never as the headline.

    This measures a genuinely different and much easier question: how well the
    model does on ground it has already seen, just on a different day. 80% of
    our 567 events sit within 5 km of another event, so under a random split
    most test terrain is also training terrain, and the model can score by
    recognising a particular wash rather than by learning how water moves.

    Legitimate to report for "somewhere already mapped". Not legitimate as a
    transfer claim, which is what the region split measures.
    """
    rng = np.random.default_rng(seed)
    n = len(idx)
    r = rng.permutation(n)
    n_te = int(n * test_frac)
    n_va = int(n * val_frac)
    te = np.zeros(n, bool); va = np.zeros(n, bool)
    te[r[:n_te]] = True
    va[r[n_te:n_te + n_va]] = True
    return ~(te | va), va, te


def region_split(idx: pd.DataFrame, test: list[str], val: list[str]):
    r = idx.huc4.astype(str)
    te = r.isin(test).values
    va = r.isin(val).values
    tr = ~(te | va)
    return tr, va, te


def evaluate(model, X, Y, M, cols, mean, std, thr, dev, bs=32):
    """Per-tile within-tile AUC for the model and for the elevation rule."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs][:, cols].astype("float32")).to(dev)
            xb = (xb - mean) / std
            xb = torch.nan_to_num(xb)
            p = torch.sigmoid(model(xb))[:, 0].cpu().numpy()
            out.append(p)
    P = np.concatenate(out)
    lab = (Y > thr).astype("float32")
    return P, np.array([auc_within(P[i], lab[i], M[i]) for i in range(len(P))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/tiles_v4")
    ap.add_argument("--neg", default="data/tiles_neg_v4")
    ap.add_argument("--rung", default="10",
                    help="terrain resolution in metres. Several may be given "
                         "comma-separated (e.g. 1,10,30) to stack them")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--groups", default="terrain,basin,rain,rain_routed,soil,landcover")
    ap.add_argument("--test-regions", default="1606,1809")
    ap.add_argument("--val-regions", default="1507")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank-weight", type=float, default=0.3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--engineered", action="store_true",
                    help="add the ratio/product features from the team's "
                         "XGBoost notebook")
    ap.add_argument("--loss", default="bce",
                    choices=["bce", "focal", "confidence"],
                    help="focal down-weights easy pixels; confidence weights "
                         "each pixel by how far its coherence drop stands above "
                         "its own measurement noise")
    ap.add_argument("--tta", action="store_true",
                    help="average predictions over flips at test time")
    ap.add_argument("--split", default="region", choices=["region", "random"],
                    help="region = held-out deserts, the transfer claim. "
                         "random = same ground, different day; much easier and "
                         "reported separately, never as the headline")
    ap.add_argument("--target", default="binary",
                    choices=["binary", "continuous"],
                    help="continuous trains on the coherence drop itself as a "
                         "soft target instead of a yes/no cut at 0.15. "
                         "Thresholding throws away magnitude: a pixel that "
                         "dropped 0.75 and one that dropped 0.16 are currently "
                         "the same label, and a pixel at 0.14 is treated as "
                         "just as dry as one at -0.5")
    ap.add_argument("--soft-scale", type=float, default=0.30,
                    help="drop value mapped to a soft target of 1.0. The "
                         "default puts the 0.15 threshold at exactly 0.5, so "
                         "the decision boundary is unchanged")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="weight on the flood class in BCE. Only 5-10%% of "
                         "pixels flood, so the cheapest way to cut the loss is "
                         "to predict dry everywhere; this pushes back")
    ap.add_argument("--min-train-flood", type=float, default=0.0,
                    help="drop TRAINING tiles whose flood fraction is below "
                         "this. The test set is never filtered - selecting the "
                         "test set on the label raises precision purely because "
                         "floods become commoner, while AUC falls (measured: "
                         "0.6950 -> 0.5791 at a 5%% cut)")
    ap.add_argument("--train-on", default="all", choices=["all", "flood_only"],
                    help="flood_only drops the quiet controls from TRAINING. "
                         "With them in, the easiest way to cut the loss is to "
                         "learn 'no rain today', which needs no spatial skill "
                         "at all. Removing them forces the model to learn WHERE")
    ap.add_argument("--prior", default="downhill",
                    choices=["downhill", "free", "off"],
                    help="elevation term. downhill forces 'lower floods more', which is BELOW chance on this dataset")
    ap.add_argument("--select-on", default="within",
                    choices=["within", "pixel"],
                    help="which validation metric picks the best epoch")
    ap.add_argument("--init-from", default="",
                    help="tag of a saved model to start from (fine-tuning). Its "
                         "channels must match, and its input scaling is reused, "
                         "so the weights see inputs on the scale they learned")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    groups = [g for g in args.groups.split(",") if g]
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}; choose from {list(GROUPS)}")
    tag = args.tag or "+".join(groups)

    log(f"device {dev}   groups {groups}   threshold {args.threshold}")
    X, Y, M, idx, names = load_tiles(ROOT / args.tiles, args.rung)
    log(f"flood tiles {len(X)} from {idx.event.nunique()} events, "
        f"regions {sorted(idx.huc4.astype(str).unique())}")

    is_neg = np.zeros(len(X), dtype=bool)
    if not args.no_negatives and (ROOT / args.neg).exists():
        Xn, Yn, Mn, idxn, names_n = load_tiles(ROOT / args.neg, args.rung)
        if names_n != names:
            raise SystemExit("negative tiles have a different channel layout")
        log(f"quiet-control tiles {len(Xn)} from {idxn.event.nunique()} dates")
        X = np.concatenate([X, Xn])
        Y = np.concatenate([Y, Yn])
        M = np.concatenate([M, Mn])
        is_neg = np.concatenate([is_neg, np.ones(len(Xn), dtype=bool)])
        idx = pd.concat([idx, idxn], ignore_index=True)

    if args.engineered:
        # From the team's XGBoost notebook. Trees cannot form a ratio or a
        # product, so those must be handed over. A CNN gets DIFFERENCES free in
        # its first convolution, which is a weighted sum of input channels - so
        # only the ratio and the product are expected to add anything here.
        r = str(args.rung).split(",")[0]
        def ch(n):
            return X[:, names.index(n)] if n in names else None
        extra = {}
        rough, emean = ch(f"roughness_{r}m"), ch(f"elev_mean_{r}m")
        if rough is not None and emean is not None:
            extra["terrain_roughness"] = rough / (np.abs(emean) + 1e-6)
        micro, slope = ch(f"microrelief_{r}m"), ch(f"slope_{r}m")
        if micro is not None and slope is not None:
            extra["slope_variability"] = micro / (np.abs(slope) + 1e-6)
        twi, spi = ch("basin_twi"), ch("basin_spi")
        if twi is not None and spi is not None:
            extra["twi_spi_interaction"] = twi * spi
        emin = ch(f"elev_min_{r}m")
        if emean is not None and emin is not None:
            extra["local_depression"] = emean - emin
        if extra:
            X = np.concatenate([X] + [v[:, None] for v in extra.values()], axis=1)
            names = names + list(extra)
            log(f"engineered features added: {list(extra)}")

    cols = pick_channels(names, groups)
    if not cols:
        raise SystemExit("no channels selected")
    chosen = [names[i] for i in cols]
    log(f"channels {len(cols)}: {chosen}")

    first_rung = str(args.rung).split(",")[0]
    elev_name = f"elev_mean_{first_rung}m"
    elev_index = chosen.index(elev_name) if elev_name in chosen else None
    log(f"elevation prior: mode={args.prior}"
        + (f", elevation is channel {elev_index}" if elev_index is not None
           and args.prior != "off" else " (no elevation term in the output)"))

    test_r = [s for s in args.test_regions.split(",") if s]
    val_r = [s for s in args.val_regions.split(",") if s]
    if args.split == "random":
        tr, va, te = random_split(idx, seed=args.seed)
        log("SPLIT: RANDOM - test ground also appears in training. This is the "
            "easier question and must never be quoted as a transfer result.")
    else:
        tr, va, te = region_split(idx, test_r, val_r)
    lab_all_pre = (Y > args.threshold)
    log("\nsplit by REGION - held-out deserts, this is the transfer claim:"
        if args.split == "region" else
        "\nsplit RANDOMLY - same ground on a different day, the easier question:")
    for nm, sel in (("train", tr), ("val", va), ("test", te)):
        log(f"  {nm:<6}{int(sel.sum()):>6} tiles  "
            f"({int((sel & is_neg).sum())} quiet)  "
            f"regions {sorted(idx.huc4.astype(str)[sel].unique())}")
    if tr.sum() < 50 or te.sum() < 50:
        raise SystemExit("split leaves too few tiles - check --test-regions")

    # Normalisation from TRAIN ONLY, over valid pixels only.
    sub = X[tr][:, cols].astype("float32")
    msk = M[tr]
    mean = np.zeros(len(cols), "float32")
    std = np.ones(len(cols), "float32")
    for c in range(len(cols)):
        v = sub[:, c][msk]
        v = v[np.isfinite(v)]
        if v.size:
            mean[c] = v.mean()
            std[c] = v.std() if v.std() > 1e-6 else 1.0
    del sub
    init = None
    if args.init_from:
        init = torch.load(ROOT / f"data/model_{args.init_from}.pt", map_location="cpu",
                          weights_only=False)   # our own checkpoint
        if list(init["channels"]) != list(chosen):
            raise SystemExit(f"--init-from {args.init_from}: channels differ from this run")
        mean, std = np.asarray(init["mean"], "float32"), np.asarray(init["std"], "float32")
        log(f"starting from model_{args.init_from}.pt, reusing its input scaling")
    mean_t = torch.from_numpy(mean).view(1, -1, 1, 1).to(dev)
    std_t = torch.from_numpy(std).view(1, -1, 1, 1).to(dev)

    model = FloodUNet(in_ch=len(cols), elev_index=elev_index,
                      base=args.base, prior=args.prior).to(dev)
    if init is not None:
        model.load_state_dict(init["state"])
    log(f"\nmodel: {count_params(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if args.min_train_flood > 0:
        # Fraction of each tile that is flooded, computed on valid pixels only.
        ff = np.array([(lab_all_pre[i][M[i]].mean() if M[i].sum() > 200 else 0.0)
                       for i in range(len(Y))])
        n_before = int(tr.sum())
        tr = tr & (ff >= args.min_train_flood)
        log(f"dropping near-empty TRAINING tiles below "
            f"{100*args.min_train_flood:.0f}% flood: {n_before} -> {int(tr.sum())}"
            f"   (val and test are untouched)")
    if args.train_on == "flood_only":
        n_before = int(tr.sum())
        tr = tr & ~is_neg
        log(f"training on flood days only: {n_before} -> {int(tr.sum())} tiles "
            f"(quiet controls stay in val/test for scoring)")
    tr_i = np.flatnonzero(tr)
    if args.target == "continuous":
        # Soft targets: linear in the drop, clipped to [0,1], with the 0.15
        # threshold landing on 0.5 so the decision boundary does not move.
        # Evaluation still uses the hard label, so every number stays comparable.
        # nan_to_num BEFORE clipping. Y is NaN on masked pixels, and although
        # the loss multiplies by the mask, NaN * 0 is still NaN, so a single
        # masked pixel poisons the whole batch. The binary path never hit this
        # because (NaN > threshold) is simply False.
        lab_all = np.clip(np.nan_to_num(Y, nan=0.0) / args.soft_scale,
                          0.0, 1.0).astype("float32")
        log(f"continuous target: drop/{args.soft_scale} clipped to [0,1]; "
            f"threshold {args.threshold} sits at "
            f"{args.threshold/args.soft_scale:.2f}")
    else:
        lab_all = lab_all_pre.astype("float32")
    pos_rate = float(lab_all[M].mean())
    log(f"positive pixel rate (train+val+test, valid pixels): {100*pos_rate:.1f}%\n")

    best = (-1.0, -1)
    hist = []

    def save_ckpt():
        torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                    "channels": chosen, "elev_index": elev_index,
                    "groups": groups, "rung": args.rung,
                    "threshold": args.threshold, "prior": args.prior,
                    "select_on": args.select_on,
                    "pos_weight": args.pos_weight,
                    "train_on": args.train_on,
                    "min_train_flood": args.min_train_flood,
                    "target": args.target, "soft_scale": args.soft_scale,
                    "split": args.split, "loss": args.loss,
                    "init_from": args.init_from},
                   ROOT / f"data/model_{tag}.pt")

    if init is not None:
        Pv, a_val = evaluate(model, X[va], Y[va], M[va], cols, mean_t, std_t,
                             args.threshold, dev)
        v_within, v_pixel = float(np.nanmean(a_val)), auc_pixel(Pv, Y[va], M[va], args.threshold)
        best = (v_pixel if args.select_on == "pixel" else v_within, 0)
        hist.append({"epoch": 0, "loss": None, "val_auc": round(v_within, 4),
                     "val_pixel_auc": round(v_pixel, 4)})
        save_ckpt()
        log(f"  epoch   0  (starting model)  within-tile {v_within:.4f}  PIXEL {v_pixel:.4f}  <- best, saved")

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(tr_i)
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(tr_i), args.batch):
            b = tr_i[i:i + args.batch]
            xb = torch.from_numpy(X[b][:, cols].astype("float32")).to(dev)
            xb = torch.nan_to_num((xb - mean_t) / std_t)
            yb = torch.from_numpy(lab_all[b]).unsqueeze(1).to(dev)
            mb = torch.from_numpy(M[b]).unsqueeze(1).to(dev)
            logit = model(xb)
            if args.loss == "focal":
                loss = masked_focal(logit, yb, mb)
            elif args.loss == "confidence":
                # |drop| / sigma(drop), capped. A drop of 0.75 teaches hard; one
                # of 0.16, inside the noise of a 5-look estimate, teaches softly.
                # nan_to_num FIRST. Y is NaN on masked pixels and NaN * 0 is
                # still NaN, so one masked pixel poisons the batch - exactly
                # what killed the continuous-target run.
                wb = torch.from_numpy(
                    np.clip(np.abs(np.nan_to_num(Y[b], nan=0.0)) / 0.10,
                            0.2, 3.0)).unsqueeze(1).to(dev)
                loss = masked_bce_weighted(logit, yb, mb, wb)
            else:
                loss = masked_bce(logit, yb, mb, pos_weight=args.pos_weight)
            if args.rank_weight > 0:
                loss = loss + args.rank_weight * within_tile_ranking_loss(
                    logit, yb, mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        Pv, a_val = evaluate(model, X[va], Y[va], M[va], cols, mean_t, std_t,
                             args.threshold, dev)
        v_within = float(np.nanmean(a_val))
        v_pixel = auc_pixel(Pv, Y[va], M[va], args.threshold)
        v = v_pixel if args.select_on == "pixel" else v_within
        hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 4),
                     "val_auc": round(v_within, 4),
                     "val_pixel_auc": round(v_pixel, 4)})
        star = ""
        if v > best[0]:
            best = (v, ep)
            save_ckpt()
            star = "  <- best, saved"
        log(f"  epoch {ep:>3}  loss {tot/max(nb,1):.4f}  within-tile "
            f"{v_within:.4f}  PIXEL {v_pixel:.4f}{star}")
        if ep - best[1] >= args.patience:
            log(f"  no improvement for {args.patience} epochs, stopping")
            break

    # ---- test on held-out regions, against the free baseline ----------------
    ck = torch.load(ROOT / f"data/model_{tag}.pt", map_location=dev,
                    weights_only=False)   # our own checkpoint
    model.load_state_dict(ck["state"])
    P, a_model = evaluate(model, X[te], Y[te], M[te], cols, mean_t, std_t,
                          args.threshold, dev)
    Xte, Yte, Mte = X[te], Y[te], M[te]
    ei = names.index(elev_name)
    lab_te = (Yte > args.threshold).astype("float32")
    a_elev = np.array([auc_within(-Xte[i, ei].astype("float32"), lab_te[i], Mte[i])
                       for i in range(len(Xte))])

    ok = np.isfinite(a_model) & np.isfinite(a_elev)
    diff = a_model[ok] - a_elev[ok]
    from scipy import stats
    w = stats.wilcoxon(diff) if ok.sum() > 10 else None

    log("\n" + "=" * 74)
    log(f"TEST on held-out regions {test_r} - {int(ok.sum())} scorable tiles")
    log("=" * 74)
    log(f"  model                        {np.nanmean(a_model):.4f}")
    log(f"  elevation-only rule (free)   {np.nanmean(a_elev):.4f}")
    log(f"  paired difference            {diff.mean():+.4f}  "
        f"(median {np.median(diff):+.4f})")
    log(f"  model wins on                {100*float((diff>0).mean()):.1f}% of tiles")
    if w is not None:
        log(f"  Wilcoxon p                   {w.pvalue:.3g}")
        verdict = ("MODEL BEATS THE FREE BASELINE" if diff.mean() > 0 and w.pvalue < 0.05
                   else "NO SIGNIFICANT GAIN OVER THE FREE BASELINE")
        log(f"  VERDICT: {verdict}")

    # ---- false alarms: what does it say on days with no flood? --------------
    res = {"tag": tag, "groups": groups, "channels": chosen,
           "test_regions": test_r, "val_regions": val_r,
           "threshold": args.threshold, "rung": args.rung,
           "params": count_params(model), "history": hist,
           "test_tiles": int(ok.sum()),
           "auc_model": round(float(np.nanmean(a_model)), 4),
           "auc_elevation": round(float(np.nanmean(a_elev)), 4),
           "paired_diff": round(float(diff.mean()), 4),
           "win_rate": round(float((diff > 0).mean()), 4),
           "wilcoxon_p": float(w.pvalue) if w is not None else None}

    neg_te = te & is_neg
    pos_te = te & ~is_neg
    if neg_te.sum() > 5:
        Pn, _ = evaluate(model, X[neg_te], Y[neg_te], M[neg_te], cols, mean_t,
                         std_t, args.threshold, dev)
        Pp, _ = evaluate(model, X[pos_te], Y[pos_te], M[pos_te], cols, mean_t,
                         std_t, args.threshold, dev)
        mn = np.array([Pn[i][M[neg_te][i]].mean() for i in range(len(Pn))])
        mp = np.array([Pp[i][M[pos_te][i]].mean() for i in range(len(Pp))])
        log("\n  false alarms - mean predicted flood probability per tile")
        log(f"    flood-day tiles  {mp.mean():.4f}  (n={len(mp)})")
        log(f"    quiet-day tiles  {mn.mean():.4f}  (n={len(mn)})")
        sep = stats.mannwhitneyu(mp, mn, alternative="greater")
        log(f"    separation AUC   {sep.statistic/(len(mp)*len(mn)):.4f}  "
            f"p={sep.pvalue:.3g}")
        res["quiet_day_mean_prob"] = round(float(mn.mean()), 4)
        res["flood_day_mean_prob"] = round(float(mp.mean()), 4)
        res["flood_vs_quiet_auc"] = round(
            float(sep.statistic / (len(mp) * len(mn))), 4)
    else:
        log("\n  no quiet-control tiles in the test regions - false alarm rate "
            "not measurable on this split")

    out = ROOT / f"data/train_{tag}.json"
    out.write_text(json.dumps(res, indent=1))
    log(f"\nwrote {out.relative_to(ROOT)} and data/model_{tag}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
