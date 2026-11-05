#!/usr/bin/env python3
"""
Two figures for the findings that had none: the spread across configurations,
and what the choice of split does.

Design notes
    Form first. 21 configurations is too many for bars, so panel A is a sorted
    dot plot with a median reference line: the reader needs the DISTRIBUTION and
    where our headline sits in it, not 21 magnitudes. Panel B plots AUC against
    channel count because the monotonic decline is the actual finding.

    Colour last, and only two hues carry meaning: accent for honest
    measurements, warning for the inflated one. Both validated against the
    light chart surface (CVD deltaE 17.9 protan, 28.5 normal, contrast >= 3:1).
    The shuffled control is not a third hue - it is a validity check, so it is
    drawn hatched in muted ink, which is a secondary encoding rather than a
    colour anyone must distinguish.

    Single theme on purpose: these are print figures for a paper.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "RECORD/figures"

INK, MUTED, LINE = "#171a1f", "#5f6672", "#dfe3e9"
ACCENT, WARN = "#1668a8", "#c33c33"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

df = pd.read_csv(ROOT / "RECORD/model_spread.csv").sort_values("auc")
med = df.auc.median()
HEAD = "terrain+basin+rain+rain_routed+soil+landcover"
seeds = ["sw_seed1", "sw_seed2", "sw_seed3"]

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.5),
                       gridspec_kw={"width_ratios": [1.35, 1]})

# ---- panel A: the spread -------------------------------------------------
a = ax[0]
y = np.arange(len(df))
a.hlines(y, df.auc.min() - 0.004, df.auc, color=LINE, lw=1, zorder=1)
for i, (_, r) in enumerate(df.iterrows()):
    is_head = r.run == HEAD
    is_seed = r.run in seeds
    a.plot(r.auc, i, "o", ms=9 if is_head else 7,
           color=WARN if is_head else (ACCENT if is_seed else "#9fb8cf"),
           mec="white", mew=1.2, zorder=3)
a.axvline(med, color=MUTED, ls="--", lw=1.2, zorder=2)
a.text(med, len(df) + 0.4, f" median {med:.3f}", color=MUTED, fontsize=8.5,
       va="bottom")
labels = [r.run if len(r.run) < 22 else r.run[:20] + "..." for _, r in df.iterrows()]
a.set_yticks(y); a.set_yticklabels(labels, fontsize=7.5)
a.set_xlabel("per-pixel AUC on the held-out basins")
a.set_title("A.  21 comparable configurations", fontsize=10.5, loc="left",
            fontweight="semibold", pad=14)
a.grid(axis="x", color=LINE, lw=0.7, alpha=0.8); a.set_axisbelow(True)
# Labels sit to the LEFT of their own dot, on the empty side of the plot,
# so no annotation crosses another mark.
hi = list(df.run).index(HEAD)
a.annotate("our headline, 30 channels", xy=(df.auc.iloc[hi], hi),
           xytext=(df.auc.min() - 0.001, hi), fontsize=8.5, color=WARN,
           va="center", ha="left",
           arrowprops=dict(arrowstyle="-", color=WARN, lw=1, alpha=.7))
a.annotate("terrain alone, 5 channels", xy=(df.auc.iloc[-1], len(df) - 1),
           xytext=(df.auc.min() - 0.001, len(df) - 1), fontsize=8.5, color=INK,
           va="center", ha="left",
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, alpha=.7))
sy = sorted(list(df.run).index(s) for s in seeds)
a.annotate("same configuration,\nthree random seeds", xy=(df.auc.iloc[sy[0]], sy[0]),
           xytext=(df.auc.min() - 0.001, sy[0] - 0.2), fontsize=8.5, color=ACCENT,
           va="center", ha="left",
           arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1, alpha=.7))
a.set_xlim(df.auc.min() - 0.030, df.auc.max() + 0.006)

# ---- panel B: more features, lower score ---------------------------------
b = ax[1]
sel = {"terrain": 5, "terrain+basin": 10, "terrain+basin+rain": 16,
       "terrain+basin+rain+rain_routed": 23, HEAD: 30}
pts = [(n, df[df.run == r].auc.iloc[0]) for r, n in sel.items()
       if (df.run == r).any()]
pts.sort()
xs, ys = zip(*pts)
# The seed-noise band is the point of this panel. Three runs of one
# configuration span 0.051, so a shaded band of that width around any point
# shows that almost every difference here is inside it. Drawing the trend line
# without it would imply a relationship the data does not support: the values
# are 0.745, 0.674, 0.713, 0.708, 0.695, which is not monotonic.
SEED_SPAN = 0.051
b.fill_between([min(xs) - 1, max(xs) + 1],
               np.mean(ys) - SEED_SPAN / 2, np.mean(ys) + SEED_SPAN / 2,
               color=ACCENT, alpha=0.10, zorder=1, lw=0)
b.text(max(xs) + 0.5, np.mean(ys) + SEED_SPAN / 2, " seed noise\n (0.051 wide)",
       fontsize=8, color=MUTED, va="top", ha="left")
b.plot(xs, ys, "-", color=ACCENT, lw=2, zorder=2)
b.plot(xs, ys, "o", ms=8, color=ACCENT, mec="white", mew=1.4, zorder=3)
for x_, y_ in pts:
    b.annotate(f"{y_:.3f}", (x_, y_), textcoords="offset points",
               xytext=(0, 10), ha="center", fontsize=8.5, color=INK)
b.set_xlabel("number of input channels")
b.set_ylabel("per-pixel AUC")
b.set_title("B.  Terrain alone scores highest,\n      but most gaps are inside seed noise",
            fontsize=10.5, loc="left", fontweight="semibold", pad=14)
b.grid(color=LINE, lw=0.7, alpha=0.8); b.set_axisbelow(True)
b.set_xlim(min(xs) - 1.5, max(xs) + 6)
b.set_ylim(min(ys) - 0.016, max(ys) + 0.022)
fig.tight_layout()
fig.savefig(OUT / "13_model_spread.png", dpi=150, facecolor="white")
plt.close(fig)
print("wrote 13_model_spread.png")

# ---- figure: what the split does -----------------------------------------
sp = [("Split by pixel,\nmean of 6 basins", 0.926, WARN, False),
      ("Split by basin\n(transfer, ours)", 0.695, ACCENT, False),
      ("Split by event,\nmean of 6 basins", 0.675, ACCENT, False),
      ("Shuffled label\n(validity check)", 0.500, "#b9bfc9", True)]
fig, ax = plt.subplots(figsize=(7.6, 3.5))
ypos = np.arange(len(sp))[::-1]
for i, (lab, v, col, hatch) in zip(ypos, sp):
    ax.barh(i, v - 0.45, left=0.45, height=0.52, color=col,
            hatch="///" if hatch else None,
            edgecolor="white" if not hatch else MUTED, linewidth=1.6, zorder=3)
    ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=10,
            color=INK, fontweight="semibold")
ax.set_yticks(ypos); ax.set_yticklabels([s[0] for s in sp], fontsize=9)
ax.set_xlim(0.45, 1.02); ax.set_xlabel("per-pixel AUC")
ax.set_title("How the data is split changes the result more than the model does",
             fontsize=10.5, loc="left", fontweight="semibold", pad=12)
ax.grid(axis="x", color=LINE, lw=0.7, alpha=0.8); ax.set_axisbelow(True)
ax.annotate("", xy=(0.926, ypos[0] - 0.42), xytext=(0.695, ypos[0] - 0.42),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
ax.text((0.926 + 0.695) / 2, ypos[0] - 0.62,
        "+0.231 from spatial autocorrelation", ha="center", fontsize=8.5,
        color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "14_split_effect.png", dpi=150, facecolor="white")
plt.close(fig)
print("wrote 14_split_effect.png")
