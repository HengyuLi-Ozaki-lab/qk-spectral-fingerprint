"""lambda-curve figure for the steering paper: dose-defying concentration.

(a) population-median rope_imag_frac trajectory: the population converts at the smallest
    dose and saturates there.
(b) THE PHENOMENON: population rif (flat at floor) vs the load-bearing (argmax-prev) head's
    rif, which RISES with lambda. The ban converts the head only by leaving phase nowhere.
(c) cost vs lambda against the ban's cost: the dominance window (strict at lambda in {1,3}).

Trajectories from results/h2/runs/m22/*.parquet (pop_med_rif).
Per-head final values from the final.pt battery, as tabulated in the FINDING P-M31lambda
entry of results/h2/LOG.md (verified independently; not recomputed here).
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results" / "h2" / "runs" / "m22"
OUT = Path(__file__).resolve().parent.parent / "paper" / "figures" / "m31_lambda_curve.pdf"
PREV = Path("/tmp/claude-1005/-home-li-LLM-research-QK-analyze/057fe19d-01e8-4be9-aca1-4e8f4b2ad31f/scratchpad")

BLUE, VERM, ORANGE, GREEN, PURPLE, GREY = (
    "#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#7F7F7F")

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False, "pdf.fonttype": 42,
})

# ---- final.pt battery values (FINDING P-M31lambda, results/h2/LOG.md) ----
LAM = np.array([1, 3, 10])
ARGMAX_RIF = {0: np.array([0.143, 0.195, 0.326]), 1: np.array([0.027, 0.012, 0.244])}
POP_RIF    = {0: np.array([0.011, 0.006, 0.006]), 1: np.array([0.011, 0.006, 0.010])}
COST       = {0: np.array([0.013, 0.022, 0.055]), 1: np.array([0.021, 0.030, 0.064])}
BAN = dict(argmax=(0.005, 0.006), pop=(0.003, 0.003), cost=(0.039, 0.048))
FREE_RIF = 0.51

def traj(tag):
    out = []
    for s in (0, 1):
        f = R / f"m22_{tag}_seed{s}.parquet" if tag in ("free", "constraint") else \
            R / f"m22_select_seed{s}_{tag}.parquet"
        if f.exists():
            d = pd.read_parquet(f).sort_values("tokens")
            out.append((d["tokens"].values / 1e9, d["pop_med_rif"].values))
    return out

fig = plt.figure(figsize=(7.0, 4.7))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.92], hspace=0.52, wspace=0.9)
axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]

# ---------- (a) population trajectory ----------
ax = axes[0]
for tag, lab, c, ls in [("free", "free", BLUE, "-"), ("m31bL1", "$\\lambda{=}1$", GREEN, "-"),
                        ("m31b", "$\\lambda{=}3$", ORANGE, "--"),
                        ("m31bL10", "$\\lambda{=}10$", PURPLE, "-."),
                        ("constraint", "ban $\\lambda{=}10$", VERM, ":")]:
    for i, (t, y) in enumerate(traj(tag)):
        ax.plot(t, y, color=c, ls=ls, lw=1.2, alpha=0.9, label=lab if i == 0 else None)
ax.set_xlabel("tokens (B)"); ax.set_ylabel("population-median $\\mathtt{rope\\_imag\\_frac}$")
ax.set_ylim(-0.02, 0.58)
ax.legend(loc="center right", handlelength=1.6, labelspacing=0.3, fontsize=7.2)
ax.text(0.02, 1.03, "(a) population: converts at any dose",
        transform=ax.transAxes, fontsize=8, color="0.3")

# ---------- (b) THE PHENOMENON ----------
ax = axes[1]
for s, mk in [(0, "o"), (1, "s")]:
    ax.plot(LAM, ARGMAX_RIF[s], color=VERM, marker=mk, ms=5, lw=1.6,
            label="load-bearing head" if s == 0 else None)
    ax.plot(LAM, POP_RIF[s], color=BLUE, marker=mk, ms=4, lw=1.2, ls="--",
            label="population median" if s == 0 else None)
ax.axhline(FREE_RIF, ls=":", c=GREY, lw=1)
ax.text(9.6, FREE_RIF + 0.018, "free default (0.51)", fontsize=6.6, color="0.35", ha="right")
ax.scatter([10, 10], BAN["argmax"], marker="x", s=38, color="0.2", zorder=5,
           label="ban $\\lambda{=}10$ (converts)")
ax.set_xscale("log"); ax.set_xticks(LAM); ax.set_xticklabels(["1", "3", "10"])
ax.set_xlabel("regularizer dose $\\lambda$")
ax.set_ylabel("$\\mathtt{rope\\_imag\\_frac}$")
ax.set_ylim(-0.02, 0.60)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), handlelength=1.5,
          labelspacing=0.25, fontsize=6.8)
ax.text(0.02, 1.03, "(b) working head: retains $\\it{more}$",
        transform=ax.transAxes, fontsize=8, color="0.3")

# ---------- (c) dominance window ----------
ax = axes[2]
for s, mk in [(0, "o"), (1, "s")]:
    ax.plot(LAM, COST[s], color=GREEN, marker=mk, ms=5, lw=1.6, label=f"steer (seed {s})")
ax.axhspan(BAN["cost"][0], BAN["cost"][1], color=VERM, alpha=0.17)
ax.axhline(np.mean(BAN["cost"]), color=VERM, ls="--", lw=1.2, label="ban $\\lambda{=}10$")
ax.set_xscale("log"); ax.set_xticks(LAM); ax.set_xticklabels(["1", "3", "10"])
ax.set_xlabel("regularizer dose $\\lambda$")
ax.set_ylabel("capability cost (nats vs free)")
ax.axvspan(0.85, 3.6, color=GREEN, alpha=0.08)
ax.set_ylim(0.005, 0.072)
ax.text(1.75, 0.0075, "dominance window", fontsize=7.5, color=GREEN, ha="center")
ax.legend(loc="upper left", handlelength=1.5, labelspacing=0.3, fontsize=6.8)
ax.text(0.02, 1.03, "(c) cost vs the ban",
        transform=ax.transAxes, fontsize=8, color="0.3")

fig.savefig(OUT); fig.savefig(PREV / "m31_lambda_curve.png", dpi=170)
print(f"saved {OUT}")
