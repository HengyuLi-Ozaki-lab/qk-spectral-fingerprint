"""Scale-transfer acceleration figure (steering paper): free vs functional-plant assist
at Pythia-160M. (a) induction-head score vs tokens through the formation window with the
0.5 crossing; (b) adoption: max prev-token attention over the seeded slots, full run.
Data: results/h2/runs/m22/m22_{free,assist*redo}_seed{0,1}.parquet."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "h2" / "runs" / "m22"
OUT = ROOT / "paper" / "figures" / "m22_accel.pdf"
PREV = Path("/tmp/claude-1005/-home-li-LLM-research-QK-analyze/057fe19d-01e8-4be9-aca1-4e8f4b2ad31f/scratchpad")

BLUE, GREEN, GREY = "#0072B2", "#009E73", "#7F7F7F"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False, "pdf.fonttype": 42,
})

def load(name):
    d = pd.read_parquet(R / f"{name}.parquet").sort_values("tokens")
    return d["tokens"].values / 1e9, d

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))

# ---------- (a) induction formation ----------
ax = axes[0]
for s, ls in [(0, "-"), (1, "--")]:
    t, d = load(f"m22_free_seed{s}")
    ax.plot(t, d["ind_beh"], color=BLUE, ls=ls, lw=1.3,
            label="free" if s == 0 else None)
    t, d = load(f"m22_assist_seed{s}_redo")
    ax.plot(t, d["ind_beh"], color=GREEN, ls=ls, lw=1.3,
            label="assisted (functional plant)" if s == 0 else None)
ax.axhline(0.5, ls=":", c=GREY, lw=0.9)
ax.scatter([0.364, 0.388], [0.5, 0.5], marker="x", s=30, color=BLUE, zorder=5)
ax.scatter([0.267, 0.267], [0.5, 0.5], marker="x", s=30, color=GREEN, zorder=5)
ax.annotate("$-27\\%/-31\\%$", xy=(0.267, 0.5), xytext=(0.52, 0.30), fontsize=7.5,
            color="0.25", arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
ax.set_xlim(0, 1.2); ax.set_ylim(-0.03, 1.02)
ax.set_xlabel("tokens (B)"); ax.set_ylabel("induction-head score")
ax.legend(loc="lower right", handlelength=1.8)
ax.text(0.01, 1.05, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")

# ---------- (b) adoption: functional bias plant vs inert weight plant ----------
VERM = "#D55E00"
ax = axes[1]
for s, ls in [(0, "-"), (1, "--")]:
    t, d = load(f"m22_assist_seed{s}_redo")
    ax.plot(t, d["seeded_prev_max"], color=GREEN, ls=ls, lw=1.3,
            label="functional plant (rotary bias)" if s == 0 else None)
    t, d = load(f"m22_assist_seed{s}")
    ax.plot(t, d["seeded_prev_max"], color=VERM, ls=ls, lw=1.3,
            label="inert plant (rotary weights)" if s == 0 else None)
ax.axhline(1/64, ls=":", c=GREY, lw=0.9)
ax.text(3.95, 1/64 + 0.03, "uniform attention (1/64)", fontsize=6.8, color="0.35", ha="right")
ax.set_xlim(0, 4); ax.set_ylim(-0.03, 1.02)
ax.set_xlabel("tokens (B)")
ax.set_ylabel("max prev-token attention,\nseeded slots")
ax.legend(loc="center right", handlelength=1.8)
ax.text(0.01, 1.05, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT); fig.savefig(PREV / "m22_accel.png", dpi=170)
print(f"saved {OUT}")
