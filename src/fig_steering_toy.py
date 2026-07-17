"""Figure 1 of the steering paper, restyled to the P1/P2 family style
(Okabe-Ito, STIX serif, no top/right spines, vector PDF).
Data: results/h1/m12_summary.parquet (per-run formation steps, toy hinge grid)."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures" / "m12_formation.pdf"
PREV = Path("/tmp/claude-1005/-home-li-LLM-research-QK-analyze/057fe19d-01e8-4be9-aca1-4e8f4b2ad31f/scratchpad")

BLUE, VERM, ORANGE, GREEN, SKY, PURPLE, GREY = (
    "#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#7F7F7F")

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.6, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False, "pdf.fonttype": 42,
})

d = pd.read_parquet(ROOT / "results" / "h1" / "m12_summary.parquet")

ARMS = [("free", None, "free", GREY),
        ("assist_init_algebra", None, "algebra-init", SKY),
        ("assist_init_solution", None, "solution-init", BLUE),
        ("assist_reg", 1.0, "reg $\\lambda{=}1$", VERM),
        ("assist_reg", 10.0, "reg $\\lambda{=}10$", PURPLE),
        ("placebo_cross", None, "placebo: cross", ORANGE),
        ("placebo_random", None, "placebo: random", GREEN)]

def sel(scheme, cons, lam):
    m = (d.scheme == scheme) & (d.cons == cons)
    if lam is not None:
        m &= np.isclose(d.lam, lam)
    return d[m].fstep.values

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
rng = np.random.default_rng(2)
for ax, scheme, ptitle in [(axes[0], "rope", "RoPE"), (axes[1], "ape", "APE")]:
    free_med = np.median(sel(scheme, "free", None))
    ax.axhline(free_med, ls=(0, (4, 4)), c=GREY, lw=0.9, zorder=1)
    for i, (cons, lam, lab, col) in enumerate(ARMS):
        v = sel(scheme, cons, lam)
        if len(v) == 0:
            continue
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(v)), v, s=26, color=col,
                   alpha=0.85, linewidths=0, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=col, lw=2.2,
                solid_capstyle="butt", zorder=4)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[2] for a in ARMS], rotation=32, ha="right")
    ax.set_title(ptitle, fontsize=9.5)
axes[0].set_ylabel("formation step")
axes[0].text(0.01, 1.06, "(a)", transform=axes[0].transAxes, fontsize=10, fontweight="bold")
axes[1].text(0.01, 1.06, "(b)", transform=axes[1].transAxes, fontsize=10, fontweight="bold")
fig.tight_layout(w_pad=2.0)
fig.savefig(OUT); fig.savefig(PREV / "m12_formation_new.png", dpi=170)
print(f"saved {OUT}")
