"""Publication-style regeneration of ALL main-paper figures (review round 1, format pass).

Style: Okabe-Ito colorblind-safe palette, STIX fonts, no top/right spines, vector PDF.
Every figure is regenerated from cached per-head tables / trajectories — no GPU needed
(kernel figures are re-extracted separately by paper_kernel_refresh.py, which caches data).
Outputs: paper/figures/*.pdf (+ PNG previews for inspection).

Data provenance:
  fig1  gpt2_dirfrac_vs_Dhead      <- results/cache/gpt2_head_full.parquet
  fig2  paper_main_result          <- results/cache/{model}_head_full.parquet x7 (+ Table-3 constants)
  fig4  ckpt_dynamics_2models      <- results/cache/pythia-{410m,160m}_ckpt_analysis.parquet
  fig5  trainB_main                <- results/cache/trainB/*.parquet
Percentile convention: r/n (matches published table medians; verified 7/7).
"""
from __future__ import annotations
import json, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIGS = Path(__file__).resolve().parent.parent / "paper" / "figures"
PREV = Path("/tmp/claude-1005/-home-li-LLM-research-QK-analyze/057fe19d-01e8-4be9-aca1-4e8f4b2ad31f/scratchpad")

# ---- Okabe-Ito ----
BLUE, VERM, ORANGE, GREEN, SKY, PURPLE, YELLOW, GREY = (
    "#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442", "#7F7F7F")

def style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.4, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })

def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(PREV / f"{name}_preview.png", dpi=150)
    plt.close(fig)
    print(f"  saved {name}.pdf")

def tag(ax, s):
    ax.text(0.01, 1.02, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

# --------------------------------------------------------------- fig 1: GPT-2 plane
def fig1():
    d = pd.read_parquet(C.CACHE / "gpt2_head_full.parquet")
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    sc = ax.scatter(d["dir_frac"], d["D_head"], c=d["layer"], cmap="viridis",
                    s=16, alpha=0.85, linewidths=0)
    ax.axvline(1/np.sqrt(2), ls="--", c=GREY, lw=1)
    ax.axhline(0.608, ls=":", c=GREY, lw=1)
    ax.annotate("dir_frac null", xy=(1/np.sqrt(2), 0.02), xytext=(0.72, 0.02),
                fontsize=7, color="0.35")
    ax.annotate("$D_{\\mathrm{head}}$ null", xy=(0.25, 0.615), fontsize=7, color="0.35")
    ax.set_xlabel("dir_frac  $=\\|M_A\\|_F/\\|M\\|_F$")
    ax.set_ylabel("$D_{\\mathrm{head}}$")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=28)
    cb.set_label("layer", fontsize=8); cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "gpt2_dirfrac_vs_Dhead")

# --------------------------------------------------------------- fig 2: main result
def fig2():
    MODELS = [("gpt2", "GPT-2", "abs"), ("opt-1.3b", "OPT-1.3B", "abs"),
              ("gpt-neo-1.3b", "GPT-Neo-1.3B", "abs"), ("bloom-1b1", "BLOOM-1b1", "alibi"),
              ("pythia-410m", "Pythia-410m", "rope"), ("pythia-1.4b", "Pythia-1.4B", "rope"),
              ("llama-3-8b", "Llama-3-8B", "rope")]
    col = {"abs": VERM, "alibi": ORANGE, "rope": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    rng = np.random.default_rng(1)
    for i, (key, lab, sch) in enumerate(MODELS):
        d = pd.read_parquet(C.CACHE / f"{key}_head_full.parquet")
        pct = rankdata(d["D_head"]) / len(d)
        top5 = np.argsort(-d["prev"].values)[:5]
        v = pct[top5]
        ax.scatter(i + rng.uniform(-0.09, 0.09, 5), v, color=col[sch], s=26, alpha=0.85,
                   linewidths=0, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color=col[sch], lw=2.2, zorder=4,
                solid_capstyle="butt")
    ax.axhline(0.5, ls=":", c=GREY, lw=0.9)
    ax.set_xticks(range(7))
    ax.set_xticklabels([lab for _, lab, _ in MODELS], rotation=24, ha="right", fontsize=7.3)
    ax.set_ylabel("$D_{\\mathrm{head}}$ percentile (within model)")
    ax.set_ylim(0, 1.02)
    handles = [plt.Line2D([], [], marker="o", ls="", color=col[k], ms=5) for k in ("abs", "alibi", "rope")]
    ax.legend(handles, ["learned-absolute", "ALiBi", "RoPE"], ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0,
              columnspacing=1.2, handletextpad=0.3)
    tag(ax, "(a)")
    ax = axes[1]
    xs = [25, 25, 100]; hyb = [61, 38, 98]; labs = ["410m", "1.4B", "8B"]
    ax.scatter(xs, hyb, color=BLUE, s=42, zorder=3, linewidths=0)
    for x, y, l in zip(xs, hyb, labs):
        ax.annotate(l, (x, y), xytext=(6, 2), textcoords="offset points", fontsize=8)
    ax.set_xlabel("RoPE fraction of head dims (%)")
    ax.set_ylabel("attenuation by phase controls (%)")
    ax.set_xlim(0, 112); ax.set_ylim(0, 105)
    ax.text(0.03, 0.04, "rank-based estimator: 47 / 10 / 70%", transform=ax.transAxes,
            fontsize=7, color="0.35")
    tag(ax, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "paper_main_result")

# --------------------------------------------------------------- fig 4: ckpt dynamics
TOK_PER_STEP = 1024 * 2048  # Pythia batch tokens per optimizer step

def _traj(model):
    d = pd.read_parquet(C.CACHE / f"{model}_ckpt_analysis.parquet").sort_values("step")
    s = pd.read_parquet(C.CACHE / f"{model}_ckpt_summary.parquet").sort_values("step")
    d = d.merge(s[["step", "prev_D_head"]], on="step", how="left")
    rifp = [c for c in d.columns if c.startswith("rif_pct")][0]
    tok = np.maximum(d["step"].values, 1) * TOK_PER_STEP
    return tok, d, rifp, "prev_D_head"

def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
    models = [("pythia-410m", "Pythia-410m", BLUE), ("pythia-160m", "Pythia-160m", VERM)]
    for ax in axes:
        ax.set_xscale("log")
        ax.axvspan(1e9, 4e9, color="0.88", zorder=0)
        ax.set_xlabel("tokens")
    for m, lab, c in models:
        tok, d, rifp, dcol = _traj(m)
        axes[0].plot(tok, d["prev_beh"], marker=".", ms=4, color=c)
        axes[1].plot(tok, d[rifp], marker=".", ms=4, color=c)
        axes[2].plot(tok, d["pop_med_D"], marker=".", ms=4, color=c)
        axes[2].plot(tok, d[dcol], marker=".", ms=3, color=c, ls=":", alpha=0.8)
    axes[0].set_ylabel("prev-token attention of eventual prev head")
    tag(axes[0], "(a)")
    axes[1].set_ylabel("rope_imag_frac percentile (within model)")
    axes[1].axhline(0.5, ls=":", c=GREY, lw=0.9)
    tag(axes[1], "(b)")
    axes[2].set_ylabel("$D_{\\mathrm{head}}$")
    axes[2].axhline(0.608, ls="--", c=GREY, lw=0.9)
    tag(axes[2], "(c)")
    handles = [plt.Line2D([], [], color=BLUE, marker=".", ms=4, label="Pythia-410m"),
               plt.Line2D([], [], color=VERM, marker=".", ms=4, label="Pythia-160m"),
               plt.Line2D([], [], color="0.3", ls="-", label="population median (c)"),
               plt.Line2D([], [], color="0.3", ls=":", label="prev head (c)"),
               plt.Line2D([], [], color=GREY, ls="--", label="Ginibre null")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               frameon=False, columnspacing=1.4, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.6, rect=[0, 0.075, 1, 1])
    save(fig, "ckpt_dynamics_2models")

# --------------------------------------------------------------- fig 5: trainB main
def fig5():
    TB = C.CACHE / "trainB"
    conds = [("ape_free_lam0.0", "APE free", VERM, "-"),
             ("ape_sym_lam10.0", "APE sym-$M$", VERM, "--"),
             ("rope_free_lam0.0", "RoPE free", BLUE, "-"),
             ("rope_sym_lam10.0", "RoPE sym-$M$", BLUE, "--"),
             ("rope_imag_lam10.0", "RoPE Im-suppressed", ORANGE, "-.")]
    load = lambda c, s: pd.read_parquet(TB / f"{c}_seed{s}.parquet")
    thr = 0.5 * (math.log(64) + 0.15)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9))
    ax = axes[0, 0]
    for c, lab, col, ls in conds:
        cur = [load(c, s).ce_pred.values for s in range(5)]
        ax.plot(load(c, 0).step, np.mean(cur, 0), color=col, ls=ls, label=lab, lw=1.3)
    ax.axhline(thr, ls=":", c=GREY, lw=0.8)
    ax.set_xlabel("training step"); ax.set_ylabel("CE, induction-predictable tokens")
    ax.set_xlim(0, 4000)
    ax.legend(fontsize=7, handlelength=1.8, labelspacing=0.35, loc="upper right")
    tag(ax, "(a)")
    ax = axes[0, 1]; rng = np.random.default_rng(0)
    for i, (c, lab, col, ls) in enumerate(conds):
        v = [int(load(c, s)[load(c, s).ce_pred < thr].step.iloc[0]) for s in range(5)]
        ax.scatter(i + rng.uniform(-0.09, 0.09, 5), v, color=col, s=24, alpha=0.85,
                   linewidths=0, zorder=3)
        ax.plot([i - 0.24, i + 0.24], [np.mean(v)] * 2, c=col, lw=2.2, solid_capstyle="butt")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["APE\nfree", "APE\nsym-$M$", "RoPE\nfree", "RoPE\nsym-$M$", "RoPE\nIm-sup."],
                       fontsize=7.5)
    ax.set_ylabel("formation step")
    tag(ax, "(b)")
    ax = axes[1, 0]
    sigs = [("free", "rope_free_lam0.0"), ("sym-$M$", "rope_sym_lam10.0"),
            ("Im-sup.", "rope_imag_lam10.0")]
    w = 0.35
    for j, (lab, c) in enumerate(sigs):
        rifs, dirs = [], []
        for s in range(5):
            last = load(c, s).iloc[-1]
            bi = int(np.argmax(json.loads(last.prev_all))); wm = json.loads(last.wm)
            rifs.append(wm[bi]["rope_imag_frac"]); dirs.append(wm[bi]["dir_frac"])
        ax.bar(j - w / 2, np.mean(rifs), w, color=PURPLE, label="rope_imag_frac" if j == 0 else None)
        ax.bar(j + w / 2, np.mean(dirs), w, color=GREEN, label="dir_frac" if j == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for l, _ in sigs], fontsize=8)
    ax.set_ylabel("value of best prev head")
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=7, loc="upper right", handlelength=1.2)
    tag(ax, "(c)")
    ax = axes[1, 1]; offs = list(range(-8, 1))
    for c, lab, col, ls in [("rope_free_lam0.0", "free", BLUE, "-"),
                            ("rope_sym_lam10.0", "sym-$M$", GREEN, "--"),
                            ("rope_imag_lam10.0", "Im-sup.", ORANGE, "-.")]:
        ks = []
        for s in range(5):
            last = load(c, s).iloc[-1]
            bi = int(np.argmax(json.loads(last.prev_all)))
            k = np.array(json.loads(last.kernels)[bi]); ks.append(k - k.mean())
        ax.plot(offs, np.mean(ks, 0), color=col, ls=ls, marker=".", ms=4, label=lab, lw=1.3)
    ax.axvline(-1, ls=":", c=GREY, lw=0.8)
    ax.set_xlabel("relative offset $\\Delta$"); ax.set_ylabel("mean score (centered)")
    ax.legend(fontsize=7, loc="upper left", handlelength=1.7)
    tag(ax, "(d)")
    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    save(fig, "trainB_main")

if __name__ == "__main__":
    style()
    fig1(); fig2(); fig4(); fig5()
    print("done")
