"""Re-extract the two kernel figures' curves (Pythia-410m) with data caching + new style.
Reuses p8_remaining.corpus_kernels' exact computation (corpus n=48, seq_len=96, K=20).
Saves curves -> results/cache/pythia-410m_kernel_curves.parquet, figures -> paper/figures/*.pdf.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model
from observables import get_tokens
from rope import rope_params
from p5_rope_ablation import _hooks
from paper_figs_v2 import style, save, tag, BLUE, VERM, ORANGE, GREEN, GREY

DEV = "cuda"
N_SEQ, SEQ_LEN, K = 48, 96, 20

@torch.no_grad()
def main():
    model = load_model("pythia-410m")
    _, npair, theta = rope_params(model.cfg)
    df = pd.read_parquet(C.CACHE / "pythia-410m_head_full.parquet")
    heads = [tuple(map(int, r)) for r in df.sort_values("prev", ascending=False).head(4)[["layer", "head"]].values]
    toks = get_tokens(model, N_SEQ, SEQ_LEN).to(DEV)
    rows = []
    # (1) relpos kernels of top-4 prev heads
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_attn_scores"))
    curves = {}
    for l, h in heads:
        S = cache[f"blocks.{l}.attn.hook_attn_scores"][:, h].float().cpu().numpy()
        P = S.shape[1]
        ker = [float(np.mean([S[:, i, i + dl] for i in range(P) if 0 <= i + dl < P])) for dl in range(-K, 1)]
        curves[f"{l}.{h}"] = ker
        rows += [dict(fig="relpos", head=f"{l}.{h}", offset=dl, score=v)
                 for dl, v in zip(range(-K, 1), ker)]
    del cache
    # (2) intact vs Im(M_t)-killed for head 5.2
    ab = {}
    for tagname, hooks in [("intact", []), ("killed", _hooks(model, 5, 2, npair, theta))]:
        with model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_attn_scores"))
        S = cache[f"blocks.5.attn.hook_attn_scores"][:, 2].float().cpu().numpy()
        P = S.shape[1]
        ab[tagname] = [float(np.mean([S[:, i, i + dl] for i in range(P) if 0 <= i + dl < P])) for dl in range(-K, 1)]
        rows += [dict(fig="ablate_5_2", head=tagname, offset=dl, score=v)
                 for dl, v in zip(range(-K, 1), ab[tagname])]
        del cache
    pd.DataFrame(rows).to_parquet(C.CACHE / "pythia-410m_kernel_curves.parquet")
    print("curves cached")
    # ---- plots (new style) ----
    style()
    offs = list(range(-K, 1))
    cols = [BLUE, ORANGE, GREEN, VERM]
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for (name, ker), c in zip(curves.items(), cols):
        ax.plot(offs, ker, marker=".", ms=3.5, lw=1.2, color=c, label=name)
    ax.axvline(-1, ls=":", c=GREY, lw=0.9)
    ax.set_xlabel("relative offset $\\Delta$ = key $-$ query")
    ax.set_ylabel("mean pre-softmax score")
    ax.legend(fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0, columnspacing=0.9, handlelength=1.2, handletextpad=0.35)
    fig.tight_layout(); save(fig, "pythia-410m_relpos_kernel")
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.plot(offs, ab["intact"], marker=".", ms=3.5, lw=1.2, color=BLUE, label="intact")
    ax.plot(offs, ab["killed"], marker=".", ms=3.5, lw=1.2, color=ORANGE, ls="--", label="Im($M_t$) killed")
    ax.axvline(-1, ls=":", c=GREY, lw=0.9)
    ax.set_xlabel("relative offset $\\Delta$")
    ax.set_ylabel("mean pre-softmax score")
    ax.legend(fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(); save(fig, "pythia-410m_ablate_kernel_5_2")

if __name__ == "__main__":
    main()
