"""Remaining review items (REVIEW_P1): K-composition verification, corpus-based kernel
figures, per-model null integration.

(A) K-composition: does the induction head's QK read the prev-token head's OV output?
    Column convention: score gain = x_i^T M_eff^{h2} (W_V W_O)^{h1,T} v, so
    r(h1→h2) = ‖M_eff^{h2} @ OV^{h1}ᵀ‖_F / (‖M_eff‖_F ‖OV‖_F), ranked over all earlier h1.
    Prediction: the canonical prev head ranks top for each induction head.
(B) Kernel figures from a CORPUS sample (fixes the single-duplicated-sentence caveat).
(C) 410m per-model null census integration: z-scores for prev heads vs population.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model, centering_matrix
from observables import get_tokens
from rope import rope_params
from p5_rope_ablation import _hooks

DEV = "cuda"


@torch.no_grad()
def kcomp_ranks(model, targets, prev_named, norm="LN"):
    """For each target (l2,h2): rank of each prev-head candidate by r(h1→h2) over all earlier heads."""
    d = model.cfg.d_model
    Cm = torch.eye(d, device=DEV) - 1.0 / d if norm == "LN" else torch.eye(d, device=DEV)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    out = []
    for (l2, h2) in targets:
        M = Cm @ (model.W_Q[l2, h2].float() @ model.W_K[l2, h2].float().T) @ Cm
        nM = torch.linalg.norm(M)
        rows = []
        for l1 in range(l2):
            for h1 in range(H):
                OV = model.W_V[l1, h1].float() @ model.W_O[l1, h1].float()      # [d,d] row-acting
                r = float(torch.linalg.norm(M @ OV.T) / (nM * torch.linalg.norm(OV) + 1e-30))
                rows.append((l1, h1, r))
        df = pd.DataFrame(rows, columns=["l1", "h1", "r"]).sort_values("r", ascending=False).reset_index(drop=True)
        n = len(df)
        for name, (pl, ph) in prev_named.items():
            if pl < l2:
                rank = int(df[(df.l1 == pl) & (df.h1 == ph)].index[0]) + 1
                out.append(dict(target=f"{l2}.{h2}", prev=name, rank=rank, n=n,
                                r=float(df[(df.l1 == pl) & (df.h1 == ph)].r.iloc[0]),
                                top1=f"{df.l1[0]}.{df.h1[0]}"))
    return pd.DataFrame(out)


@torch.no_grad()
def corpus_kernels(model, model_key, heads, ablate_head, npair, theta, n_seq=48, seq_len=96):
    toks = get_tokens(model, n_seq, seq_len).to(DEV)
    K = 20
    # (1) relative-position kernels, corpus-averaged
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_attn_scores"))
    plt.figure(figsize=(6, 4))
    for l, h in heads:
        S = cache[f"blocks.{l}.attn.hook_attn_scores"][:, h].float().cpu().numpy()
        P = S.shape[1]
        ker = [np.mean([S[:, i, i + dl] for i in range(P) if 0 <= i + dl < P]) for dl in range(-K, 1)]
        plt.plot(range(-K, 1), ker, marker=".", label=f"{l}.{h}")
    plt.axvline(-1, ls="--", c="grey", lw=1)
    plt.xlabel("relative offset $\\Delta$ = key $-$ query"); plt.ylabel("mean pre-softmax score")
    plt.title(f"{model_key}: prev-token heads peak at $\\Delta=-1$ (corpus, n={n_seq})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(C.FIGS / f"{model_key}_relpos_kernel.png", dpi=120); plt.close()
    del cache
    # (2) before/after Im(M_t) ablation kernel for the causal head
    l, h = ablate_head
    plt.figure(figsize=(6, 4))
    for tag, hooks in [("intact", []), ("Im($M_t$) killed", _hooks(model, l, h, npair,
                                                                   torch.tensor(theta, device=DEV, dtype=torch.float32) if False else theta))]:
        with model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_attn_scores"))
        S = cache[f"blocks.{l}.attn.hook_attn_scores"][:, h].float().cpu().numpy()
        P = S.shape[1]
        ker = [np.mean([S[:, i, i + dl] for i in range(P) if 0 <= i + dl < P]) for dl in range(-K, 1)]
        plt.plot(range(-K, 1), ker, marker=".", label=tag)
        del cache
    plt.axvline(-1, ls="--", c="grey", lw=1)
    plt.xlabel("relative offset $\\Delta$"); plt.ylabel("mean pre-softmax score")
    plt.title(f"{model_key} head {l}.{h}: killing Im($M_t$) flattens $\\Delta=-1$ (corpus)")
    plt.legend(); plt.tight_layout()
    plt.savefig(C.FIGS / f"{model_key}_ablate_kernel_{l}_{h}.png", dpi=120); plt.close()


def main():
    # ---------- (A) K-composition on GPT-2 ----------
    print("=" * 70); print("(A) K-composition — GPT-2")
    m = load_model("gpt2", DEV)
    tg = kcomp_ranks(m, targets=[(5, 1), (5, 5), (6, 9)], prev_named={"4.11(prev)": (4, 11)})
    print(tg.to_string(index=False))
    del m; torch.cuda.empty_cache()

    # ---------- (A') K-composition + (B) kernels on Pythia-410m ----------
    print("=" * 70); print("(A') K-composition — Pythia-410m")
    m = load_model("pythia-410m", DEV)
    full = pd.read_parquet(C.CACHE / "pythia-410m_head_full.parquet")
    ind_heads = [(int(r["layer"]), int(r["head"])) for _, r in
                 full[full["ind"] > 0.8].sort_values("ind", ascending=False).head(3).iterrows()]
    print(f"   induction heads (ind>0.8): {ind_heads}")
    tp = kcomp_ranks(m, targets=ind_heads, prev_named={"5.2(prev)": (5, 2)})
    print(tp.to_string(index=False))
    print("=" * 70); print("(B) corpus kernel figures — Pythia-410m")
    npair, theta = rope_params(m.cfg)[1], rope_params(m.cfg)[2]
    top_prev = [(int(r["layer"]), int(r["head"])) for _, r in
                full.sort_values("prev", ascending=False).head(4).iterrows()]
    corpus_kernels(m, "pythia-410m", top_prev, (5, 2), npair, theta)
    print(f"   figures regenerated -> {C.FIGS}")
    del m; torch.cuda.empty_cache()

    # ---------- (C) 410m null census integration ----------
    print("=" * 70); print("(C) per-model null integration — Pythia-410m")
    met = pd.read_parquet(C.CACHE / "pythia-410m_head_metrics.parquet")
    d = full.merge(met[["layer", "head", "D_head_null_mean", "D_head_null_std",
                        "dir_frac_null_mean", "dir_frac_null_std"]], on=["layer", "head"])
    dz = (d.D_head - d.D_head_null_mean) / d.D_head_null_std
    fz = (d.dir_frac - d.dir_frac_null_mean) / d.dir_frac_null_std
    top = d.sort_values("prev", ascending=False).head(5).index
    print(f"   D_head null mean={d.D_head_null_mean.mean():.3f}")
    print(f"   top-5 prev heads: D_head z = {[f'{v:+.1f}' for v in dz[top]]},  dir_frac z = {[f'{v:+.1f}' for v in fz[top]]}")
    print(f"   population median z: D_head {dz.median():+.1f},  dir_frac {fz.median():+.1f}")


if __name__ == "__main__":
    main()
