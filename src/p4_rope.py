"""P4 — RoPE replication (spec §8). Does the GPT-2 verdict hold on a RoPE model?

RoPE: score = q_iᵀ R_{j-i} k_j. The static M = W_Q W_Kᵀ is the CONTENT operator at
zero relative position; at i=j the rotation cancels (score_ii = n_iᵀ M n_i), so we
validate the convention on the DIAGONAL. We then recompute the P1 spectral metrics
(fast low-rank path, since d_model is large) + P2b taxonomy on Pythia and compare.

Key questions:
  Q1  Is the content M more symmetric in a RoPE model (positional routing offloaded
      to RoPE phase)?  → compare dir_frac distribution to GPT-2.
  Q2  Does static-M directionality still flag prev-token heads, or has RoPE taken over?
  Q3  Does D_head still fail to beat dir_frac (F3)?
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model, get_factors, cache_factors, centering_matrix
from metrics import kspace_spectrum
from semantics import qk_self_match
from observables import taxonomy_scores, copying_scores

DEV = "cuda"
RTOL = 1e-9


@torch.no_grad()
def diagonal_logit_match(model, fac):
    """Validate M = W_Q W_Kᵀ via the i=j diagonal (RoPE cancels there)."""
    toks = model.to_tokens(["The cat sat on the mat.", "Attention is all you need."])
    _, cache = model.run_with_cache(toks)
    scale = fac["meta"]["attn_scale"]; L = fac["meta"]["n_layers"]
    worst = 0.0
    for l in range(L):
        n = cache[f"blocks.{l}.ln1.hook_normalized"].float().cpu().numpy().astype(np.float64)
        q = np.einsum("bpd,hde->bphe", n, fac["Wq"][l]) + fac["bQ"][l]
        k = np.einsum("bpd,hde->bphe", n, fac["Wk"][l]) + fac["bK"][l]
        diag = np.einsum("bphe,bphe->bph", q, k) / scale                # score_ii
        ref = cache[f"blocks.{l}.attn.hook_attn_scores"].float().cpu().numpy().astype(np.float64)
        ref_diag = np.diagonal(ref, axis1=2, axis2=3)                   # [b,h,pos]
        worst = max(worst, float(np.abs(diag.transpose(0, 2, 1) - ref_diag).max()))
    return worst


def thin_svd(Wqe, Wke):
    Q1, R1 = np.linalg.qr(Wqe); Q2, R2 = np.linalg.qr(Wke)
    Ux, Sx, Vxt = np.linalg.svd(R1 @ R2.T)
    keep = Sx > RTOL * Sx[0]
    return (Q1 @ Ux)[:, keep], Sx[keep], (Q2 @ Vxt.T)[:, keep]          # Uk, Sk, Vk


def head_metrics_fast(fac, l, h, Cmat, WE_gpu):
    Wqe = Cmat @ fac["Wq"][l, h] if Cmat is not None else fac["Wq"][l, h]
    Wke = Cmat @ fac["Wk"][l, h] if Cmat is not None else fac["Wk"][l, h]
    Uk, Sk, Vk = thin_svd(Wqe, Wke)
    spec = kspace_spectrum(Sk, Vk.T @ Uk)                               # dir_frac, D_head, ...
    # content_pos_frac via M_S projected onto span([Uk,Vk])
    Bb, _ = np.linalg.qr(np.concatenate([Uk, Vk], axis=1))
    au, av = Bb.T @ Uk, Bb.T @ Vk
    BMB = (au * Sk) @ av.T
    es = np.linalg.eigvalsh(0.5 * (BMB + BMB.T))
    es = es[np.abs(es) > RTOL * np.abs(es).max()]
    cpf = float(es[es > 0].sum() / (np.abs(es).sum() + 1e-300))
    sm = float(qk_self_match(WE_gpu,
                             torch.tensor(Uk, device=DEV, dtype=WE_gpu.dtype),
                             torch.tensor(Sk, device=DEV, dtype=WE_gpu.dtype),
                             torch.tensor(Vk, device=DEV, dtype=WE_gpu.dtype)))
    return dict(layer=l, head=h, rank=len(Sk), content_pos_frac=cpf, self_match=sm,
                dir_frac=spec["dir_frac"], D_head=spec["D_head"], imag_mass=spec["imag_mass"],
                henrici=spec["henrici"], spectral_radius=spec["spectral_radius"])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="pythia-1.4b"); args = ap.parse_args()
    print(f"[P4] loading {args.model} (RoPE) on GPU1 ...")
    model = load_model(args.model, DEV)
    fac = get_factors(model, args.model)
    if fac["meta"]["d_model"] <= 2048:            # skip multi-GB factor cache for large models (Llama)
        cache_factors(fac, args.model)
    m = fac["meta"]; print(f"[P4] cfg: {m}")

    w = diagonal_logit_match(model, fac)
    print(f"[P4] diagonal logit-match (i=j, RoPE cancels): max|Δ|={w:.2e}  ({'PASS' if w<1e-3 else 'CHECK'})")

    Cmat = centering_matrix(m["d_model"]) if m["norm"] == "LN" else None
    WE_gpu = model.W_E.detach()
    # validate fast low-rank path against full-matrix dir_frac on head (0,0)
    from extract import M_eff as _Meff
    M00 = _Meff(fac, 0, 0, Cmat); df0 = np.linalg.norm(0.5 * (M00 - M00.T)) / np.linalg.norm(M00)
    fm0 = head_metrics_fast(fac, 0, 0, Cmat, WE_gpu)
    assert abs(fm0["dir_frac"] - df0) < 1e-7, f"fast-path bug: {fm0['dir_frac']} vs {df0}"
    print(f"[P4] fast-path validated (dir_frac Δ={abs(fm0['dir_frac']-df0):.1e})")
    print(f"[P4] spectral metrics over {m['n_layers']*m['n_heads']} heads (fast low-rank) ...")
    rows = [head_metrics_fast(fac, l, h, Cmat, WE_gpu)
            for l in range(m["n_layers"]) for h in range(m["n_heads"])]
    df = pd.DataFrame(rows)

    print("[P4] taxonomy (prev/dup/induction) + copying ...")
    tax = taxonomy_scores(model, n_seq=200, seq_len=64)
    cop = copying_scores(model)
    for name, grid in [("prev", tax["prev"]), ("dup", tax["dup"]), ("ind", tax["ind"]), ("copying", cop)]:
        df[name] = [grid[int(r["layer"]), int(r["head"])] for _, r in df.iterrows()]
    df.to_parquet(C.CACHE / f"{args.model}_head_full.parquet")

    # ---- comparison to GPT-2 ----
    g = pd.read_parquet(C.CACHE / "gpt2_head_full.parquet")
    print("\n[P4] ===== RoPE (Pythia) vs learned-absolute (GPT-2) =====")
    print(f"  dir_frac  median:  GPT-2 {g.dir_frac.median():.3f}   Pythia {df.dir_frac.median():.3f}")
    print(f"  D_head    median:  GPT-2 {g.D_head.median():.3f}   Pythia {df.D_head.median():.3f}")
    print(f"  content_pos_frac:  GPT-2 {g.content_pos_frac.median():.3f}   Pythia {df.content_pos_frac.median():.3f}")

    def corrs(d, tag):
        print(f"\n  [{tag}] taxonomy correlations with spectral metrics:")
        for t in ["prev", "dup", "ind"]:
            for pcol in ["dir_frac", "D_head", "content_pos_frac", "self_match"]:
                rho, p = spearmanr(d[pcol], d[t])
                print(f"     corr({pcol:16s}, {t:4s}) = {rho:+.3f} (p={p:.1e})", end="   ")
            print()
    corrs(g, "GPT-2")
    corrs(df, "Pythia")

    print("\n[P4] Q3 — does D_head beat dir_frac for prev-token? (both models)")
    for d, tag in [(g, "GPT-2"), (df, "Pythia")]:
        rd, _ = spearmanr(d.dir_frac, d.prev); rD, _ = spearmanr(d.D_head, d.prev)
        print(f"   {tag}: corr(dir_frac,prev)={rd:+.3f}  corr(D_head,prev)={rD:+.3f}  "
              f"→ D_head {'beats' if abs(rD)>abs(rd) else 'loses to'} dir_frac")
    print(f"\n[P4] saved → {C.CACHE / (args.model + '_head_full.parquet')}")


if __name__ == "__main__":
    main()
