"""P3 — semantics of the flagged heads (spec §7).

Two robust quantitative probes across all heads (QK SVD directions are known-noisy,
Millidge & Black 2022), plus qualitative token projections for the circuit-critical heads:

  self_match  : diagonal dominance of the full QK circuit C = W_E·M·W_Eᵀ — does a head,
                by content, prefer to attend query-token→same key-token? (duplicate/content signature)
  pos_frac_MA : fraction of the antisymmetric part M_A acting within the positional-embedding
                subspace (top PCs of W_pos) — is a head's directionality POSITIONAL routing?
  token proj  : W_E projection of top query-read (u_r) / key-read (v_r) SVD directions.

All the scalar probes are computed in the rank-k core (cheap, exact).
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import decompose as D
from extract import load_model, load_factors, centering_matrix, M_eff

DEV = "cuda"
FLAGGED = [(5, 1, "induction"), (4, 11, "prev-token"), (6, 9, "induction"),
           (0, 10, "duplicate"), (0, 1, "duplicate"), (10, 10, "directional")]


@torch.no_grad()
def qk_self_match(WE, Uk, Sk, Vk):
    """Diagonal dominance of C = W_E·M·W_Eᵀ via the low-rank core (all torch on GPU)."""
    A = WE @ Uk; B = WE @ Vk                      # [vocab,k]
    V = A.shape[0]
    diag_sum = float(((A * B) * Sk).sum())        # Σ_t Σ_r S_r A_tr B_tr
    csA = A.sum(0); csB = B.sum(0)                 # [k]
    total = float((csA * Sk * csB).sum())         # Σ_{s,t} C_st
    fro2 = float((((A.T @ A) * Sk) * ((B.T @ B) * Sk).T).sum())  # ‖C‖_F²  (trace form)
    diag_mean = diag_sum / V
    off_mean = (total - diag_sum) / (V * V - V)
    rms = (fro2 / (V * V)) ** 0.5 + 1e-30
    return (diag_mean - off_mean) / rms


def pos_energy(Qpos, Uk, Sk, Vk):
    """Fraction of M and M_A acting within the positional subspace span(Qpos)."""
    a = Qpos.T @ Uk; b = Qpos.T @ Vk              # [kp,k]
    core = (a * Sk) @ b.T                          # Qposᵀ M Qpos  [kp,kp]
    fro_M = np.sqrt(np.sum(Sk ** 2))
    pos_M = np.linalg.norm(core) / (fro_M + 1e-30)
    coreA = 0.5 * (core - core.T)
    # ‖M_A‖ from core-independent full: use fro of antisym of full M via Sk,B? approximate with total M_A norm
    return pos_M, coreA, core


def tokens_of(model, vec, k=8):
    v = torch.tensor(vec, device=DEV, dtype=model.W_E.dtype)
    proj = model.W_E @ v                           # [vocab]  (embedding-side read)
    top = proj.topk(k).indices.tolist()
    return [repr(model.tokenizer.decode([t])) for t in top]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gpt2"); args = ap.parse_args()
    model = load_model(args.model, DEV)
    fac = load_factors(args.model)
    m = fac["meta"]; Cmat = centering_matrix(m["d_model"]) if m["norm"] == "LN" else None
    df = pd.read_parquet(C.CACHE / f"{args.model}_head_full.parquet").reset_index(drop=True)

    WE = model.W_E.detach()                        # [vocab,d]
    # positional subspace (top-64 PCs of W_pos)
    Wpos = model.W_pos.detach().cpu().numpy().astype(np.float64)
    _, _, VtP = np.linalg.svd(Wpos, full_matrices=False)
    Qpos = VtP[:64].T                              # [d,64]

    sm, pfM, pfMA = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for i, r in df.iterrows():
        l, h = int(r["layer"]), int(r["head"])
        M = M_eff(fac, l, h, Cmat)
        U, S, Vt = D.svd(M); k = int((S > 1e-9 * S[0]).sum())
        Uk, Sk, Vk = U[:, :k], S[:k], Vt[:k].T
        sm[i] = qk_self_match(WE, torch.tensor(Uk, device=DEV, dtype=WE.dtype),
                              torch.tensor(Sk, device=DEV, dtype=WE.dtype),
                              torch.tensor(Vk, device=DEV, dtype=WE.dtype))
        pfM[i], _, _ = pos_energy(Qpos, Uk, Sk, Vk)
        # M_A positional fraction: ‖Qposᵀ M_A Qpos‖ / ‖M_A‖  (full M_A norm)
        MA = 0.5 * (M - M.T); a = Qpos.T @ Uk; b = Qpos.T @ Vk
        coreA = 0.5 * ((a * Sk) @ b.T - (b * Sk) @ a.T)
        pfMA[i] = np.linalg.norm(coreA) / (np.linalg.norm(MA) + 1e-30)
    df["self_match"] = sm; df["pos_frac_M"] = pfM; df["pos_frac_MA"] = pfMA
    df.to_parquet(C.CACHE / f"{args.model}_head_full.parquet")

    from scipy.stats import spearmanr
    print("\n[P3] QK-circuit self-matching (content signature) vs head type:")
    for t in ["dup", "content_pos_frac", "dir_frac", "prev", "ind"]:
        rho, p = spearmanr(df[t], df["self_match"]); print(f"   corr(self_match, {t:16s}) = {rho:+.3f} (p={p:.1e})")
    print("\n[P3] positional-subspace energy of M_A vs head type:")
    for t in ["prev", "dir_frac", "D_head", "ind", "dup"]:
        rho, p = spearmanr(df[t], df["pos_frac_MA"]); print(f"   corr(pos_frac_MA, {t:16s}) = {rho:+.3f} (p={p:.1e})")

    print("\n[P3] flagged-head profiles + token projections (W_E-side; QK SVD is noisy — read qualitatively):")
    for l, h, lab in FLAGGED:
        row = df[(df["layer"] == l) & (df["head"] == h)].iloc[0]
        M = M_eff(fac, l, h, Cmat); U, S, Vt = D.svd(M)
        print(f"\n  === {l}.{h} [{lab}] dir_frac={row.dir_frac:.2f} D_head={row.D_head:.2f} "
              f"self_match={row.self_match:+.1f} pos_frac_MA={row.pos_frac_MA:.2f} "
              f"prev={row.prev:.2f} dup={row.dup:.2f} ind={row.ind:.2f} dCE_sym={row.dCE_sym:+.3f} "
              f"dICL_sym={row.get('dICL_sym', float('nan')):+.3f}")
        for r in range(2):
            print(f"     dir{r} (σ={S[r]:.1f})  query-reads(u): {tokens_of(model, U[:, r])}")
            print(f"               key-offers(v): {tokens_of(model, Vt[r])}")

    # figure: self_match & pos_frac_MA heatmaps
    for mcol in ["self_match", "pos_frac_MA"]:
        grid = df.pivot(index="layer", columns="head", values=mcol).values
        plt.figure(figsize=(5, 4)); plt.imshow(grid, aspect="auto", cmap="coolwarm", origin="lower")
        plt.colorbar(label=mcol); plt.xlabel("head"); plt.ylabel("layer"); plt.title(f"{args.model}: {mcol}")
        plt.tight_layout(); plt.savefig(C.FIGS / f"{args.model}_{mcol}_heatmap.png", dpi=120); plt.close()
    print(f"\n[P3] figures → {C.FIGS}")


if __name__ == "__main__":
    main()
