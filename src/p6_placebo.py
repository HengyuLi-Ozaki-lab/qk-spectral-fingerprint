"""Placebo controls for the RoPE-phase attenuation story (REVIEW_P1 pending item 1).

Three controls, all from cached factors (CPU):
  (a) PERMUTED PAIRING (Pythia-410m/1.4B): randomly permute the rotary columns of the
      *folded* W_Q, W_K (same permutation on both sides — M is unchanged, only the
      pairing/frequency bookkeeping is scrambled), recompute rope_imag_frac /
      freq_centroid, and redo the attenuation of partial(D_head, prev | dir_frac).
      If the TRUE pairing carries the mechanism, scrambled pairing attenuates less.
  (b) PSEUDO-PAIRING on GPT-2 (no RoPE): compute "rope_imag_frac" with a fake
      rotate-half pairing over all 64 head dims; it should carry no prev signal.
  (c) NON-ROTARY BLOCK (Pythia): dir_frac of M restricted to the non-rotary dims;
      if directionality concentrates in the rotary block, this should NOT track prev
      positively.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_factors, centering_matrix
from rope import head_rope_metrics

ROPE_CFG = {"pythia-410m": (16, 10000.0), "pythia-1.4b": (32, 10000.0)}  # (rotary_dim, base)
N_PERM = 50


def resid(a, b):
    b = np.asarray(b, float); b = np.atleast_2d(b)
    if b.shape[0] != len(a): b = b.T
    A = np.column_stack([np.ones(len(a)), b])
    be, *_ = np.linalg.lstsq(A, a, rcond=None)
    return a - A @ be


def partial(df, x, y, ctrl):
    Cm = np.column_stack([df[c].values for c in ctrl])
    return spearmanr(resid(df[x].values, Cm), resid(df[y].values, Cm))[0]


def folded_factors(mk):
    fac = load_factors(mk)
    d = fac["meta"]["d_model"]
    Cm = centering_matrix(d) if fac["meta"]["norm"] == "LN" else None
    L, H = fac["meta"]["n_layers"], fac["meta"]["n_heads"]
    WQ = np.empty_like(fac["Wq"]); WK = np.empty_like(fac["Wk"])
    for l in range(L):
        for h in range(H):
            WQ[l, h] = Cm @ fac["Wq"][l, h] if Cm is not None else fac["Wq"][l, h]
            WK[l, h] = Cm @ fac["Wk"][l, h] if Cm is not None else fac["Wk"][l, h]
    return WQ, WK, fac["meta"]


def rope_cols(WQ, WK, meta, rd, base, perm=None):
    """rope_imag_frac & freq_centroid per head, optionally with permuted rotary columns."""
    npair = rd // 2
    theta = base ** (-2.0 * np.arange(npair) / rd)
    L, H = meta["n_layers"], meta["n_heads"]
    rif = np.zeros((L, H)); fc = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            q, k = WQ[l, h], WK[l, h]
            if perm is not None:
                q = q.copy(); k = k.copy()
                q[:, :rd] = q[:, perm]; k[:, :rd] = k[:, perm]
            m = head_rope_metrics(q, k, npair, theta)
            rif[l, h] = m["rope_imag_frac"]; fc[l, h] = m["freq_centroid"]
    return rif, fc


def main():
    out = {}
    print("=" * 70)
    print("(a) PERMUTED-PAIRING placebo (attenuation of partial(D_head,prev|dir_frac))")
    for mk, (rd, base) in ROPE_CFG.items():
        df = pd.read_parquet(C.CACHE / f"{mk}_head_full.parquet").reset_index(drop=True)
        WQ, WK, meta = folded_factors(mk)
        base_p = partial(df, "D_head", "prev", ["dir_frac"])
        # sanity: reproduce TRUE attenuation via the same code path
        rif_t, fc_t = rope_cols(WQ, WK, meta, rd, base)
        dt = df.copy(); dt["rif"] = [rif_t[int(r["layer"]), int(r["head"])] for _, r in dt.iterrows()]
        dt["fc"] = [fc_t[int(r["layer"]), int(r["head"])] for _, r in dt.iterrows()]
        chk = float(np.abs(dt["rif"] - dt["rope_imag_frac"]).max())
        true_ctrl = partial(dt, "D_head", "prev", ["dir_frac", "rif", "fc"])
        true_att = 100 * (1 - true_ctrl / base_p)
        # permuted pairing
        atts = []
        rng = np.random.default_rng(0)
        for s in range(N_PERM):
            perm = rng.permutation(rd)
            rif_p, fc_p = rope_cols(WQ, WK, meta, rd, base, perm=perm)
            dp = df.copy()
            dp["rif"] = [rif_p[int(r["layer"]), int(r["head"])] for _, r in dp.iterrows()]
            dp["fc"] = [fc_p[int(r["layer"]), int(r["head"])] for _, r in dp.iterrows()]
            ctrl = partial(dp, "D_head", "prev", ["dir_frac", "rif", "fc"])
            atts.append(100 * (1 - ctrl / base_p))
        atts = np.array(atts)
        print(f"  {mk:12s} recompute-check max|Δrif|={chk:.1e} | base partial={base_p:+.3f}")
        print(f"     TRUE pairing attenuation   = {true_att:5.1f}%")
        print(f"     PERMUTED pairing (n={N_PERM})   = {atts.mean():5.1f}% ± {atts.std():.1f}  "
              f"(range {atts.min():.0f}–{atts.max():.0f}%)")
        out[mk] = dict(base_partial=base_p, true_attenuation=true_att,
                       perm_attenuation_mean=float(atts.mean()), perm_attenuation_std=float(atts.std()),
                       perm_attenuations=atts.tolist(), recompute_check=chk)

    print("\n(b) PSEUDO-PAIRING on GPT-2 (no RoPE — should carry no prev signal)")
    dfg = pd.read_parquet(C.CACHE / "gpt2_head_full.parquet").reset_index(drop=True)
    WQ, WK, meta = folded_factors("gpt2")
    rd, base = 64, 10000.0                     # pretend full-dim rotate-half pairing
    rif_g, fc_g = rope_cols(WQ, WK, meta, rd, base)
    dfg["rif"] = [rif_g[int(r["layer"]), int(r["head"])] for _, r in dfg.iterrows()]
    dfg["fc"] = [fc_g[int(r["layer"]), int(r["head"])] for _, r in dfg.iterrows()]
    r1, p1 = spearmanr(dfg["rif"], dfg["prev"])
    base_g = partial(dfg, "D_head", "prev", ["dir_frac"])
    ctrl_g = partial(dfg, "D_head", "prev", ["dir_frac", "rif", "fc"])
    print(f"  corr(pseudo rope_imag_frac, prev) = {r1:+.3f} (p={p1:.2f})   [RoPE models: +0.42..+0.53]")
    print(f"  GPT-2 partial {base_g:+.3f} -> +pseudo-phase {ctrl_g:+.3f}  (attenuation {100*(1-ctrl_g/base_g):.0f}% of a NEGATIVE partial)")
    out["gpt2_pseudo"] = dict(corr_rif_prev=r1, p=p1, base_partial=base_g, ctrl_partial=ctrl_g)

    print("\n(c) NON-ROTARY-BLOCK directionality (Pythia) — does it track prev?")
    for mk, (rd, _) in ROPE_CFG.items():
        df = pd.read_parquet(C.CACHE / f"{mk}_head_full.parquet").reset_index(drop=True)
        WQ, WK, meta = folded_factors(mk)
        L, H = meta["n_layers"], meta["n_heads"]
        dnr = np.zeros((L, H))
        for l in range(L):
            for h in range(H):
                Mnr = WQ[l, h][:, rd:] @ WK[l, h][:, rd:].T
                dnr[l, h] = np.linalg.norm(0.5 * (Mnr - Mnr.T)) / (np.linalg.norm(Mnr) + 1e-30)
        df["dnr"] = [dnr[int(r["layer"]), int(r["head"])] for _, r in df.iterrows()]
        r2, p2 = spearmanr(df["dnr"], df["prev"])
        print(f"  {mk:12s} corr(non-rotary dir_frac, prev) = {r2:+.3f} (p={p2:.1e})")
        out[f"{mk}_nonrotary"] = dict(corr=r2, p=p2)

    (C.CACHE / "placebo_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {C.CACHE / 'placebo_results.json'}")


if __name__ == "__main__":
    main()
