"""P1 — per-head scalar metrics + the random-null baseline (RESEARCH_PLAN §2).

Central metrics:
  dir_frac   = ‖M_A‖_F/‖M‖_F            (published baseline ≙ Saponati; dir_frac²=(1−s)/2)
  D_head     = Σ|Im λ| / Σ|λ|           (PRIMARY — complex/Schur directionality)
  imag_mass  = Σ|Im λ| / ‖M‖_F
  henrici    = √(‖M‖²−Σ|λ|²)/‖M‖        (non-normality = where D_head departs from dir_frac)
  cond_eig   = cond(eigvecs)            (near-defectiveness / exceptional-point flag)
  content_pos_frac = Σλ⁺/Σ|λ| of M_S   (QK-side echo of Elhage OV positivity)

Everything the spectral hypothesis tests is the DEVIATION FROM A MATCHED RANDOM NULL:
a random-orientation operator with the SAME singular values (dir_frac_null≈0.71, D_head
null also high). Reported as z = (real − null_mean)/null_std.

Fast null: nonzero eig(M)=eig(diag(Sₖ)·B) with B=Q2ᵀQ1 (k×k), so real head and every
null draw share one k-space routine — no 768×768 eig in the inner loop.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import decompose as D
from extract import load_factors, centering_matrix, M_eff

RANK_TOL = 1e-9


def kspace_spectrum(Sk: np.ndarray, B: np.ndarray) -> dict:
    """All spectral metrics of M = (orientation)·diag(Sk)·(orientation) from the k×k core."""
    core = Sk[:, None] * B                      # diag(Sk) @ B  (k×k), real
    fro2 = float(np.sum(Sk ** 2))
    tr2 = float(np.trace(core @ core).real)
    dir_frac = float(np.sqrt(max(0.5 * (1.0 - tr2 / fro2), 0.0)))
    w, V = np.linalg.eig(core)                   # nonzero eigenvalues of M
    absw = np.abs(w); imag = np.abs(w.imag)
    denom = absw.sum() + 1e-300
    try:
        cond = float(np.linalg.cond(V))
    except np.linalg.LinAlgError:
        cond = np.inf
    return dict(
        dir_frac=dir_frac,
        D_head=float(imag.sum() / denom),
        imag_mass=float(imag.sum() / (np.sqrt(fro2) + 1e-300)),
        henrici=float(np.sqrt(max(fro2 - np.sum(absw ** 2), 0.0)) / (np.sqrt(fro2) + 1e-300)),
        cond_eig=cond,
        spectral_radius=float(absw.max()) if absw.size else 0.0,
    )


def _rand_orthonormal(d, k, rng, Cmat):
    G = rng.standard_normal((d, k))
    if Cmat is not None:
        G = Cmat @ G                             # keep null in the centered subspace
    Q, _ = np.linalg.qr(G)
    return Q[:, :k]


def null_distribution(Sk, d, R, rng, Cmat):
    keys = ["dir_frac", "D_head", "imag_mass", "henrici"]
    acc = {kk: np.empty(R) for kk in keys}
    k = Sk.size
    for r in range(R):
        B = _rand_orthonormal(d, k, rng, Cmat).T @ _rand_orthonormal(d, k, rng, Cmat)
        s = kspace_spectrum(Sk, B)
        for kk in keys:
            acc[kk][r] = s[kk]
    return acc


def head_metrics(fac, l, h, R=300, rng=None, Cmat=None):
    if rng is None:
        rng = np.random.default_rng(C.SEED + 1000 * l + h)
    d = fac["meta"]["d_model"]
    if Cmat is None and fac["meta"]["norm"] == "LN":
        Cmat = centering_matrix(d)
    M = M_eff(fac, l, h, Cmat)

    # SVD → low-rank core (real head uses B = Vₖᵀ Uₖ, same routine as the null)
    U, S, Vt = D.svd(M)
    k = int((S > RANK_TOL * S[0]).sum())
    Uk, Sk, Vtk = U[:, :k], S[:k], Vt[:k]
    real = kspace_spectrum(Sk, Vtk @ Uk)

    # content axis: signed spectrum of the symmetric part (Elhage-analog)
    M_S, _ = D.sym_antisym(M)
    es = np.linalg.eigvalsh(M_S)
    es = es[np.abs(es) > RANK_TOL * np.abs(es).max()]
    content_pos_frac = float(es[es > 0].sum() / (np.abs(es).sum() + 1e-300))

    # query/key subspace geometry (spec §7.2)
    Wq_eff = (Cmat @ fac["Wq"][l, h]) if Cmat is not None else fac["Wq"][l, h]
    Wk_eff = (Cmat @ fac["Wk"][l, h]) if Cmat is not None else fac["Wk"][l, h]
    pa_mean, pa_max = D.principal_angles_deg(Wq_eff, Wk_eff)

    # null baseline + z-scores
    nd = null_distribution(Sk, d, R, rng, Cmat)
    row = dict(layer=l, head=h, rank=k, **real,
               content_pos_frac=content_pos_frac,
               prin_angle_mean=pa_mean, prin_angle_max=pa_max)
    for m in ["dir_frac", "D_head", "imag_mass", "henrici"]:
        mu, sd = float(nd[m].mean()), float(nd[m].std())
        row[f"{m}_null_mean"] = mu
        row[f"{m}_null_std"] = sd
        row[f"{m}_z"] = float((real[m] - mu) / (sd + 1e-300))
    return row


def build_table(model_key="gpt2", R=300, verbose=True) -> pd.DataFrame:
    fac = load_factors(model_key)
    m = fac["meta"]
    Cmat = centering_matrix(m["d_model"]) if m["norm"] == "LN" else None
    rows = []
    for l in range(m["n_layers"]):
        for h in range(m["n_heads"]):
            rng = np.random.default_rng(C.SEED + 1000 * l + h)
            rows.append(head_metrics(fac, l, h, R=R, rng=rng, Cmat=Cmat))
        if verbose:
            print(f"  layer {l:2d} done", flush=True)
    df = pd.DataFrame(rows)
    out = C.CACHE / f"{model_key}_head_metrics.parquet"
    df.to_parquet(out)
    if verbose:
        print(f"[P1] metrics table → {out}  ({len(df)} heads)")
    return df
