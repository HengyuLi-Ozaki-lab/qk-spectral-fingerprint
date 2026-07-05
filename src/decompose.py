"""P1 — the four decompositions of M_eff (spec §4). float64 throughout.

  (1) SVD                 M = U Σ Vᵀ                 (orientation-agnostic strength)
  (2) symmetric/skew      M = M_S + M_A              (content vs directional)
  (3) real Schur          M = Q T Qᵀ                 (2×2 blocks → rotation angle θ)
  (4) complex eig         eig(M), cond(eigvecs)      (non-Hermitian operator view)

Views (2)/(3) agree in the limits but weight differently; lead with (3). All angles
in degrees folded to [0°,180°]. The Schur/complex spectrum differs from the M_A
spectrum exactly by M's non-normality (RESEARCH_PLAN §1).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import schur, subspace_angles

TOL = 1e-9  # relative cutoff for "nonzero" eigenvalue (M_eff is rank ≤ d_head)


def svd(M):
    U, S, Vt = np.linalg.svd(M)
    return U, S, Vt


def sym_antisym(M):
    return 0.5 * (M + M.T), 0.5 * (M - M.T)


def complex_eig(M):
    """Eigenvalues (complex) and the eigenvector condition number (non-normality flag)."""
    w, V = np.linalg.eig(M)
    try:
        cond = float(np.linalg.cond(V))
    except np.linalg.LinAlgError:
        cond = np.inf
    return w, cond


def schur_blocks(M, rel_tol=1e-6):
    """Real Schur → per-block (|λ|, θ_deg) for blocks with |λ| above rel_tol·max|λ|.

    1×1 real block → θ = 0° (λ>0) or 180° (λ<0); 2×2 block λ=μ±iν → θ = atan2(ν,μ)."""
    T, _ = schur(M, output="real")
    n = T.shape[0]
    lam_abs, theta = [], []
    i = 0
    while i < n:
        is_block = (i + 1 < n) and abs(T[i + 1, i]) > 1e-14
        if is_block:
            a, b = T[i, i], T[i, i + 1]
            c, d = T[i + 1, i], T[i + 1, i + 1]
            mu = 0.5 * (a + d)
            disc = (0.5 * (a - d)) ** 2 + b * c
            nu = np.sqrt(max(-disc, 0.0))
            mag = float(np.hypot(mu, nu))
            th = float(np.degrees(np.arctan2(nu, mu)))          # ν≥0 ⇒ θ∈[0,180]
            lam_abs.append(mag); theta.append(th)
            i += 2
        else:
            lam = float(T[i, i])
            lam_abs.append(abs(lam)); theta.append(0.0 if lam >= 0 else 180.0)
            i += 1
    lam_abs = np.asarray(lam_abs); theta = np.asarray(theta)
    if lam_abs.size:
        keep = lam_abs > rel_tol * lam_abs.max()
        lam_abs, theta = lam_abs[keep], theta[keep]
    return lam_abs, theta


# ------------------------------------------------------------------ non-normality

def henrici_departure(M, eigvals=None):
    """Henrici's relative departure from normality: √(‖M‖_F² − Σ|λ|²) / ‖M‖_F ∈ [0,1)."""
    if eigvals is None:
        eigvals = np.linalg.eigvals(M)
    fro2 = float(np.sum(M * M))                    # ‖M‖_F² (real M)
    lam2 = float(np.sum(np.abs(eigvals) ** 2))
    return float(np.sqrt(max(fro2 - lam2, 0.0)) / (np.sqrt(fro2) + 1e-300))


# ------------------------------------------------------------------ subspace geometry

def principal_angles_deg(A, B):
    """Principal angles (degrees) between col(A) and col(B); returns (mean, max)."""
    ang = np.degrees(subspace_angles(A, B))
    return float(ang.mean()), float(ang.max())
