"""P0 — Plumbing & conventions.

Load a model via TransformerLens with LayerNorm folded into the weights, extract
the per-head QK factors, and build the LN-folded operator M_eff whose spectrum P1
analyzes. Includes the two make-or-break P0 tests (spec §1, §11):

  * logit-match : reconstruct the model's own pre-softmax attention scores from
                  M = W_Q W_K^T and match to < 1e-4  (convention is correct).
  * gauge check : W_Q→W_Q G, W_K→W_K G^{-T} leaves M (and its spectrum) invariant,
                  while raw column norms change  (only gauge-invariants are physical).

Convention (verified on GPT-2 small, TL 3.5.0):
  TL stores W_Q,W_K as [n_layers, n_heads, d_model, d_head]; q = n @ W_Q + b_Q with
  n = LN-normalized residual (hook `blocks.{l}.ln1.hook_normalized`, already centered
  for LayerNorm). Hence
        score(i,j) = (n_i·M·n_j + bias terms) / attn_scale,   M = W_Q W_K^T ∈ R^{d×d}.
  The bilinear operator is M (rank ≤ d_head). LN gain γ is already folded into W_Q/W_K
  by fold_ln; LN centering C = I − 11^T/d is folded symmetrically into M_eff = C M C
  (a congruence — it changes the spectrum, which is correct; spec §3).
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# ----------------------------------------------------------------------------- model / factors

def load_model(name: str = "gpt2", device: str = "cuda"):
    import torch
    from transformer_lens import HookedTransformer
    spec = C.MODELS[name]
    center = (spec["norm"] == "LN")   # centering tricks are output-preserving only for LayerNorm
    kw = dict(fold_ln=True, center_writing_weights=center, center_unembed=center, device=device)
    if spec.get("dtype"):
        kw["dtype"] = getattr(torch, spec["dtype"])
    model = HookedTransformer.from_pretrained(spec["tl_name"], **kw)
    model.eval()
    return model


def get_factors(model, model_key: str) -> dict:
    """Extract per-head QK factors as float64 numpy arrays + metadata.
    Handles GQA (n_kv_heads < n_heads) by expanding K/V heads to query heads, and
    upcasts bf16 → float before numpy (numpy has no bf16)."""
    WK, bK = model.W_K, model.b_K
    nH = model.W_Q.shape[1]
    if WK.shape[1] != nH:                              # GQA: query head h reads KV head h // rep
        rep = nH // WK.shape[1]
        WK = WK.repeat_interleave(rep, dim=1)
        bK = bK.repeat_interleave(rep, dim=1)
    to = lambda t: t.detach().float().cpu().numpy().astype(C.SPECTRAL_DTYPE)
    fac = dict(
        Wq=to(model.W_Q), Wk=to(WK),                  # [L, H, d_model, d_head]
        bQ=to(model.b_Q), bK=to(bK),                  # [L, H, d_head]
        meta=dict(
            model=model_key,
            norm=C.MODELS[model_key]["norm"],
            d_model=int(model.cfg.d_model), n_heads=int(model.cfg.n_heads),
            n_layers=int(model.cfg.n_layers), d_head=int(model.cfg.d_head),
            n_kv_heads=int(model.W_K.shape[1]),
            attn_scale=float(model.blocks[0].attn.attn_scale),
        ),
    )
    return fac


def centering_matrix(d: int) -> np.ndarray:
    return np.eye(d, dtype=C.SPECTRAL_DTYPE) - np.full((d, d), 1.0 / d, dtype=C.SPECTRAL_DTYPE)


def M_qk(fac: dict, l: int, h: int) -> np.ndarray:
    """Raw folded bilinear operator M = W_Q W_K^T (acts on centered normalized input)."""
    return fac["Wq"][l, h] @ fac["Wk"][l, h].T


def M_eff(fac: dict, l: int, h: int, C_mat: np.ndarray | None = None) -> np.ndarray:
    """LN-folded operator: M_eff = C M C for LayerNorm; = M for RMSNorm (no centering)."""
    M = M_qk(fac, l, h)
    if fac["meta"]["norm"] == "LN":
        if C_mat is None:
            C_mat = centering_matrix(fac["meta"]["d_model"])
        M = C_mat @ M @ C_mat
    return M

# ----------------------------------------------------------------------------- P0 test 1: logit match

DEFAULT_TEXTS = [
    "The cat sat on the mat and looked outside.",
    "In 1969, astronauts first walked on the surface of the Moon.",
    "def add(a, b):\n    return a + b",
    "Paris is the capital of France, and Tokyo is the capital of Japan.",
    "She sold seashells by the seashore every single summer.",
    "The mitochondria is the powerhouse of the cell, biologists say.",
    "To be, or not to be, that is the question.",
    "Attention weights are computed from queries and keys.",
]


def logit_match(model, fac: dict, texts=DEFAULT_TEXTS) -> float:
    """Reconstruct pre-softmax scores from the cached factors; return max |Δ| on the
    causal (lower-triangular) region across all layers & heads. Asserts < C.RECON_RTOL."""
    toks = model.to_tokens(texts)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)
    L, scale = fac["meta"]["n_layers"], fac["meta"]["attn_scale"]
    P = toks.shape[1]
    tril = np.tril(np.ones((P, P), dtype=bool))
    worst = 0.0
    for l in range(L):
        n = cache[f"blocks.{l}.ln1.hook_normalized"].cpu().numpy().astype(np.float64)  # [b,p,d]
        q = np.einsum("bpd,hde->bphe", n, fac["Wq"][l]) + fac["bQ"][l]                  # [b,p,h,e]
        k = np.einsum("bpd,hde->bphe", n, fac["Wk"][l]) + fac["bK"][l]
        s = np.einsum("bqhe,bkhe->bhqk", q, k) / scale                                  # [b,h,q,k]
        ref = cache[f"blocks.{l}.attn.hook_attn_scores"].cpu().numpy().astype(np.float64)
        worst = max(worst, float(np.abs(s - ref)[:, :, tril].max()))
    assert worst < C.RECON_RTOL, f"logit-match FAILED: max|Δ|={worst:.2e} ≥ {C.RECON_RTOL}"
    return worst

# ----------------------------------------------------------------------------- P0 test 2: gauge invariance

def gauge_check(fac: dict, l: int = 0, h: int = 0, seed: int = C.SEED) -> dict:
    """Apply a random G ∈ GL(d_head): W_Q→W_Q G, W_K→W_K G^{-T}. M and its spectrum
    must be invariant; raw column norms must change."""
    rng = np.random.default_rng(seed)
    dh = fac["meta"]["d_head"]
    G = rng.standard_normal((dh, dh))
    while abs(np.linalg.det(G)) < 1e-3:                    # keep it well-conditioned
        G = rng.standard_normal((dh, dh))
    Wq, Wk = fac["Wq"][l, h], fac["Wk"][l, h]
    Wq2, Wk2 = Wq @ G, Wk @ np.linalg.inv(G).T
    M1, M2 = Wq @ Wk.T, Wq2 @ Wk2.T
    e1 = np.sort_complex(np.linalg.eigvals(M1))
    e2 = np.sort_complex(np.linalg.eigvals(M2))
    denom = np.linalg.norm(M1) + 1e-30
    return dict(
        m_reldiff=float(np.linalg.norm(M2 - M1) / denom),
        eig_absdiff=float(np.abs(e2 - e1).max()),
        colnorm_reldiff=float(np.linalg.norm(np.linalg.norm(Wq2, axis=0) - np.linalg.norm(Wq, axis=0))
                             / (np.linalg.norm(np.linalg.norm(Wq, axis=0)) + 1e-30)),
    )

# ----------------------------------------------------------------------------- cache

def cache_factors(fac: dict, model_key: str) -> Path:
    path = C.CACHE / f"{model_key}_factors.npz"
    np.savez_compressed(path, Wq=fac["Wq"], Wk=fac["Wk"], bQ=fac["bQ"], bK=fac["bK"],
                        meta=json.dumps(fac["meta"]))
    (C.CACHE / f"{model_key}_meta.json").write_text(json.dumps(fac["meta"], indent=2))
    return path


def load_factors(model_key: str) -> dict:
    d = np.load(C.CACHE / f"{model_key}_factors.npz", allow_pickle=True)
    return dict(Wq=d["Wq"], Wk=d["Wk"], bQ=d["bQ"], bK=d["bK"], meta=json.loads(str(d["meta"])))

# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=list(C.MODELS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[P0] loading {args.model} on {args.device} (GPU pinned via CUDA_VISIBLE_DEVICES) ...")
    model = load_model(args.model, args.device)
    fac = get_factors(model, args.model)
    m = fac["meta"]
    print(f"[P0] cfg: {m}")

    print("[P0] test 1/3 — logit match (convention) ...")
    worst = logit_match(model, fac)
    print(f"       PASS  max|Δ| on causal region = {worst:.2e}  (< {C.RECON_RTOL})")

    print("[P0] test 2/3 — gauge invariance (random G ∈ GL(d_head)) ...")
    g = gauge_check(fac)
    assert g["m_reldiff"] < C.GAUGE_ATOL and g["eig_absdiff"] < 1e-6, f"gauge FAILED: {g}"
    assert g["colnorm_reldiff"] > 1e-3, f"column norms unchanged?! {g}"
    print(f"       PASS  M rel-Δ={g['m_reldiff']:.1e}, eig Δ={g['eig_absdiff']:.1e}, "
          f"colnorm rel-Δ={g['colnorm_reldiff']:.2f} (changed, as required)")

    print("[P0] test 3/3 — M_eff on-data equivalence (C M C ≡ M on centered input) ...")
    # sanity: centering acts as identity on the (already-centered) normalized input
    toks = model.to_tokens(DEFAULT_TEXTS[:2])
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)
    n = cache["blocks.0.ln1.hook_normalized"][0].cpu().numpy().astype(np.float64)  # [p,d]
    s_qk  = n @ M_qk(fac, 0, 0) @ n.T
    s_eff = n @ M_eff(fac, 0, 0) @ n.T
    ondata = float(np.abs(s_qk - s_eff).max())
    assert ondata < 1e-8, f"M_eff changed on-data scores?! {ondata:.2e}"
    print(f"       PASS  on-data |Δ(M_eff, M_qk)| = {ondata:.1e}  (M_eff only cleans the 1-direction)")

    path = cache_factors(fac, args.model)
    print(f"[P0] cached factors → {path}  ({path.stat().st_size/1e6:.1f} MB)")
    print("[P0] DONE — conventions nailed, ready for P1.")


if __name__ == "__main__":
    main()
