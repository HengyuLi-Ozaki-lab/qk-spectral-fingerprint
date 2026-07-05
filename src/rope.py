"""§6.4 / §8 — RoPE-frequency mechanism behind F6.

TL uses rotate-half pairing (verified): pair t couples dims (t, t+rd/2), t=0..rd/2-1,
θ_t = base^{-2t/rotary_dim}. Per-frequency complex QK sub-operator M_t = w_q^t·conj(w_k^t)ᵀ,
w_q^t = W_Q[:,t] + i·W_Q[:,t+rd/2]. The static M = Σ_t Re(M_t) + M_nonrot; the DIRECTIONAL
(relative-position-asymmetric) content is carried by Im(M_t):
    score(Δ) − score(−Δ) = Σ_t 2 sin(Δθ_t) · nᵀ Im(M_t) n.

Per head we measure how much rotary structure is directional (rope_imag_frac) and at what
frequency (freq_centroid), then test whether this RoPE deployment EXPLAINS F6 — i.e. whether
D_head's advantage for prev-token heads is mediated by the RoPE phase content.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model, centering_matrix

DEV = "cuda"


def rope_params(cfg):
    rd = cfg.rotary_dim; npair = rd // 2
    base = getattr(cfg, "rotary_base", 10000) or 10000
    theta = base ** (-2.0 * np.arange(npair) / rd)      # t=0 → highest freq
    return rd, npair, theta


@torch.no_grad()
def validate(model, Cmat, npair, theta, l=0, h=0):
    """Reconstruct the model's own RoPE'd scores (half pairing + θ + biases) → must match."""
    toks = model.to_tokens(["The quick brown fox jumps over the lazy dog today."])
    _, cache = model.run_with_cache(
        toks, names_filter=lambda n: "ln1.hook_normalized" in n or "hook_attn_scores" in n)
    n = cache[f"blocks.{l}.ln1.hook_normalized"][0].double().cpu().numpy()
    fold = lambda W: (Cmat @ W) if Cmat is not None else W
    WQ = fold(model.W_Q[l, h].detach().double().cpu().numpy())
    WK = fold(model.W_K[l, h].detach().double().cpu().numpy())
    bQ = model.b_Q[l, h].detach().double().cpu().numpy(); bK = model.b_K[l, h].detach().double().cpu().numpy()
    scale = float(model.blocks[0].attn.attn_scale)
    P = n.shape[0]; pos = np.arange(P)
    q = n @ WQ + bQ; k = n @ WK + bK

    def rot(x):
        out = x.copy()
        for t in range(npair):
            cA, sA = np.cos(pos * theta[t]), np.sin(pos * theta[t])
            xp, xq = x[:, t].copy(), x[:, t + npair].copy()
            out[:, t] = xp * cA - xq * sA; out[:, t + npair] = xp * sA + xq * cA
        return out

    score = (rot(q) @ rot(k).T) / scale
    ref = cache[f"blocks.{l}.attn.hook_attn_scores"][0, h].double().cpu().numpy()
    tril = np.tril(np.ones((P, P), bool))
    return float(np.abs(score - ref)[tril].max())


def head_rope_metrics(WQ, WK, npair, theta):
    a, c = WQ[:, :npair], WQ[:, npair:2 * npair]        # [d, npair]  query pair (real, imag)
    b, d = WK[:, :npair], WK[:, npair:2 * npair]        # key pair
    na2, nb2, nc2, nd2 = (a * a).sum(0), (b * b).sum(0), (c * c).sum(0), (d * d).sum(0)
    ac, bd = (a * c).sum(0), (b * d).sum(0)
    rre = np.sqrt(np.maximum(na2 * nb2 + nc2 * nd2 + 2 * ac * bd, 0))   # ‖Re(M_t)‖_F
    rim = np.sqrt(np.maximum(nc2 * nb2 + na2 * nd2 - 2 * ac * bd, 0))   # ‖Im(M_t)‖_F (directional)
    Rre, Rim = rre.sum(), rim.sum()
    return dict(rope_imag_frac=float(Rim / (Rre + Rim + 1e-30)),
                freq_centroid=float((theta * rim).sum() / (Rim + 1e-30)),
                hi_freq_dir=float(rim[:max(1, npair // 2)].sum() / (Rim + 1e-30)))


def partial(y, x, ctrl):
    def resid(a, b):
        A = np.column_stack([np.ones(len(b)), b]); be, *_ = np.linalg.lstsq(A, a, rcond=None); return a - A @ be
    return spearmanr(resid(x, ctrl), resid(y, ctrl))


def analyze(model_key):
    model = load_model(model_key, DEV)
    cfg = model.cfg
    rd, npair, theta = rope_params(cfg)
    Cmat = centering_matrix(cfg.d_model) if C.MODELS[model_key]["norm"] == "LN" else None
    fold = lambda W: (Cmat @ W) if Cmat is not None else W
    verr = validate(model, Cmat, npair, theta)
    tol = 1e-3 if model.W_Q.dtype.itemsize >= 4 else 1e-1     # relax for bf16 models
    print(f"[§6.4 {model_key}] rotary_dim={rd} npair={npair} base={getattr(cfg,'rotary_base',None)} | "
          f"score reconstruction max|Δ|={verr:.2e} ({'PASS' if verr<tol else 'CHECK'})")

    WQ = model.W_Q.detach().double().cpu().numpy(); WK = model.W_K.detach().double().cpu().numpy()
    rows = []
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            m = head_rope_metrics(fold(WQ[l, h]), fold(WK[l, h]), npair, theta)
            m.update(layer=l, head=h); rows.append(m)
    rp = pd.DataFrame(rows)
    df = pd.read_parquet(C.CACHE / f"{model_key}_head_full.parquet").merge(rp, on=["layer", "head"])
    df.to_parquet(C.CACHE / f"{model_key}_head_full.parquet")

    print(f"[§6.4 {model_key}] correlations (n={len(df)}):")
    for a, b in [("prev", "rope_imag_frac"), ("prev", "freq_centroid"), ("prev", "hi_freq_dir"),
                 ("D_head", "rope_imag_frac"), ("D_head", "freq_centroid"),
                 ("dir_frac", "rope_imag_frac")]:
        rho, p = spearmanr(df[a], df[b]); print(f"   corr({a:9s}, {b:15s}) = {rho:+.3f} (p={p:.1e})")
    # mechanism test: does the RoPE phase content explain D_head's prev-token advantage?
    pr_base, pb = partial(df.prev.values, df.D_head.values, df.dir_frac.values)
    pr_med, pm = partial(df.prev.values, df.D_head.values,
                         np.column_stack([df.dir_frac.values, df.rope_imag_frac.values, df.freq_centroid.values]))
    print(f"[§6.4 {model_key}] MECHANISM — partial(prev, D_head | dir_frac)            = {pr_base:+.3f} (p={pb:.1e})")
    print(f"                           partial(prev, D_head | dir_frac + RoPE phase)  = {pr_med:+.3f} (p={pm:.1e})"
          f"   → {'RoPE MEDIATES D_head' if abs(pr_med) < 0.6 * abs(pr_base) else 'partial'}")
    return df, model, npair, theta


@torch.no_grad()
def kernel_figure(model_key, df, model, top=4):
    """Empirical relative-position score kernel S(Δ) for top prev-token heads."""
    g = torch.Generator(device="cpu").manual_seed(0)
    toks = model.to_tokens(["In 1969, astronauts walked on the Moon while millions watched the broadcast live."] * 8)
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_attn_scores"))
    P = toks.shape[1]; K = min(24, P - 1)
    heads = df.sort_values("prev", ascending=False).head(top)[["layer", "head"]].values
    plt.figure(figsize=(6, 4))
    for l, h in heads:
        S = cache[f"blocks.{int(l)}.attn.hook_attn_scores"][:, int(h)].float().cpu().numpy()  # [b,q,k]
        ker = np.array([np.nanmean([S[:, i, i + dlt] for i in range(-dlt, P) if 0 <= i + dlt < P and i < P]) for dlt in range(-K, 1)])
        plt.plot(range(-K, 1), ker, marker=".", label=f"{int(l)}.{int(h)}")
    plt.axvline(-1, ls="--", c="grey", lw=1); plt.xlabel("relative offset Δ = key − query"); plt.ylabel("mean pre-softmax score")
    plt.title(f"{model_key}: prev-token heads peak at Δ=−1 (RoPE)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(C.FIGS / f"{model_key}_relpos_kernel.png", dpi=120); plt.close()
    print(f"[§6.4] kernel figure → {C.FIGS / (model_key + '_relpos_kernel.png')}")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["pythia-410m", "pythia-1.4b"])
    args = ap.parse_args()
    for mk in args.models:
        df, model, npair, theta = analyze(mk)
        kernel_figure(mk, df, model)
        print()


if __name__ == "__main__":
    main()
