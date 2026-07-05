"""RoPE-aware causal ablation — the causal counterpart to §6.4 (F6 mechanism).

Per head, ZERO the directional RoPE content Im(M_t) (keep Re(M_t)). This symmetrizes the
relative-position kernel: score_t(Δ)=[nᵀRe(M_t)n]cos(Δθ)+[nᵀIm(M_t)n]sin(Δθ) → drop the
sin (odd-in-Δ) part. Then measure the causal effect on an induction task (needs prev-token→
induction routing) and on general CE, and correlate with prev / rope_imag_frac / D_head.

If killing Im(M_t) hurts prev-token/high-D_head heads causally, the RoPE rotational phase
(which D_head reads) does real functional work — turning F6's mediation into causation.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model
from observables import get_tokens, ce_loss
from p2b_targeted import induction_ce
from rope import rope_params

DEV = "cuda"


def _delta_im(n, WQ, WK, npair, theta):
    """Σ_t nᵀIm(M_t)n · sin((j−i)θ_t)   →  [b,P,P]  (the part removed by the ablation)."""
    b_, P, _ = n.shape
    pos = torch.arange(P, device=n.device, dtype=n.dtype)
    delta = torch.zeros(b_, P, P, device=n.device, dtype=n.dtype)
    for t in range(npair):
        a, c = WQ[:, t], WQ[:, t + npair]
        bb, d = WK[:, t], WK[:, t + npair]
        qa, qc, kb, kd = n @ a, n @ c, n @ bb, n @ d           # [b,P]
        cP, sP = torch.cos(pos * theta[t]), torch.sin(pos * theta[t])
        # nᵀIm(M_t)n = qc·kb − qa·kd ; times sin((j−i)θ) = sin(jθ)cos(iθ)−cos(jθ)sin(iθ)
        t1 = torch.einsum("bi,bj->bij", qc * cP, kb * sP) - torch.einsum("bi,bj->bij", qc * sP, kb * cP)
        t2 = torch.einsum("bi,bj->bij", qa * cP, kd * sP) - torch.einsum("bi,bj->bij", qa * sP, kd * cP)
        delta += t1 - t2
    return delta


def _hooks(model, l, h, npair, theta):
    WQ = model.W_Q[l, h].float(); WK = model.W_K[l, h].float()
    th = torch.tensor(theta, device=DEV, dtype=torch.float32)
    scale = float(model.blocks[0].attn.attn_scale)
    store = {}
    def grab(act, hook): store["n"] = act; return act
    def modify(scores, hook):
        scores[:, h] = scores[:, h] - _delta_im(store["n"], WQ, WK, npair, th) / scale
        return scores
    return [(f"blocks.{l}.ln1.hook_normalized", grab), (f"blocks.{l}.attn.hook_attn_scores", modify)]


@torch.no_grad()
def validate_delta(model, npair, theta, l=0, h=0):
    """full weight-score = symmetric(cos) part + delta(sin) part → validates the decomposition."""
    toks = model.to_tokens(["The quick brown fox jumps over the lazy dog today."])
    _, cache = model.run_with_cache(toks, names_filter=lambda n: "ln1.hook_normalized" in n)
    n = cache[f"blocks.{l}.ln1.hook_normalized"].float()
    WQ = model.W_Q[l, h].float(); WK = model.W_K[l, h].float()
    rd = 2 * npair; scale = 1.0
    q0, k0 = n @ WQ, n @ WK                                    # [1,P,dh] weight-only
    P = n.shape[1]; pos = torch.arange(P, device=DEV, dtype=torch.float32)

    def rot(x):
        out = x.clone()
        for t in range(npair):
            cP, sP = torch.cos(pos * theta[t]), torch.sin(pos * theta[t])
            xp, xq = x[..., t].clone(), x[..., t + npair].clone()
            out[..., t] = xp * cP - xq * sP; out[..., t + npair] = xp * sP + xq * cP
        return out

    full = torch.einsum("bpd,bqd->bpq", rot(q0), rot(k0))      # weight-only RoPE'd score
    # symmetric (cos-only) part
    th = torch.tensor(theta, device=DEV, dtype=torch.float32)
    sym = torch.einsum("bpd,bqd->bpq", q0[..., rd:], k0[..., rd:])  # non-rotary content
    for t in range(npair):
        a, c = WQ[:, t], WQ[:, t + npair]; bb, d = WK[:, t], WK[:, t + npair]
        qa, qc, kb, kd = n @ a, n @ c, n @ bb, n @ d
        cP, sP = torch.cos(pos * theta[t]), torch.sin(pos * theta[t])
        Re = torch.einsum("bi,bj->bij", qa, kb) + torch.einsum("bi,bj->bij", qc, kd)  # nᵀRe(M_t)n
        cos_ij = torch.einsum("bi,bj->bij", qa * 0 + cP, qa * 0 + cP) + torch.einsum("bi,bj->bij", qa * 0 + sP, qa * 0 + sP)
        sym = sym + Re * cos_ij
    delta = _delta_im(n, WQ, WK, npair, th)
    err = float((full - (sym + delta)).abs().max())
    return err


@torch.no_grad()
def kernel_before_after(model, npair, theta, l, h, model_key):
    toks = model.to_tokens(["In 1969, astronauts walked on the Moon while millions watched the broadcast live."] * 8)
    outs = {}
    for tag, hooks in [("intact", []), ("Im(M_t) killed", _hooks(model, l, h, npair, theta))]:
        with model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_attn_scores"))
        S = cache[f"blocks.{l}.attn.hook_attn_scores"][:, h].float().cpu().numpy()
        P = S.shape[1]; K = min(20, P - 1)
        outs[tag] = np.array([np.nanmean([S[:, i, i + dl] for i in range(P) if 0 <= i + dl < P]) for dl in range(-K, 1)])
    plt.figure(figsize=(6, 4))
    for tag, ker in outs.items():
        plt.plot(range(-len(ker) + 1, 1), ker, marker=".", label=tag)
    plt.axvline(-1, ls="--", c="grey", lw=1); plt.xlabel("relative offset Δ"); plt.ylabel("mean pre-softmax score")
    plt.title(f"{model_key} head {l}.{h}: killing Im(M_t) flattens the Δ=−1 peak"); plt.legend()
    plt.tight_layout(); plt.savefig(C.FIGS / f"{model_key}_ablate_kernel_{l}_{h}.png", dpi=120); plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--n_seq", type=int, default=400); ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--ind_seq", type=int, default=200); ap.add_argument("--ind_len", type=int, default=64)
    args = ap.parse_args()
    model = load_model(args.model, DEV)
    npair, theta = rope_params(model.cfg)[1], rope_params(model.cfg)[2]

    err = validate_delta(model, npair, theta)
    print(f"[RoPE-ablate {args.model}] delta decomposition check: max|full−(sym+delta)|={err:.2e} "
          f"({'PASS' if err < 1e-3 else 'CHECK'})")

    df = pd.read_parquet(C.CACHE / f"{args.model}_head_full.parquet").reset_index(drop=True)

    # induction-task corpus (repeated-random) + general corpus
    g = torch.Generator(device="cpu").manual_seed(0)
    r = torch.randint(0, model.cfg.d_vocab, (args.ind_seq, args.ind_len), generator=g)
    bos = model.tokenizer.bos_token_id
    bos = model.tokenizer.eos_token_id if bos is None else bos
    itoks = torch.cat([torch.full((args.ind_seq, 1), bos), r, r], dim=1).to(DEV)
    base_ind = induction_ce(model, itoks, args.ind_len)
    corpus = get_tokens(model, args.n_seq, args.seq_len)
    base_ce = ce_loss(model, corpus, batch_size=128)
    print(f"[RoPE-ablate] baseline induction CE={base_ind:.4f}  general CE={base_ce:.4f}")

    dind = np.zeros(len(df)); dce = np.zeros(len(df))
    for i, row in df.iterrows():
        l, h = int(row["layer"]), int(row["head"])
        hk = _hooks(model, l, h, npair, theta)
        dind[i] = induction_ce(model, itoks, args.ind_len, hk) - base_ind
        dce[i] = ce_loss(model, corpus, batch_size=128, fwd_hooks=hk) - base_ce
        if h == 0:
            print(f"  layer {l:2d} dInd(head0)={dind[i]:+.4f}", flush=True)
    df["dInd_rope"] = dind; df["dCE_rope"] = dce
    df.to_parquet(C.CACHE / f"{args.model}_head_full.parquet")

    print("\n[RoPE-ablate] causal effect of killing Im(M_t) vs head properties:")
    for t in ["prev", "rope_imag_frac", "D_head", "dir_frac", "ind", "freq_centroid"]:
        rho, p = spearmanr(df[t], df["dInd_rope"]); print(f"   corr(dInd_rope, {t:15s}) = {rho:+.3f} (p={p:.1e})")
    print(f"   dInd_rope: median {np.median(dind):+.4f}  max {dind.max():+.4f}  (baseline {base_ind:.3f})")
    print(f"   dCE_rope : median {np.median(dce):+.4f}  max {dce.max():+.4f}  (baseline {base_ce:.3f})")
    cols = ["layer", "head", "dInd_rope", "dCE_rope", "prev", "rope_imag_frac", "D_head"]
    print("\n[RoPE-ablate] top-8 heads whose Im(M_t) the induction task needs:")
    print(df.sort_values("dInd_rope", ascending=False)[cols].head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    top = df.sort_values("prev", ascending=False).iloc[0]
    kernel_before_after(model, npair, theta, int(top["layer"]), int(top["head"]), args.model)
    print(f"\n[RoPE-ablate] before/after kernel → results/figures/{args.model}_ablate_kernel_{int(top['layer'])}_{int(top['head'])}.png")


if __name__ == "__main__":
    main()
