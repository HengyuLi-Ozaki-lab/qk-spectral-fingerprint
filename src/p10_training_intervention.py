"""Experiment B — constrained-QK training intervention (INTERVENTION_PLAN §2, flagship).

Grid: {APE, RoPE} × {free, sym-M, Im(M_t)-suppressed (RoPE only)} × seeds.
Task: per-sequence random bigram map (in-context Markov / statistical-induction task):
  x_{t+1} = f_seq(x_t) w.p. 1−ε, uniform noise w.p. ε; f_seq resampled per sequence.
  Predicting map-steps requires INDUCTION (find previous occurrence of x_t, copy successor);
  no global memorization (map is per-sequence), no fixed-offset shortcut (offsets vary).

Constraints (soft penalties, relative, denominator detached):
  sym-M : λ · mean_h ‖M_A‖²_F / ‖M‖²_F.detach()           (kills antisymmetric part of static M)
  imag  : λ · mean_h Σ_t‖Im(M_t)‖²_F / Σ_t(‖Re‖²+‖Im‖²).detach()   (kills RoPE phase channel)
  KEY ALGEBRA: sym-M does NOT zero Im(M_t) (static M = Σ Re(M_t)); the two arms dissociate them.

Outcomes per eval: predictable-token CE (induction capability), best prev-token attention score,
best induction attention score, penalized ratios, per-head dir_frac/D_head/rope_imag_frac.
Pre-registered P1–P4 in INTERVENTION_PLAN §2 (locked 2026-07-03).
"""
from __future__ import annotations
import argparse, json, math
import numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG

DEV = "cuda"
V = 64            # vocab
SEQ = 256         # tokens per sequence (targets: SEQ-1)
EPS = 0.1         # noise prob in the map process


# ----------------------------------------------------------------------------- data

def gen_batch(bs, rng: torch.Generator):
    """Per-sequence random map sequences. Returns tokens [bs,SEQ], map_mask [bs,SEQ-1]
    (True where x_{t+1} was produced by the map — the predictable-if-induction targets)."""
    f = torch.randint(0, V, (bs, V), generator=rng)                 # per-seq map
    x = torch.empty(bs, SEQ, dtype=torch.long)
    x[:, 0] = torch.randint(0, V, (bs,), generator=rng)
    noise = torch.rand(bs, SEQ - 1, generator=rng) < EPS
    rnd = torch.randint(0, V, (bs, SEQ - 1), generator=rng)
    for t in range(SEQ - 1):
        nxt = f[torch.arange(bs), x[:, t]]
        x[:, t + 1] = torch.where(noise[:, t], rnd[:, t], nxt)
    return x, ~noise


def eval_masks(x):
    """seen_mask[b,t] = current token x_t already appeared at some j<t (lookup possible)."""
    bs, L = x.shape
    seen = torch.zeros(bs, L, dtype=torch.bool)
    for t in range(1, L):
        seen[:, t] = (x[:, :t] == x[:, t:t + 1]).any(1)
    return seen


# ----------------------------------------------------------------------------- model & penalties

def make_model(scheme: str, seed: int):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=128, n_ctx=SEQ, d_head=32, n_heads=4,
        d_vocab=V, attn_only=True, normalization_type="LN",
        positional_embedding_type=("rotary" if scheme == "rope" else "standard"),
        rotary_dim=32, rotary_base=10000, seed=seed, act_fn=None,
    )
    return HookedTransformer(cfg).to(DEV)


def penalty_sym(model):
    tot = 0.0
    for l in range(model.cfg.n_layers):
        WQ, WK = model.W_Q[l], model.W_K[l]                        # [H,d,dh]
        M = torch.einsum("hde,hDe->hdD", WQ, WK)                   # [H,d,d]
        MA = 0.5 * (M - M.transpose(1, 2))
        num = (MA ** 2).sum((1, 2))
        den = (M ** 2).sum((1, 2)).detach() + 1e-8
        tot = tot + (num / den).mean()
    return tot / model.cfg.n_layers


def penalty_imag(model):
    """Σ_t ‖Im(M_t)‖² / Σ_t(‖Re‖²+‖Im‖²), closed form (verified in rope.py audit)."""
    npair = model.cfg.rotary_dim // 2
    tot = 0.0
    for l in range(model.cfg.n_layers):
        WQ, WK = model.W_Q[l], model.W_K[l]
        a, c = WQ[:, :, :npair], WQ[:, :, npair:2 * npair]         # [H,d,npair]
        b, d = WK[:, :, :npair], WK[:, :, npair:2 * npair]
        na2, nb2 = (a * a).sum(1), (b * b).sum(1)                  # [H,npair]
        nc2, nd2 = (c * c).sum(1), (d * d).sum(1)
        ac, bd = (a * c).sum(1), (b * d).sum(1)
        im2 = nc2 * nb2 + na2 * nd2 - 2 * ac * bd
        re2 = na2 * nb2 + nc2 * nd2 + 2 * ac * bd
        num = im2.sum(1); den = (im2 + re2).sum(1).detach() + 1e-8
        tot = tot + (num / den).mean()
    return tot / model.cfg.n_layers


# ----------------------------------------------------------------------------- evals

@torch.no_grad()
def evaluate(model, rng_eval):
    x, mmask = gen_batch(64, rng_eval)
    seen = eval_masks(x)
    xg = x.to(DEV)
    logits, cache = model.run_with_cache(
        xg, names_filter=lambda n: n.endswith("hook_pattern") or n.endswith("hook_attn_scores"))
    lp = torch.log_softmax(logits.float(), -1)
    ce = -lp[:, :-1].gather(-1, xg[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()   # [bs,SEQ-1]
    pred_mask = mmask & seen[:, :-1]                # map-step AND lookup available
    ce_pred = float(ce[pred_mask].mean())
    ce_unpred = float(ce[~mmask].mean())            # noise steps: irreducible ~ln V
    # attention scores: prev = mass on t-1; induction = mass on {j: x_{j-1}=x_t}
    bs, L = x.shape
    qpos = torch.arange(64, L)                      # skip warmup positions
    prev_best, ind_best = 0.0, 0.0
    prev_scores, ind_scores = [], []
    # induction target mask [bs,Q,K]
    tgt = (x[:, None, :-1] == x[:, qpos, None]).float()            # x_{j-1} == x_t → mass at j
    tgt = torch.cat([torch.zeros(bs, len(qpos), 1), tgt], dim=2)   # shift: j = (j-1)+1
    causal = torch.arange(L)[None, :] < qpos[:, None]              # j < t  [Q,K]
    tgt = tgt * causal[None].float()
    kernels = []
    offs = list(range(-8, 1))
    for l in range(model.cfg.n_layers):
        A = cache[f"blocks.{l}.attn.hook_pattern"][:, :, qpos].float().cpu()  # [bs,H,Q,K]
        pv = A[:, :, torch.arange(len(qpos)), qpos - 1].mean((0, 2))          # [H]
        iv = (A * tgt[:, None]).sum(-1).mean((0, 2))                          # [H]
        prev_scores += pv.tolist(); ind_scores += iv.tolist()
        S = cache[f"blocks.{l}.attn.hook_attn_scores"][:, :, qpos].float().cpu()
        for h in range(model.cfg.n_heads):
            kernels.append([float(S[:, h, torch.arange(len(qpos)), qpos + o].mean()) for o in offs])
    return dict(ce_pred=ce_pred, ce_unpred=ce_unpred,
                prev_best=float(max(prev_scores)), ind_best=float(max(ind_scores)),
                prev_all=prev_scores, ind_all=ind_scores, kernels=kernels)


def weight_metrics(model):
    """dir_frac / D_head / rope_imag_frac of every head (float64, reuse audited paths)."""
    from metrics import kspace_spectrum
    from p4_rope import thin_svd
    from rope import head_rope_metrics
    out = []
    npair = (model.cfg.rotary_dim // 2) if model.cfg.positional_embedding_type == "rotary" else 1
    theta = 10000.0 ** (-2.0 * np.arange(npair) / (model.cfg.rotary_dim if npair > 1 else 2))
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            WQ = model.W_Q[l, h].detach().double().cpu().numpy()
            WK = model.W_K[l, h].detach().double().cpu().numpy()
            Uk, Sk, Vk = thin_svd(WQ, WK)
            s = kspace_spectrum(Sk, Vk.T @ Uk)
            rm = head_rope_metrics(WQ, WK, npair, theta) if npair > 1 else dict(rope_imag_frac=np.nan)
            out.append(dict(layer=l, head=h, dir_frac=s["dir_frac"], D_head=s["D_head"],
                            rope_imag_frac=rm["rope_imag_frac"]))
    return out


# ----------------------------------------------------------------------------- training

def run(scheme, constraint, lam, seed, steps=6000, bs=64, lr=1e-3, eval_every=100, outdir=None,
        wd=0.01, init_fn=None, extra_penalties=None, save_weights=False, tag_suffix=""):
    """H1 extension (2026-07-03, only-add): new kwargs all have behavior-preserving
    defaults — wd=0.01 (was hardcoded), init_fn=None (post-init weight surgery hook),
    extra_penalties=None (additive constraint-registry entries), save_weights=False
    (end-of-run state_dict), tag_suffix="" (disambiguates e.g. wd/bs variants).
    trainB semantics are unchanged at the defaults."""
    model = make_model(scheme, seed)
    if init_fn is not None:
        init_fn(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200))
    rng = torch.Generator().manual_seed(10_000 + seed)
    rng_eval = torch.Generator().manual_seed(99)                    # fixed eval stream
    registry = dict(free=None, sym=penalty_sym, imag=penalty_imag)
    if extra_penalties:
        registry.update(extra_penalties)
    pen_fn = registry[constraint]
    hist = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            model.eval()
            ev = evaluate(model, torch.Generator().manual_seed(99))
            pen_now = float(pen_fn(model).item()) if pen_fn else float("nan")
            wm = weight_metrics(model)
            hist.append(dict(step=step,
                             **{k: v for k, v in ev.items() if k not in ("prev_all", "ind_all", "kernels")},
                             penalty=pen_now,
                             prev_all=json.dumps(ev["prev_all"]), ind_all=json.dumps(ev["ind_all"]),
                             kernels=json.dumps(ev["kernels"]), wm=json.dumps(wm)))
            model.train()
        if step == steps:
            break
        x, _ = gen_batch(bs, rng)
        x = x.to(DEV)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        if pen_fn is not None:
            loss = loss + lam * pen_fn(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    import pandas as pd
    df = pd.DataFrame(hist)
    tag = f"{scheme}_{constraint}_lam{lam}_seed{seed}{tag_suffix}"
    if outdir:
        df.to_parquet(outdir / f"{tag}.parquet")
        if save_weights:
            torch.save(model.state_dict(), outdir / f"{tag}.pt")
    return df, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="smoke")
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    outdir = CFG.CACHE / "trainB"; outdir.mkdir(exist_ok=True)
    if args.grid == "smoke":
        combos = [("rope", "free", 0.0, 0)]
    elif args.grid == "calib":
        combos = [("rope", "imag", 1.0, 0), ("rope", "imag", 10.0, 0),
                  ("rope", "sym", 1.0, 0), ("ape", "sym", 1.0, 0)]
    else:  # full
        combos = []
        for seed in (0, 1, 2):
            combos += [("ape", "free", 0.0, seed), ("ape", "sym", 10.0, seed),
                       ("rope", "free", 0.0, seed), ("rope", "sym", 10.0, seed),
                       ("rope", "imag", 1.0, seed), ("rope", "imag", 10.0, seed)]
    for scheme, cons, lam, seed in combos:
        df, tag = run(scheme, cons, lam, seed, steps=args.steps, outdir=outdir)
        last = df.iloc[-1]
        # formation step: first eval with ce_pred below halfway between ln(V) and floor(0.15)
        thresh = 0.5 * (math.log(V) + 0.15)
        formed = df[df.ce_pred < thresh]
        fstep = int(formed.step.iloc[0]) if len(formed) else -1
        print(f"[{tag}] final ce_pred={last.ce_pred:.3f} (unpred {last.ce_unpred:.2f}) "
              f"prev_best={last.prev_best:.2f} ind_best={last.ind_best:.2f} "
              f"penalty={last.penalty:.4f} formation_step={fstep}", flush=True)


if __name__ == "__main__":
    main()
