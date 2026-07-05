"""H1 Workstream A — M1.2 assistance hinge + M1.3 selection pilot (entry script).

Design: H1_EXECUTION_BRIEF §3–§4; frozen predictions P-A1/P-A2/P-A3 in ROADMAP §3.
Corrections/operationalizations C1, C2, S1–S3 recorded in results/h1/LOG.md (Step 0)
BEFORE any run — this file implements them:
  C1  APE solution plant uses the corrected orientation M = Σ_k p_k p_{k−1}ᵀ.
  C2  assist-reg penalties are bounded complements with LIVE denominators.
  S2  RoPE kernel gates evaluated over Δ∈[−8,−1] (self-offset reported, not gated).
  S3  init ce_pred gate = [ln V − 0.05, 5.0].

Init arms seed heads (L0,H0) and (L0,H1); heads are exchangeable at init. All
constructions are norm-matched per matrix (‖W_Q‖_F, ‖W_K‖_F preserved exactly).

Subcommands:
  gate  — build every init arm × scheme × seed, run the step-0 eval, print + save the
          verification-gate table (results/h1/m12_gates.parquet). No training.
  time  — 400-step timing probe (throwaway) to calibrate worker count.
  run   — execute a shard of the worklist: --phase m12|m13 --shard i --nshards K.
          Idempotent (skips existing parquets). Writes {tag}.parquet, {tag}.pt,
          {tag}_meta.json under results/h1/runs/.

Launch pattern (GPU0, brief §0.3):
  CUDA_VISIBLE_DEVICES=0 nohup python src/h1_hinge.py run --phase m12 --shard $i --nshards 3 &
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, json, math, time
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG
import p10_training_intervention as p10

OUT = CFG.RESULTS / "h1" / "runs"
OUT.mkdir(parents=True, exist_ok=True)

NPAIR = 16                                   # rotary_dim 32 → 16 freq pairs (half-split)
THETA = 10000.0 ** (-2.0 * np.arange(NPAIR) / 32)
SEED_HEADS = (0, 1)                          # L0 H0, H1 (prev-candidate layer, 2 of 4)
LAYER = 0
LNV = math.log(64)
FORM_THRESH = 0.5 * (LNV + 0.15)             # verbatim p10 formation definition


def formation_step(df):
    formed = df[df.ce_pred < FORM_THRESH]
    return int(formed.step.iloc[0]) if len(formed) else -1


# ------------------------------------------------------------------ constructions
# All in-place on the REAL parameters model.blocks[0].attn.W_Q/W_K (L0, seeded heads),
# norm-matched per matrix. NB: model.W_Q/W_K at HookedTransformer level are PROPERTIES
# returning stacked copies — writing to them is a silent no-op (caught at gate time).

def _params(model):
    att = model.blocks[LAYER].attn
    return att.W_Q, att.W_K                     # Parameters [n_heads, d_model, d_head]


def _norm_match(W_new, target_norm):
    return W_new * (target_norm / (W_new.norm() + 1e-12))


def _rope_pair_rotation(model, phases_per_head):
    """W_K[L0,h] pair t := R(φ_t) applied to W_Q[L0,h] pair t (TL rotate-half convention:
    b = a·cosφ − c·sinφ, d = a·sinφ + c·cosφ). Effective coherent kernel ∝ cos(θΔ + φ)."""
    meta = {}
    PQ, PK = _params(model)
    with torch.no_grad():
        for h, ph in phases_per_head.items():
            WQ, WK = PQ[h], PK[h]
            kn = WK.norm()
            a, c = WQ[:, :NPAIR], WQ[:, NPAIR:]
            t = torch.as_tensor(np.asarray(ph), dtype=WQ.dtype, device=WQ.device)
            b = a * torch.cos(t) - c * torch.sin(t)
            d = a * torch.sin(t) + c * torch.cos(t)
            WK.copy_(_norm_match(torch.cat([b, d], dim=1), kn))
            meta[f"h{h}"] = dict(phases=[float(x) for x in np.asarray(ph)])
    return meta


def _plant_factored(model, h, M_target):
    """Replace (W_Q, W_K)[L0,h] by rank-d_head truncated-SVD factors of M_target (d×d),
    each factor rescaled to the original ‖·‖_F. W_Q W_Kᵀ ∝ best rank-32 approx of M."""
    PQ, PK = _params(model)
    with torch.no_grad():
        WQ, WK = PQ[h], PK[h]
        dh = WQ.shape[1]
        U, s, Vt = np.linalg.svd(M_target)
        Uq = U[:, :dh] * np.sqrt(s[:dh])
        Vk = Vt[:dh].T * np.sqrt(s[:dh])
        WQ.copy_(_norm_match(torch.as_tensor(Uq, dtype=WQ.dtype, device=WQ.device), WQ.norm()))
        WK.copy_(_norm_match(torch.as_tensor(Vk, dtype=WK.dtype, device=WK.device), WK.norm()))


def _skew_plant(model, seed):
    """APE algebra assist / RoPE cross-placebo: M := random skew (G−Gᵀ)/2, rank-32
    truncated (singular pairs → truncation stays skew; dir_frac = 1)."""
    d = model.cfg.d_model
    for h in SEED_HEADS:
        rng = np.random.default_rng(555_000_000 + seed * 1000 + h)
        G = rng.standard_normal((d, d))
        _plant_factored(model, h, (G - G.T) / 2)
    return dict(kind="skew", heads=list(SEED_HEADS))


def _sym_plant(model, seed):
    """M1.3 APE mirror: M := random symmetric (G+Gᵀ)/2, rank-32 truncated."""
    d = model.cfg.d_model
    for h in SEED_HEADS:
        rng = np.random.default_rng(666_000_000 + seed * 1000 + h)
        G = rng.standard_normal((d, d))
        _plant_factored(model, h, (G + G.T) / 2)
    return dict(kind="sym", heads=list(SEED_HEADS))


def _ape_solution_plant(model):
    """APE positive control (LOG C1 corrected orientation): M := Σ_k p_k p_{k−1}ᵀ on the
    model's own initial position embeddings → coherent Δ=−1 kernel; rank-32 truncated."""
    Wp = model.W_pos.detach().double().cpu().numpy()          # [n_ctx, d]
    M = Wp[1:].T @ Wp[:-1]                                    # query side p_k, key side p_{k−1}
    for h in SEED_HEADS:
        _plant_factored(model, h, M)
    return dict(kind="pos_prev", heads=list(SEED_HEADS))


def _wk_ortho(model, seed):
    """placebo_random: W_K := O·W_Q, O random orthogonal (same pipeline, no target algebra)."""
    d = model.cfg.d_model
    meta = {}
    PQ, PK = _params(model)
    with torch.no_grad():
        for h in SEED_HEADS:
            rng = np.random.default_rng(888_000_000 + seed * 1000 + h)
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            WQ, WK = PQ[h], PK[h]
            O = torch.as_tensor(Q, dtype=WQ.dtype, device=WQ.device)
            WK.copy_(_norm_match(O @ WQ, WK.norm()))
            meta[f"h{h}"] = dict(kind="ortho")
    return meta


def _algebra_phases(seed, h):
    rng = np.random.default_rng(777_000_000 + seed * 1000 + h)
    return rng.uniform(0.0, 2.0 * np.pi, NPAIR)


def make_init(arm, scheme, seed, sol_sign=+1):
    """Returns (init_fn, meta_holder) — init_fn=None for arms without weight surgery."""
    holder = {}
    if arm in ("free", "assist_reg"):
        return None, holder

    def fn(model):
        if arm == "assist_init_algebra":
            if scheme == "rope":
                holder.update(_rope_pair_rotation(
                    model, {h: _algebra_phases(seed, h) for h in SEED_HEADS}))
            else:
                holder.update(_skew_plant(model, seed))
        elif arm == "assist_init_solution":
            if scheme == "rope":
                holder.update(_rope_pair_rotation(
                    model, {h: sol_sign * THETA for h in SEED_HEADS}))
                holder["sol_sign"] = sol_sign
            else:
                holder.update(_ape_solution_plant(model))
        elif arm == "placebo_cross":
            if scheme == "rope":
                holder.update(_skew_plant(model, seed))       # APE-style algebra under RoPE
            else:
                holder.update(_rope_pair_rotation(            # RoPE-style algebra under APE
                    model, {h: _algebra_phases(seed, h) for h in SEED_HEADS}))
        elif arm == "placebo_random":
            holder.update(_wk_ortho(model, seed))
        elif arm == "assist_nondefault":                      # M1.3
            if scheme == "rope":
                holder.update(_rope_pair_rotation(            # symmetric-real: φ_t = 0
                    model, {h: np.zeros(NPAIR) for h in SEED_HEADS}))
            else:
                holder.update(_sym_plant(model, seed))
        else:
            raise ValueError(arm)
        holder["arm"], holder["scheme"], holder["seed"] = arm, scheme, seed
    return fn, holder


# ------------------------------------------------------------------ penalties (LOG C2)
# Bounded complements with LIVE denominators: value ∈ [0,1] per unit λ by construction,
# scale-invariant in W (a detached denominator would be an unbounded norm reward).

def _imag_shares(model):
    out = []
    for l in range(model.cfg.n_layers):
        WQ, WK = model.W_Q[l], model.W_K[l]
        a, c = WQ[:, :, :NPAIR], WQ[:, :, NPAIR:]
        b, d = WK[:, :, :NPAIR], WK[:, :, NPAIR:]
        na2, nb2 = (a * a).sum(1), (b * b).sum(1)
        nc2, nd2 = (c * c).sum(1), (d * d).sum(1)
        ac, bd = (a * c).sum(1), (b * d).sum(1)
        im2 = nc2 * nb2 + na2 * nd2 - 2 * ac * bd
        re2 = na2 * nb2 + nc2 * nd2 + 2 * ac * bd
        out.append(im2.sum(1) / (im2.sum(1) + re2.sum(1) + 1e-8))     # [H] live share
    return out


def _antisym_shares(model):
    out = []
    for l in range(model.cfg.n_layers):
        WQ, WK = model.W_Q[l], model.W_K[l]
        M = torch.einsum("hde,hDe->hdD", WQ, WK)
        MA = 0.5 * (M - M.transpose(1, 2))
        out.append((MA ** 2).sum((1, 2)) / ((M ** 2).sum((1, 2)) + 1e-8))
    return out


def penalty_assist_imag(model):
    """RoPE assist-reg: mean_h (1 − imag_share), live ratio."""
    sh = _imag_shares(model)
    return sum((1.0 - s).mean() for s in sh) / len(sh)


def penalty_assist_sym(model):
    """APE assist-reg: mean_h (1 − ‖M_A‖²/‖M‖²), live ratio."""
    sh = _antisym_shares(model)
    return sum((1.0 - s).mean() for s in sh) / len(sh)


def penalty_toward_real(model):
    """M1.3 RoPE non-default: complement toward REAL share = mean_h imag_share, live."""
    sh = _imag_shares(model)
    return sum(s.mean() for s in sh) / len(sh)


def penalty_toward_symM(model):
    """M1.3 APE non-default: complement toward SYMMETRIC M = mean_h antisym share, live."""
    sh = _antisym_shares(model)
    return sum(s.mean() for s in sh) / len(sh)


def extra_penalties(scheme):
    return {
        "assist_init_algebra": None, "assist_init_solution": None,
        "placebo_cross": None, "placebo_random": None,
        "assist_reg": penalty_assist_imag if scheme == "rope" else penalty_assist_sym,
        "assist_nondefault": penalty_toward_real if scheme == "rope" else penalty_toward_symM,
    }


# ------------------------------------------------------------------ worklists

def worklist(phase, schemes=("rope", "ape")):
    """Items: (scheme, constraint, lam, seed, extra) — extra overrides steps/wd/bs and
    sets the tag suffix (M1.5 variants share constraint='free' and need distinct names)."""
    combos = []
    if phase == "m12":
        for scheme in ("rope", "ape"):                        # free arms FIRST (regression gate)
            for seed in range(5):
                combos.append((scheme, "free", 0.0, seed, {}))
        for scheme in ("rope", "ape"):
            for seed in range(5):
                combos.append((scheme, "assist_init_algebra", 0.0, seed, {}))
                combos.append((scheme, "assist_init_solution", 0.0, seed, {}))
                combos.append((scheme, "assist_reg", 1.0, seed, {}))
                combos.append((scheme, "assist_reg", 10.0, seed, {}))
            for seed in range(3):
                combos.append((scheme, "placebo_cross", 0.0, seed, {}))
                combos.append((scheme, "placebo_random", 0.0, seed, {}))
    elif phase == "m13":
        for seed in range(5):
            combos.append(("rope", "assist_nondefault", 1.0, seed, {}))
    elif phase == "m13ape":
        for seed in range(5):
            combos.append(("ape", "assist_nondefault", 1.0, seed, {}))
    elif phase == "m15gate":                                  # suppression gate (M1.5 step 0)
        for scheme in schemes:
            for seed in range(3):
                combos.append((scheme, "free", 0.0, seed,
                               dict(steps=60000, suffix="_s60k")))
    elif phase == "m15grid":                                  # WD grid + bs knob (gate-passing schemes)
        for scheme in schemes:
            for wd in (0.0, 0.1):
                for seed in range(3):
                    combos.append((scheme, "free", 0.0, seed,
                                   dict(steps=60000, wd=wd, suffix=f"_s60k_wd{wd:g}")))
            for bs in (16, 256):
                for seed in range(2):
                    combos.append((scheme, "free", 0.0, seed,
                                   dict(steps=60000, bs=bs, suffix=f"_s60k_bs{bs}")))
    else:
        raise ValueError(phase)
    return combos


# ------------------------------------------------------------------ gate machinery

def _recover_phases(model, h):
    """Weight-space verification of the pair-rotation plant: least-squares recovery of
    the per-frequency phase φ̂_t from (W_Q, W_K) of L0 head h (b = a·cosφ − c·sinφ,
    d = a·sinφ + c·cosφ up to the norm rescale)."""
    att = model.blocks[LAYER].attn
    WQ = att.W_Q[h].detach().double().cpu().numpy()
    WK = att.W_K[h].detach().double().cpu().numpy()
    a, c = WQ[:, :NPAIR], WQ[:, NPAIR:]
    b, d = WK[:, :NPAIR], WK[:, NPAIR:]
    phis = []
    for t in range(NPAIR):
        A2 = np.stack([np.concatenate([a[:, t], c[:, t]]),
                       np.concatenate([-c[:, t], a[:, t]])], axis=1)
        y = np.concatenate([b[:, t], d[:, t]])
        cs, sn = np.linalg.lstsq(A2, y, rcond=None)[0]
        phis.append(math.atan2(sn, cs))
    return np.array(phis)


@torch.no_grad()
def _init_snapshot(model):
    """Step-0 eval + weight metrics (identical code paths as training evals)."""
    model.eval()
    ev = p10.evaluate(model, torch.Generator().manual_seed(99))
    wm = p10.weight_metrics(model)
    return ev, wm


def _gate_row(scheme, arm, seed, sol_sign=+1):
    ref = p10.make_model(scheme, seed)
    ref_norms = {h: (float(ref.W_Q[LAYER, h].norm()), float(ref.W_K[LAYER, h].norm()))
                 for h in SEED_HEADS}
    del ref
    torch.cuda.empty_cache()

    model = p10.make_model(scheme, seed)
    fn, meta = make_init(arm, scheme, seed, sol_sign=sol_sign)
    if fn is not None:
        fn(model)
    ev, wm = _init_snapshot(model)
    rows = []
    offs = np.arange(-8, 1)
    PQ, PK = _params(model)
    for h in SEED_HEADS:
        i = LAYER * 4 + h
        ker = np.array(ev["kernels"][i])
        neg = ker[:8]                                          # Δ ∈ [−8,−1]
        qn, kn = float(PQ[h].norm()), float(PK[h].norm())
        phase_err = float("nan")                               # weight-space plant check
        hmeta = meta.get(f"h{h}", {})
        if scheme == "rope" and isinstance(hmeta, dict) and "phases" in hmeta:
            tgt = np.asarray(hmeta["phases"])
            rec = _recover_phases(model, h)
            phase_err = float(np.abs(((rec - tgt + np.pi) % (2 * np.pi)) - np.pi).max())
        rows.append(dict(
            scheme=scheme, arm=arm, seed=seed, head=h, sol_sign=sol_sign,
            q_norm_ratio=qn / ref_norms[h][0], k_norm_ratio=kn / ref_norms[h][1],
            rif=wm[i]["rope_imag_frac"], dir_frac=wm[i]["dir_frac"], D_head=wm[i]["D_head"],
            argmax_neg=int(offs[int(np.argmax(neg))]),
            argmax_full=int(offs[int(np.argmax(ker))]),
            k_m1=float(ker[7]), k_0=float(ker[8]),
            k_neg_margin=float(ker[7] - np.max(np.delete(neg, 7))),
            phase_err=phase_err,
            ce_pred0=ev["ce_pred"],
        ))
    del model
    torch.cuda.empty_cache()
    return rows


def cmd_gate(args):
    import pandas as pd
    rows = []
    t0 = time.time()
    if args.phase == "m12":
        for scheme in ("rope", "ape"):
            rows += _gate_row(scheme, "free", 0)               # reference row
            for seed in range(5):
                rows += _gate_row(scheme, "assist_init_algebra", seed)
                if scheme == "rope":
                    for ss in (+1, -1):                        # empirical sign pick, recorded
                        rows += _gate_row(scheme, "assist_init_solution", seed, sol_sign=ss)
                else:
                    rows += _gate_row(scheme, "assist_init_solution", seed)
            for seed in range(3):
                rows += _gate_row(scheme, "placebo_cross", seed)
                rows += _gate_row(scheme, "placebo_random", seed)
    elif args.phase in ("m13", "m13ape"):
        scheme = "rope" if args.phase == "m13" else "ape"
        rows += _gate_row(scheme, "free", 0)
        for seed in range(5):
            rows += _gate_row(scheme, "assist_nondefault", seed)
    df = pd.DataFrame(rows)
    out = CFG.RESULTS / "h1" / (f"gates_{args.phase}.parquet")
    df.to_parquet(out)
    # penalty bound check on free inits
    for scheme in ("rope", "ape"):
        m = p10.make_model(scheme, 0)
        pa = float(extra_penalties(scheme)["assist_reg"](m))
        pn = float(extra_penalties(scheme)["assist_nondefault"](m))
        print(f"[pen] {scheme}: assist_reg(free init)={pa:.4f}  nondefault(free init)={pn:.4f} "
              f"(both must be in [0,1]; ≈0.5 expected)")
        del m
        torch.cuda.empty_cache()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[gate] {len(df)} rows → {out}  ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ run driver

def cmd_run(args):
    combos = worklist(args.phase, schemes=tuple(args.schemes.split(",")))
    shard = combos[args.shard::args.nshards]
    print(f"[shard {args.shard}/{args.nshards}] {len(shard)} runs", flush=True)
    for scheme, cons, lam, seed, extra in shard:
        steps = extra.get("steps", args.steps)
        wd = extra.get("wd", 0.01)
        bs = extra.get("bs", 64)
        suffix = extra.get("suffix", "")
        tag = f"{scheme}_{cons}_lam{lam}_seed{seed}{suffix}"
        if (OUT / f"{tag}.parquet").exists():
            print(f"[skip] {tag}", flush=True)
            continue
        fn, meta = make_init(cons, scheme, seed, sol_sign=args.sol_sign)
        t0 = time.time()
        df, tag = p10.run(scheme, cons, lam, seed, steps=steps, bs=bs, outdir=OUT,
                          wd=wd, init_fn=fn, extra_penalties=extra_penalties(scheme),
                          save_weights=True, tag_suffix=suffix)
        wall = time.time() - t0
        fstep = formation_step(df)
        last = df.iloc[-1]
        meta.update(tag=tag, formation_step=fstep, wall_s=round(wall, 1),
                    final_ce_pred=float(last.ce_pred), final_prev=float(last.prev_best),
                    final_ind=float(last.ind_best), steps=steps, wd=wd, bs=bs,
                    sol_sign=args.sol_sign if cons == "assist_init_solution" else None)
        (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print(f"[done] {tag}: formation={fstep} ce={last.ce_pred:.3f} "
              f"prev={last.prev_best:.2f} ind={last.ind_best:.2f} ({wall/60:.1f} min)", flush=True)


def cmd_time(args):
    t0 = time.time()
    p10.run("rope", "free", 0.0, 99, steps=400, outdir=None)
    dt = time.time() - t0
    est = dt / 400 * 6000 / 60
    print(f"[time] 400 steps in {dt:.1f}s → est {est:.1f} min per 6000-step run")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "run", "time"])
    ap.add_argument("--phase", default="m12")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--schemes", default="rope,ape")
    ap.add_argument("--sol-sign", type=int, default=+1, dest="sol_sign")
    args = ap.parse_args()
    torch.set_num_threads(4)
    {"gate": cmd_gate, "run": cmd_run, "time": cmd_time}[args.cmd](args)
