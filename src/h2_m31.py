"""M3.1-pilot (P-M31) — STEER SELECTION at 160M by PURE ASSISTANCE (no ban).

The impactful G1' claim's missing piece: does weak assistance SELECT which functionally-
equivalent implementation the induction circuit's prev head adopts, at real-corpus scale,
at capability parity — not just accelerate (P-M22-redo) but STEER which spectral algorithm
forms? Toy M1.3 showed 5/5 cos-only vs phase selection; M2.2 showed the λ10 BAN reaches
cos-only (rif 0.005) but at +0.04 nats and by prohibition, not assistance. P-M31 tests
selection by ASSISTANCE below the ban level.

Arm `select` (2 seeds; free + constraint(λ10 ban) baselines reused from M2.2):
  functional cos-only prev plant (P-M22-redo machinery: content-independent rotary-BIAS
  φ=θ → working Δ=−1 prev head at init, even/cos-only kernel character; 6 displacement-free
  naturally-inert slots; UNFROZEN) + a WEAK imag-share reg λ=3 (all heads; 3× below the M2.2
  ban; bounded share ∈[0,1], regularization-toward-real = pure assistance under G1' §0, which
  allows "regularization toward a target algebra, no channel bans"). The plant keeps the
  circuit forming (avoids the ban's +33% delay); the weak reg selects the cos-only weight
  implementation (P-M22-redo showed the plant alone reverts to default rif≈0.51).

FUNCTIONAL GATE (M2.2 lesson): seeded_prev ≥ 0.30 at init (verified), else abort in seconds.

Self-contained run loop: a faithful copy of h2_m22_train.cmd_run (identical data offsets,
eval cadence, lr schedule, ckpt) so the select arm is point-comparable to M2.2's free/
constraint — with two explicit additions (plant+gate; λ·penalty). Additive: never edits
h2_m22_train.py. Frozen predictions P-M31-a/b/c in results/h2/LOG.md before any run.

Usage:
  python src/h2_m31.py gate                    # functional gate on both seeds (CPU-ok)
  python src/h2_m31.py run --seed 0 --lam 3.0  # one 4B select run
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2_m22_train as T
import h2_m22_redo as RD

LAM_DEFAULT = 3.0
SEEDED = RD.SEEDED_OFF          # 6 displacement-free naturally-inert slots
BIAS_B = RD.BIAS_B             # 8.0


def cmd_gate(args):
    T.DEV = "cpu"                          # gate is a tiny forward pass; no GPU claim needed
    for seed in (0, 1):
        m = T.make_model(seed)
        RD.apply_plant_functional(m, SEEDED, BIAS_B, weight_plant=not args.bias_only)
        mean, mx, vals = RD.seeded_prev(m, SEEDED)
        p0 = float(T.penalty_imag(m).item())
        print(f"[gate] seed{seed} (bias_only={args.bias_only}): seeded_prev mean={mean:.3f} "
              f"max={mx:.3f} {'PASS' if mean >= RD.GATE_THRESH else 'FAIL'} | "
              f"penalty@init={p0:.4f}", flush=True)
        del m
        torch.cuda.empty_cache()


def cmd_run(args):
    T.SEEDED = SEEDED                      # so battery()/do_eval tag these heads seeded
    seed, lam = args.seed, args.lam
    bo = args.bias_only
    if args.suffix:                          # λ-curve etc.: explicit tag suffix
        tag = f"m22_select_seed{seed}{args.suffix}"
    else:
        tag = f"m22_select_seed{seed}_m31b" if bo else f"m22_select_seed{seed}_m31"
    evals_f = T.OUT / f"{tag}_evals.jsonl"
    ckdir = T.DATA / tag
    ckdir.mkdir(parents=True, exist_ok=True)
    total_steps = int(args.tokens // T.BATCH_TOKENS)
    micro = args.micro_bs
    accum = T.BATCH_TOKENS // (micro * T.CTX)
    assert T.BATCH_TOKENS % (micro * T.CTX) == 0

    mm = np.memmap(T.DATA / "tokens.bin", dtype=np.uint16, mode="r")
    val_ids = torch.from_numpy(np.array(np.memmap(
        T.DATA / "val.bin", dtype=np.uint16, mode="r")[:32 * T.CTX],
        dtype=np.int64).reshape(32, T.CTX))

    # --- model + FUNCTIONAL PLANT + gate (unfrozen) ---
    model = T.make_model(seed)
    plant_meta = RD.apply_plant_functional(model, SEEDED, BIAS_B, weight_plant=not bo)
    mean, mx, _ = RD.seeded_prev(model, SEEDED)
    assert mean >= RD.GATE_THRESH, (
        f"FUNCTIONAL GATE FAILED: seeded_prev mean={mean:.3f} < {RD.GATE_THRESH} "
        f"— plant not functional, aborting before the 4B launch")
    print(f"[{tag}] functional gate PASS: seeded_prev mean={mean:.3f} max={mx:.3f} "
          f"B={BIAS_B} lam={lam} bias_only={bo} (weak reg toward cos-only, all heads)",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=T.LR, betas=(0.9, 0.95),
                            weight_decay=0.01, fused=True)
    start_step = 0
    ck = ckdir / "ckpt_latest.pt"
    if ck.exists():
        st = torch.load(ck, map_location=T.DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_step = st["step"]
        print(f"[{tag}] resumed at step {start_step}", flush=True)

    def do_eval(step):
        model.eval()
        prev, ind, icl = T.behavioral(model)
        vl = T.val_loss(model, val_ids)
        wm = T.battery(model)
        row = dict(step=step, tokens=step * T.BATCH_TOKENS,
                   prev_beh=float(prev.max()), ind_beh=float(ind.max()),
                   icl=float(icl), val_loss=float(vl),
                   prev_all=prev.round(5).tolist(), ind_all=ind.round(5).tolist(),
                   penalty=float(T.penalty_imag(model).item()),   # TRUE imag share (unscaled)
                   pop_med_rif=float(np.median([w["rif"] for w in wm])),
                   pop_med_D=float(np.median([w["D_head"] for w in wm])),
                   pop_med_lam_mass=float(np.median([w.get("lam_mass", np.nan) for w in wm])),
                   pop_med_im_mass=float(np.median([w.get("im_mass", np.nan) for w in wm])),
                   seeded_prev_max=float(max(prev[l][h] for l, h in SEEDED)),
                   seeded_rif=[round(w["rif"], 4) for w in wm if w["seeded"]],
                   wm=wm)
        with open(evals_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        model.train()
        print(f"[{tag}] step {step} ({step*T.BATCH_TOKENS/1e9:.2f}B) val={vl:.3f} "
              f"prev={row['prev_beh']:.3f} ind={row['ind_beh']:.3f} icl={icl:+.2f} "
              f"pop_rif={row['pop_med_rif']:.3f}", flush=True)
        return row

    next_eval = start_step
    t0 = time.time(); tok0 = start_step * T.BATCH_TOKENS
    model.train()
    for step in range(start_step, total_steps + 1):
        if step >= next_eval:
            do_eval(step)
            tk = step * T.BATCH_TOKENS
            grid = T.EVAL_MT_FINE if 0.05e9 <= tk < 3e9 else T.EVAL_MT
            next_eval = step + max(1, int(grid * 1e6 // T.BATCH_TOKENS))
        if step == total_steps:
            break
        for g in opt.param_groups:
            g["lr"] = T.lr_at(step, total_steps)
        base = step * (T.BATCH_TOKENS // T.CTX)
        opt.zero_grad(set_to_none=True)
        for mi in range(accum):
            lo = (base + mi * micro) * T.CTX
            x = torch.from_numpy(np.array(mm[lo:lo + micro * T.CTX], dtype=np.int64)
                                 ).view(micro, T.CTX).to(T.DEV, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, labels=x).loss / accum
            loss.backward()
        (lam * T.penalty_imag(model)).backward()          # weak assistance toward cos-only
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            rate = ((step + 1) * T.BATCH_TOKENS - tok0) / (time.time() - t0 + 1e-9)
            eta = (total_steps - step) * T.BATCH_TOKENS / rate / 3600
            print(f"[{tag}] step {step}/{total_steps} loss={float(loss)*accum:.3f} "
                  f"{rate/1e3:.0f}k tok/s eta {eta:.1f}h", flush=True)
        if (step + 1) % (T.CKPT_EVERY // T.BATCH_TOKENS) == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            step=step + 1), ckdir / "ckpt_latest.pt")
    torch.save(dict(model=model.state_dict(), step=total_steps), ckdir / "final.pt")
    import pandas as pd
    rows = [json.loads(ln) for ln in open(evals_f)]
    pd.DataFrame([{k: v for k, v in r.items() if k != "wm"} for r in rows]
                 ).to_parquet(T.OUT / f"{tag}.parquet")
    (T.OUT / f"{tag}_meta.json").write_text(json.dumps(dict(
        arm="select", seed=seed, tokens=args.tokens, lam=lam, bias_B=BIAS_B,
        bias_only=bo, plant=plant_meta,
        seeded=[f"L{l}H{h}" for l, h in SEEDED]), indent=1, default=str))
    print(f"[{tag}] COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "run"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=LAM_DEFAULT)
    ap.add_argument("--tokens", type=float, default=4e9)
    ap.add_argument("--micro-bs", type=int, default=16, dest="micro_bs")
    ap.add_argument("--bias-only", action="store_true", dest="bias_only",
                    help="drop the phi=theta weight plant (P-M31 diagnosis) — cleaner selection")
    ap.add_argument("--suffix", default="",
                    help="explicit run-tag suffix (e.g. _m31bL10 for the λ-curve)")
    a = ap.parse_args()
    a.tokens = int(a.tokens)
    torch.set_num_threads(4)
    {"gate": cmd_gate, "run": cmd_run}[a.cmd](a)
