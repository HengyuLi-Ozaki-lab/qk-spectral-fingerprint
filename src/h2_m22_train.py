"""M2.2 trainer — from-scratch Pythia-160M on the fixed pre-tokenized pile stream.

Design + pre-registration: results/h2/LOG.md (M2.2 entry, 2026-07-05). Arms:
  free       — Pythia-160M config, HF init (seeded).
  assist     — frozen pair-coherent solution plant (φ_t=+θ_t) on the k-rotary slice of
               heads H0,H1 in layers L1–L4 (8/144), norm-matched; q/k rotary slices of
               seeded heads FROZEN via restore-after-step (exact, incl. WD).
  constraint — P1-B imag-share penalty (detached denominator, trainB semantics) λ=10
               on all heads' rotary pairs.
Identical data order / schedule / hyperparams across arms; init seed is the only
per-seed difference. Probes: p9 behavioral semantics ported (repeated-random [BOS,r,r],
attention prev/ind per head, ICL = 2nd-copy − 1st-copy NLL), val loss, M-side battery
(audited thin_svd/kspace/rope paths, float64).

CLI:
  gate   — layout+rotary reconstruction, plant phase-recovery, penalty sanity,
           30-step throughput bench (GPU; run inside a ledger claim).
  run    — --arm {free,assist,constraint} --seed N [--tokens 4e9] [--suffix _smoke]
Outputs: results/h2/runs/m22/{tag}_evals.jsonl (+ .parquet at end);
checkpoints /large/share/li_qk/h2_m22/{tag}/.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, contextlib, json, math, time
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG
from p4_rope import thin_svd
from metrics import kspace_spectrum
from rope import head_rope_metrics

DATA = Path(os.environ.get("QK_LARGE", "/large")) / "share/li_qk/h2_m22"   # box B: QK_LARGE=$HOME/large (SYNC.md; default unchanged)
OUT = CFG.RESULTS / "h2" / "runs" / "m22"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"

D, L, H, DH = 768, 12, 12, 64
ROT = 16                                    # rotary_ndims = 64 * 0.25
NPAIR = ROT // 2
THETA = 10000.0 ** (-2.0 * np.arange(NPAIR) / ROT)
CTX = 2048
SEEDED = [(l, h) for l in (1, 2, 3, 4) for h in (0, 1)]
BATCH_TOKENS = 1_048_576                    # 512 seqs × 2048
LR, LR_MIN, WARMUP_FRAC = 6e-4, 6e-5, 0.01
EVAL_MT, EVAL_MT_FINE = 50, 25              # fine grid inside 0.5–3B window
CKPT_EVERY = 500_000_000
WEIGHT_PLANT_EXPECTED = True   # opt-out flag for bias-only plants (W2 2026-07-15); True = M2.2/redo-exact behavior


def make_model(seed):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(
        vocab_size=50304, hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
        intermediate_size=3072, rotary_pct=0.25, rotary_emb_base=10000,
        max_position_embeddings=CTX, hidden_dropout=0.0, attention_dropout=0.0,
        use_parallel_residual=True, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    model = GPTNeoXForCausalLM(cfg)
    return model.to(DEV)


@contextlib.contextmanager
def eager_attn(model):
    old = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        model.config._attn_implementation = old


# ------------------------------------------------------------------ weight addressing
# qkv weight [H*3*DH, D]; head h rows: q = [h*3DH, h*3DH+DH), k = +DH, v = +2DH.
# Rotary slice = first ROT rows of the q/k blocks; pairs (t, t+NPAIR) within it.

def _qkv(model, l):
    return model.gpt_neox.layers[l].attention.query_key_value.weight


def q_rows(h):
    return slice(h * 3 * DH, h * 3 * DH + ROT)


def k_rows(h):
    return slice(h * 3 * DH + DH, h * 3 * DH + DH + ROT)


def apply_plant(model):
    """assist arm: k-rotary rows := R(+θ_t)·(q-rotary rows), norm-matched per block."""
    meta = {}
    cos = torch.as_tensor(np.cos(THETA), dtype=torch.float32, device=DEV)[:, None]
    sin = torch.as_tensor(np.sin(THETA), dtype=torch.float32, device=DEV)[:, None]
    with torch.no_grad():
        for l, h in SEEDED:
            W = _qkv(model, l)
            q = W[q_rows(h)]                              # [16, 768]
            a, c = q[:NPAIR], q[NPAIR:]
            b = a * cos - c * sin
            d = a * sin + c * cos
            new = torch.cat([b, d], 0)
            kn = W[k_rows(h)].norm()
            W[k_rows(h)] = new * (kn / (new.norm() + 1e-12))
            meta[f"L{l}H{h}"] = float(kn)
    return meta


def frozen_slices(model):
    out = []
    for l, h in SEEDED:
        W = _qkv(model, l)
        for sl in (q_rows(h), k_rows(h)):
            out.append((W, sl, W[sl].detach().clone()))
    return out


def restore(slices):
    with torch.no_grad():
        for W, sl, saved in slices:
            W[sl].copy_(saved)


def recover_phases(model, l, h):
    W = _qkv(model, l).detach().double().cpu().numpy()
    q, k = W[q_rows(h)], W[k_rows(h)]
    a, c = q[:NPAIR], q[NPAIR:]
    b, d = k[:NPAIR], k[NPAIR:]
    phis = []
    for t in range(NPAIR):
        A2 = np.stack([np.concatenate([a[t], c[t]]),
                       np.concatenate([-c[t], a[t]])], 1)
        y = np.concatenate([b[t], d[t]])
        cs, sn = np.linalg.lstsq(A2, y, rcond=None)[0]
        phis.append(math.atan2(sn, cs))
    return np.array(phis)


def penalty_imag(model):
    """trainB-semantics imag-share (detached denominator), all heads' rotary pairs."""
    tot = 0.0
    for l in range(L):
        W = _qkv(model, l)
        Wv = W.view(H, 3 * DH, D)
        q = Wv[:, :ROT]                                   # [H, 16, 768]
        k = Wv[:, DH:DH + ROT]
        a, c = q[:, :NPAIR], q[:, NPAIR:]
        b, d = k[:, :NPAIR], k[:, NPAIR:]
        na2, nb2 = (a * a).sum(-1), (b * b).sum(-1)       # [H, 8]
        nc2, nd2 = (c * c).sum(-1), (d * d).sum(-1)
        ac, bd = (a * c).sum(-1), (b * d).sum(-1)
        im2 = nc2 * nb2 + na2 * nd2 - 2 * ac * bd
        re2 = na2 * nb2 + nc2 * nd2 + 2 * ac * bd
        num = im2.sum(-1)
        den = (im2 + re2).sum(-1).detach() + 1e-8
        tot = tot + (num / den).mean()
    return tot / L


# ------------------------------------------------------------------ probes

@torch.no_grad()
def behavioral(model, n_seq=150, seq_len=64, seed=0):
    """p9 semantics ported: [BOS, r, r]; per-head prev/ind attention scores; ICL."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    prev = np.zeros((L, H)); ind = np.zeros((L, H)); icl = 0.0; seen = 0
    p = torch.arange(seq_len + 1, 2 * seq_len + 1)
    tprev, tind = p - 1, p - seq_len + 1
    with eager_attn(model):
        for i in range(0, n_seq, 50):
            bs = min(50, n_seq - i)
            r = torch.randint(0, 50304, (bs, seq_len), generator=g)
            toks = torch.cat([torch.zeros(bs, 1, dtype=torch.long), r, r], 1).to(DEV)
            out = model(toks, output_attentions=True)
            for l in range(L):
                A = out.attentions[l].float()
                prev[l] += A[:, :, p, tprev].mean((0, 2)).cpu().numpy() * bs
                ind[l] += A[:, :, p, tind].mean((0, 2)).cpu().numpy() * bs
            lp = torch.log_softmax(out.logits.float(), -1)
            nll = -lp[:, :-1].gather(-1, toks[:, 1:].unsqueeze(-1)).squeeze(-1)
            icl += float(nll[:, seq_len + 1:2 * seq_len].mean()
                         - nll[:, 1:seq_len].mean()) * bs
            seen += bs
            del out, lp
    return prev / seen, ind / seen, icl / seen


@torch.no_grad()
def val_loss(model, val_ids):
    tot = 0.0
    for i in range(0, len(val_ids), 8):
        x = val_ids[i:i + 8].to(DEV)
        with torch.autocast("cuda", torch.bfloat16):
            out = model(x, labels=x)
        tot += float(out.loss) * len(x)
    return tot / len(val_ids)


def battery(model):
    rows = []
    for l in range(L):
        W = _qkv(model, l).detach().double().cpu().numpy()
        for h in range(H):
            Wq = W[h * 3 * DH: h * 3 * DH + DH].T          # [768, 64] toy convention
            Wk = W[h * 3 * DH + DH: h * 3 * DH + 2 * DH].T
            Uk, Sk, Vk = thin_svd(Wq, Wk)
            B = Vk.T @ Uk
            s = kspace_spectrum(Sk, B)
            # instrument amendment 2026-07-06 (theory routing, results/h2/LOG.md):
            # λ-mass = Σ|λ|/Σσ and Im-mass = Σ|Imλ|/Σσ — the theory's process
            # variables (D_head conflates λ-collapse with residual spectral shape).
            # Present from run 4 (free_seed1) onward; runs 1–3 have endpoints only.
            w = np.linalg.eigvals(Sk[:, None] * B)
            ssum = float(Sk.sum()) + 1e-300
            rm = head_rope_metrics(Wq[:, :ROT], Wk[:, :ROT], NPAIR, THETA)
            rows.append(dict(layer=l, head=h, dir_frac=s["dir_frac"],
                             D_head=s["D_head"], rif=rm["rope_imag_frac"],
                             lam_mass=float(np.abs(w).sum() / ssum),
                             im_mass=float(np.abs(w.imag).sum() / ssum),
                             seeded=(l, h) in SEEDED))
    return rows


# ------------------------------------------------------------------ training

def lr_at(step, total_steps):
    warm = max(1, int(total_steps * WARMUP_FRAC))
    if step < warm:
        return LR * (step + 1) / warm
    t = (step - warm) / max(1, total_steps - warm)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * t))


def cmd_run(args):
    tag = f"m22_{args.arm}_seed{args.seed}{args.suffix}"
    evals_f = OUT / f"{tag}_evals.jsonl"
    ckdir = DATA / tag
    ckdir.mkdir(parents=True, exist_ok=True)
    total_steps = int(args.tokens // BATCH_TOKENS)
    micro = args.micro_bs
    accum = BATCH_TOKENS // (micro * CTX)
    assert BATCH_TOKENS % (micro * CTX) == 0

    mm = np.memmap(DATA / "tokens.bin", dtype=np.uint16, mode="r")
    val_ids = torch.from_numpy(np.array(np.memmap(
        DATA / "val.bin", dtype=np.uint16, mode="r")[:32 * CTX],
        dtype=np.int64).reshape(32, CTX))

    model = make_model(args.seed)
    plant_meta = {}
    frozen = []
    if args.arm == "assist":
        plant_meta = apply_plant(model)
        if WEIGHT_PLANT_EXPECTED:
            errs = [np.abs(((recover_phases(model, l, h) - THETA + np.pi) % (2 * np.pi))
                           - np.pi).max() for l, h in SEEDED]
            assert max(errs) < 1e-4, f"plant verification failed: {max(errs)}"
            print(f"[{tag}] plant verified: max phase err {max(errs):.2e}", flush=True)
        else:
            # bias-only plants (W2/P-ARCH1, 2026-07-15) write no k=R(θ)q WEIGHT structure by
            # design — the phase assert above would fire on exactly the component they omit.
            # Their verification is the wrapper's behavioral functional gate (run inside
            # apply_plant). Opt-in via `T.WEIGHT_PLANT_EXPECTED = False`; the default (True)
            # preserves M2.2/redo behavior bit-for-bit.
            print(f"[{tag}] weight-phase assert skipped (WEIGHT_PLANT_EXPECTED=False; "
                  f"bias-only plant — functional gate is the verification)", flush=True)
        frozen = frozen_slices(model)
    pen = penalty_imag if args.arm == "constraint" else None
    lam = 10.0

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01, fused=True)
    start_step = 0
    ck = ckdir / "ckpt_latest.pt"
    if ck.exists():                                        # resume
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step = st["step"]
        if args.arm == "assist":
            frozen = frozen_slices(model)                  # re-snapshot (identical)
        print(f"[{tag}] resumed at step {start_step}", flush=True)

    def do_eval(step):
        model.eval()
        prev, ind, icl = behavioral(model)
        vl = val_loss(model, val_ids)
        wm = battery(model)
        row = dict(step=step, tokens=step * BATCH_TOKENS,
                   prev_beh=float(prev.max()), ind_beh=float(ind.max()),
                   icl=float(icl), val_loss=float(vl),
                   prev_all=prev.round(5).tolist(), ind_all=ind.round(5).tolist(),
                   penalty=float(penalty_imag(model).item()),
                   pop_med_rif=float(np.median([w["rif"] for w in wm])),
                   pop_med_D=float(np.median([w["D_head"] for w in wm])),
                   pop_med_lam_mass=float(np.median([w.get("lam_mass", np.nan)
                                                     for w in wm])),
                   pop_med_im_mass=float(np.median([w.get("im_mass", np.nan)
                                                    for w in wm])),
                   seeded_prev_max=float(max(prev[l][h] for l, h in SEEDED)),
                   seeded_rif=[round(w["rif"], 4) for w in wm if w["seeded"]],
                   wm=wm)
        with open(evals_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        model.train()
        print(f"[{tag}] step {step} ({step*BATCH_TOKENS/1e9:.2f}B) "
              f"val={vl:.3f} prev={row['prev_beh']:.3f} ind={row['ind_beh']:.3f} "
              f"icl={icl:+.2f}", flush=True)
        return row

    next_eval = start_step
    t0 = time.time()
    tok0 = start_step * BATCH_TOKENS
    model.train()
    for step in range(start_step, total_steps + 1):
        if step >= next_eval:
            do_eval(step)
            tk = step * BATCH_TOKENS
            # grid amendment 2026-07-05 (results/h2/LOG.md): the actual formation window
            # sits earlier than the Pythia-official anchor (batch 1M vs 2M tokens/step);
            # fine grid now covers 0.05–3B so T_prev and T_ind share 25M resolution.
            grid = EVAL_MT_FINE if 0.05e9 <= tk < 3e9 else EVAL_MT
            next_eval = step + max(1, int(grid * 1e6 // BATCH_TOKENS))
        if step == total_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps)
        base = step * (BATCH_TOKENS // CTX)
        opt.zero_grad(set_to_none=True)
        for m in range(accum):
            lo = (base + m * micro) * CTX
            x = torch.from_numpy(np.array(mm[lo:lo + micro * CTX], dtype=np.int64)
                                 ).view(micro, CTX).to(DEV, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, labels=x).loss / accum
            loss.backward()
        if pen is not None:
            (lam * pen(model)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if frozen:
            restore(frozen)
        if step % 50 == 0:
            rate = ((step + 1) * BATCH_TOKENS - tok0) / (time.time() - t0 + 1e-9)
            eta = (total_steps - step) * BATCH_TOKENS / rate / 3600
            print(f"[{tag}] step {step}/{total_steps} loss={float(loss)*accum:.3f} "
                  f"{rate/1e3:.0f}k tok/s eta {eta:.1f}h", flush=True)
        if (step + 1) % (CKPT_EVERY // BATCH_TOKENS) == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            step=step + 1), ckdir / "ckpt_latest.pt")
    torch.save(dict(model=model.state_dict(), step=total_steps),
               ckdir / "final.pt")
    import pandas as pd
    rows = [json.loads(ln) for ln in open(evals_f)]
    pd.DataFrame([{k: v for k, v in r.items() if k != "wm"} for r in rows]
                 ).to_parquet(OUT / f"{tag}.parquet")
    (OUT / f"{tag}_meta.json").write_text(json.dumps(dict(
        arm=args.arm, seed=args.seed, tokens=args.tokens, plant=plant_meta,
        seeded=[f"L{l}H{h}" for l, h in SEEDED]), indent=1))
    print(f"[{tag}] COMPLETE", flush=True)


# ------------------------------------------------------------------ gate

def cmd_gate(args):
    from transformers import GPTNeoXConfig  # noqa
    model = make_model(0)
    # (1) layout + rotary reconstruction: manual scores vs module attention probs
    x = torch.randint(0, 50304, (2, 64), device=DEV)
    with eager_attn(model), torch.no_grad():
        out = model(x, output_attentions=True)
        A_ref = out.attentions[0][0, 0].float().cpu().numpy()      # L0H0 probs
        hs = model.gpt_neox.embed_in(x)
        ln = model.gpt_neox.layers[0].input_layernorm
        n0 = ln(hs)[0].double().cpu().numpy()                      # [T, 768]
    W = _qkv(model, 0).detach().double().cpu().numpy()
    bias = model.gpt_neox.layers[0].attention.query_key_value.bias
    bq = bias[0 * 3 * DH: 0 * 3 * DH + DH].detach().double().cpu().numpy()
    bk = bias[0 * 3 * DH + DH: 0 * 3 * DH + 2 * DH].detach().double().cpu().numpy()
    q = n0 @ W[0 * 3 * DH: 0 * 3 * DH + DH].T + bq                 # [T, 64]
    k = n0 @ W[0 * 3 * DH + DH: 0 * 3 * DH + 2 * DH].T + bk
    T = q.shape[0]
    pos = np.arange(T)

    def rot(v):
        o = v.copy()
        for t in range(NPAIR):
            cA, sA = np.cos(pos * THETA[t]), np.sin(pos * THETA[t])
            o[:, t] = v[:, t] * cA - v[:, t + NPAIR] * sA
            o[:, t + NPAIR] = v[:, t] * sA + v[:, t + NPAIR] * cA
        return o
    qr = np.concatenate([rot(q[:, :ROT]), q[:, ROT:]], 1)
    kr = np.concatenate([rot(k[:, :ROT]), k[:, ROT:]], 1)
    S = qr @ kr.T / math.sqrt(DH)
    S[np.triu_indices(T, 1)] = -1e30
    A_man = np.exp(S - S.max(1, keepdims=True))
    A_man /= A_man.sum(1, keepdims=True)
    err = np.abs(A_man - A_ref).max()
    print(f"[gate] layout+rotary reconstruction (L0H0 attn probs): max|Δ|={err:.2e} "
          f"{'PASS' if err < 1e-3 else 'FAIL'}")
    # (2) plant + phase recovery + norm match
    ref_norms = {(l, h): float(_qkv(model, l)[k_rows(h)].norm()) for l, h in SEEDED}
    apply_plant(model)
    errs = [np.abs(((recover_phases(model, l, h) - THETA + np.pi) % (2 * np.pi))
                   - np.pi).max() for l, h in SEEDED]
    nrm = max(abs(float(_qkv(model, l)[k_rows(h)].norm()) / ref_norms[(l, h)] - 1)
              for l, h in SEEDED)
    print(f"[gate] plant: max phase err {max(errs):.2e} "
          f"{'PASS' if max(errs) < 1e-4 else 'FAIL'} | k-rot norm dev {nrm:.2e} "
          f"{'PASS' if nrm < 1e-5 else 'FAIL'}")
    # (3) penalty
    model2 = make_model(0)
    p = penalty_imag(model2)
    p.backward()
    g = model2.gpt_neox.layers[0].attention.query_key_value.weight.grad
    print(f"[gate] penalty@init = {float(p):.4f} (∈[0,1], ≈0.5 expected) | "
          f"grad finite: {bool(torch.isfinite(g).all())}")
    del model2
    # (4) throughput bench
    mm = np.memmap(DATA / "tokens.bin", dtype=np.uint16, mode="r")
    micro, steps = args.micro_bs, 12
    accum = BATCH_TOKENS // (micro * CTX)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.01, fused=True)
    model.train()
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        for m in range(accum):
            lo = ((s * accum + m) * micro) * CTX
            x = torch.from_numpy(np.array(mm[lo:lo + micro * CTX], dtype=np.int64)
                                 ).view(micro, CTX).to(DEV)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, labels=x).loss / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    rate = steps * BATCH_TOKENS / dt
    print(f"[gate] throughput: {rate/1e3:.0f}k tok/s (micro_bs={micro}, accum={accum}) "
          f"→ {4e9/rate/3600:.1f} h per 4B-token run "
          f"{'PASS' if rate >= 110_000 else 'BELOW TARGET — trim ladder'}")
    print(f"[gate] GPU mem peak: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "run"])
    ap.add_argument("--arm", default="free", choices=["free", "assist", "constraint"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=float, default=4e9)
    ap.add_argument("--micro-bs", type=int, default=32, dest="micro_bs")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    args.tokens = int(args.tokens)
    torch.set_num_threads(4)
    {"gate": cmd_gate, "run": cmd_run}[args.cmd](args)
