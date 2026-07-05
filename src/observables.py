"""P2 — anchoring spectral metrics to behavior (decides H1). Spec §6.

  P2a  symmetrization ablation (DECISIVE): per head, swap M→M_S (kill M_A) and
       M→M_A (kill M_S) via a hook on hook_attn_scores; measure ΔCE on a corpus.
  P2b  taxonomy: previous-token / duplicate / induction scores (from attention
       patterns on repeated-random sequences) + OV copying score (Elhage).
  P2c  score asymmetry ‖S−Sᵀ‖/‖S‖ on real inputs vs weight-space dir_frac.
  P2d  incremental validity: does (D_head−null) predict ablation ΔCE beyond dir_frac?

Run on GPU1 (CUDA_VISIBLE_DEVICES=1). Ablation hook works in float32 (matches scores);
spectral metrics stay float64 (P1).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model

DEV = "cuda"


# ----------------------------------------------------------------------------- corpus & loss

def get_tokens(model, n_seq=1000, seq_len=128):
    from datasets import load_dataset
    from transformer_lens.utils import tokenize_and_concatenate
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    tok_ds = tokenize_and_concatenate(ds, model.tokenizer, max_length=seq_len, column_name="text")
    toks = tok_ds["tokens"][:n_seq]
    if not torch.is_tensor(toks):
        toks = torch.tensor(np.array(toks))
    return toks.long()


@torch.no_grad()
def ce_loss(model, tokens, batch_size=250, fwd_hooks=None):
    tot, n = 0.0, 0
    for i in range(0, len(tokens), batch_size):
        b = tokens[i:i + batch_size].to(DEV)
        loss = model.run_with_hooks(b, return_type="loss", fwd_hooks=fwd_hooks or [])
        tot += float(loss) * b.shape[0]; n += b.shape[0]
    return tot / n


# ----------------------------------------------------------------------------- P2a ablation

def _ablation_hooks(model, l, h, mode):
    """mode='sym' → keep M_S (subtract M_A part); mode='anti' → keep M_A (subtract M_S)."""
    WQ = model.W_Q[l, h].float(); WK = model.W_K[l, h].float()
    M = WQ @ WK.T
    M_op = 0.5 * (M - M.T) if mode == "sym" else 0.5 * (M + M.T)   # the part to REMOVE
    scale = float(model.blocks[0].attn.attn_scale)
    store = {}
    ln1 = f"blocks.{l}.ln1.hook_normalized"
    scr = f"blocks.{l}.attn.hook_attn_scores"

    def grab(act, hook):
        store["n"] = act
        return act

    def modify(scores, hook):
        n = store["n"]                                  # [b,pos,d] f32
        delta = torch.einsum("bpd,dD,bkD->bpk", n, M_op, n) / scale
        scores[:, h] = scores[:, h] - delta
        return scores

    return [(ln1, grab), (scr, modify)]


def run_ablation(model, tokens, df, modes=("sym", "anti"), batch_size=250):
    base = ce_loss(model, tokens, batch_size)
    print(f"[P2a] baseline CE = {base:.4f}  (n_seq={len(tokens)}, seq_len={tokens.shape[1]})")
    for mode in modes:
        col = f"dCE_{mode}"
        vals = np.full(len(df), np.nan)
        for idx, r in df.iterrows():
            l, h = int(r["layer"]), int(r["head"])
            ce = ce_loss(model, tokens, batch_size, _ablation_hooks(model, l, h, mode))
            vals[idx] = ce - base
            if h == 0:
                print(f"  [{mode}] layer {l:2d} … dCE(head0)={ce-base:+.4f}", flush=True)
        df[col] = vals
    df.attrs["baseline_ce"] = base
    return df


# ----------------------------------------------------------------------------- P2b taxonomy

@torch.no_grad()
def taxonomy_scores(model, n_seq=200, seq_len=64, batch_size=50, seed=0):
    """prev-token / duplicate / induction attention scores on repeated-random sequences."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    acc = {k: np.zeros((L, H)) for k in ("prev", "dup", "ind")}
    seen = 0
    bos_id = model.tokenizer.bos_token_id
    if bos_id is None:
        bos_id = model.tokenizer.eos_token_id if model.tokenizer.eos_token_id is not None else 0
    p = torch.arange(seq_len + 1, 2 * seq_len + 1)                 # second-copy query positions
    tgt = dict(prev=p - 1, dup=p - seq_len, ind=p - seq_len + 1)
    pat = lambda l: f"blocks.{l}.attn.hook_pattern"
    for i in range(0, n_seq, batch_size):
        bs = min(batch_size, n_seq - i)
        r = torch.randint(0, model.cfg.d_vocab, (bs, seq_len), generator=g)
        bos = torch.full((bs, 1), bos_id)
        toks = torch.cat([bos, r, r], dim=1).to(DEV)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_pattern"))
        for l in range(L):
            A = cache[pat(l)]                                      # [b,H,Q,K]
            for name, t in tgt.items():
                acc[name][l] += A[:, :, p, t].mean(dim=(0, 2)).float().cpu().numpy() * bs
        seen += bs
    return {k: v / seen for k, v in acc.items()}


@torch.no_grad()
def copying_scores(model):
    """OV copying score (Elhage): fraction of eig(W_O·W_U·W_E·W_V) with positive real part."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    rep = H // model.W_V.shape[1] if model.W_V.shape[1] != H else 1   # GQA: V head = h // rep
    UE = (model.W_U @ model.W_E).float()                          # [d_model,d_model]
    out = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            Mc = (model.W_O[l, h].float() @ UE @ model.W_V[l, h // rep].float()).cpu().numpy().astype(np.float64)
            w = np.linalg.eigvals(Mc)
            out[l, h] = float((w.real > 0).mean())
    return out


# ----------------------------------------------------------------------------- P2c score asymmetry

@torch.no_grad()
def score_asymmetry(model, tokens, n_seq=100):
    """Empirical ‖S−Sᵀ‖_F/‖S‖_F per head on real inputs (pre-mask bilinear scores)."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    scale = float(model.blocks[0].attn.attn_scale)
    toks = tokens[:n_seq].to(DEV)
    num = np.zeros((L, H)); den = np.zeros((L, H))
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("ln1.hook_normalized"))
    for l in range(L):
        n = cache[f"blocks.{l}.ln1.hook_normalized"].float()      # [b,pos,d]
        for h in range(H):
            M = (model.W_Q[l, h].float() @ model.W_K[l, h].float().T)
            S = torch.einsum("bpd,dD,bkD->bpk", n, M, n) / scale   # [b,pos,pos]
            asym = S - S.transpose(-1, -2)
            num[l, h] = float(torch.linalg.norm(asym, dim=(1, 2)).mean())
            den[l, h] = float(torch.linalg.norm(S, dim=(1, 2)).mean())
    return num / den


# ----------------------------------------------------------------------------- P2d analysis

def analyze(df: pd.DataFrame):
    from scipy.stats import spearmanr
    try:
        from scipy.stats import false_discovery_control as bh
    except Exception:
        bh = None
    print("\n[P2] ===== H1 tests (Spearman; BH-FDR across heads) =====")
    df["dir_frac_dev"] = df.dir_frac - df.dir_frac_null_mean
    df["D_head_dev"] = df.D_head - df.D_head_null_mean

    targets = [c for c in ["dCE_sym", "dCE_anti", "prev", "dup", "ind", "copying", "score_asym"] if c in df]
    preds = [c for c in ["D_head", "D_head_dev", "dir_frac", "dir_frac_dev", "henrici",
                         "content_pos_frac", "imag_mass"] if c in df]
    rows, pvals = [], []
    for t in targets:
        for pcol in preds:
            rho, p = spearmanr(df[pcol], df[t])
            rows.append(dict(target=t, predictor=pcol, rho=rho, p=p)); pvals.append(p)
    res = pd.DataFrame(rows)
    if bh is not None and len(pvals):
        res["p_fdr"] = bh(res["p"].values)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # P2a decisive: does removing M_A (dCE_sym) hurt directional heads more?
    if "dCE_sym" in df:
        print("\n[P2a] DECISIVE — ΔCE from killing M_A vs directionality:")
        for pcol in ["dir_frac_dev", "D_head_dev", "henrici"]:
            rho, p = spearmanr(df[pcol], df["dCE_sym"])
            print(f"   corr(dCE_sym, {pcol:14s}) = {rho:+.3f}  (p={p:.2e})")

    # P2d incremental validity: nested OLS on dCE_sym
    if "dCE_sym" in df:
        y = df["dCE_sym"].values
        def r2(Xcols):
            X = np.column_stack([np.ones(len(df))] + [df[c].values for c in Xcols])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
            return 1 - ss_res / ss_tot
        r2_base = r2(["dir_frac_dev"])
        r2_full = r2(["dir_frac_dev", "D_head_dev"])
        r2_hen = r2(["dir_frac_dev", "D_head_dev", "henrici"])
        # partial corr(D_head_dev, dCE_sym | dir_frac_dev)
        from numpy.polynomial import polynomial as _p
        def resid(a, b):  # residual of a after regressing on b
            A = np.column_stack([np.ones(len(b)), b]); be, *_ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ be
        ry = resid(y, df["dir_frac_dev"].values)
        rx = resid(df["D_head_dev"].values, df["dir_frac_dev"].values)
        pr, pp = spearmanr(rx, ry)
        print("\n[P2d] INCREMENTAL VALIDITY of D_head over dir_frac (the 'earns its keep' test):")
        print(f"   R²(dCE_sym ~ dir_frac_dev)                    = {r2_base:.3f}")
        print(f"   R²(+ D_head_dev)                              = {r2_full:.3f}   ΔR² = {r2_full-r2_base:+.3f}")
        print(f"   R²(+ D_head_dev + henrici)                    = {r2_hen:.3f}   ΔR² = {r2_hen-r2_base:+.3f}")
        print(f"   partial corr(D_head_dev, dCE_sym | dir_frac)  = {pr:+.3f}  (p={pp:.2e})")
    return res


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n_seq", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--skip_ablation", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(C.CACHE / f"{args.model}_head_metrics.parquet").reset_index(drop=True)
    print(f"[P2] loading {args.model} on GPU1 ...")
    model = load_model(args.model, DEV)

    print("[P2b] taxonomy (prev/dup/induction) + copying score ...")
    tax = taxonomy_scores(model)
    cop = copying_scores(model)
    for name, grid in [("prev", tax["prev"]), ("dup", tax["dup"]), ("ind", tax["ind"]), ("copying", cop)]:
        df[name] = [grid[int(r["layer"]), int(r["head"])] for _, r in df.iterrows()]

    print("[P2] tokenizing corpus (NeelNanda/pile-10k) ...")
    tokens = get_tokens(model, args.n_seq, args.seq_len)

    print("[P2c] score asymmetry on real inputs ...")
    sa = score_asymmetry(model, tokens)
    df["score_asym"] = [sa[int(r["layer"]), int(r["head"])] for _, r in df.iterrows()]

    if not args.skip_ablation:
        print("[P2a] symmetrization ablation over corpus (144 heads × 2 modes) ...")
        df = run_ablation(model, tokens, df, batch_size=args.batch)

    out = C.CACHE / f"{args.model}_head_full.parquet"
    df.to_parquet(out)
    print(f"[P2] full table → {out}")
    analyze(df)


if __name__ == "__main__":
    main()
