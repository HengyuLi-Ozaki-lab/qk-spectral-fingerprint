"""Im(M_t) causal ablation on Llama-3-8B (REVIEW_P1 pending item 3).

Same design as p5_rope_ablation, adapted for an 8B bf16 model with d_vocab≈128k:
chunked induction CE (full-batch logits would need ~13 GB), induction task only
(general-CE sweep skipped), all 1024 heads. The pre-registered identity test:
does the argmax-|ΔInd| head equal the #1 prev-scoring head (permutation p=1/1024)?
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model
from rope import rope_params
from p5_rope_ablation import _hooks, validate_delta
from p2b_targeted import induction_ce

DEV = "cuda"


@torch.no_grad()
def induction_ce_chunked(model, toks, seq_len, fwd_hooks=None, chunk=32):
    vals = []
    for i in range(0, len(toks), chunk):
        vals.append(induction_ce(model, toks[i:i + chunk], seq_len, fwd_hooks))
    return float(np.mean(vals))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    model = load_model("llama-3-8b", DEV)
    npair, theta = rope_params(model.cfg)[1], rope_params(model.cfg)[2]
    err = validate_delta(model, npair, theta)
    print(f"[llama-ablate] delta decomposition check (fp32 internal identity): {err:.2e} "
          f"({'PASS' if err < 1e-2 else 'CHECK'})", flush=True)

    g = torch.Generator(device="cpu").manual_seed(0)
    r = torch.randint(0, model.cfg.d_vocab, (args.n_seq, args.seq_len), generator=g)
    bos = model.tokenizer.bos_token_id
    bos = model.tokenizer.eos_token_id if bos is None else bos
    toks = torch.cat([torch.full((args.n_seq, 1), bos), r, r], dim=1).to(DEV)

    base = induction_ce_chunked(model, toks, args.seq_len, chunk=args.chunk)
    print(f"[llama-ablate] baseline induction CE = {base:.4f}  (n_seq={args.n_seq})", flush=True)

    df = pd.read_parquet(C.CACHE / "llama-3-8b_head_full.parquet").reset_index(drop=True)
    vals = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        l, h = int(row["layer"]), int(row["head"])
        hk = _hooks(model, l, h, npair, theta)
        vals[i] = induction_ce_chunked(model, toks, args.seq_len, hk, chunk=args.chunk) - base
        if h == 0:
            print(f"  layer {l:2d}  dInd(head0)={vals[i]:+.4f}", flush=True)
    df["dInd_rope"] = vals
    df.to_parquet(C.CACHE / "llama-3-8b_head_full.parquet")

    # ---- the pre-registered identity test + summary ----
    im, ip = int(np.nanargmax(vals)), int(df["prev"].idxmax())
    print(f"\n[llama-ablate] argmax dInd = L{df.loc[im,'layer']}.H{df.loc[im,'head']} "
          f"(dInd={vals[im]:+.3f}, prev={df.loc[im,'prev']:.3f})")
    print(f"               #1 prev head = L{df.loc[ip,'layer']}.H{df.loc[ip,'head']} (prev={df.loc[ip,'prev']:.3f})")
    print(f"               IDENTITY {'MATCH (permutation p=1/1024)' if im==ip else 'NO MATCH'}")
    pr_rank = int((df["prev"] >= df.loc[im, "prev"]).sum())
    print(f"               argmax head's prev rank = {pr_rank}/1024 (permutation p={pr_rank}/1024)")
    print(f"               frac |dInd|<0.01 = {(np.abs(vals)<0.01).mean():.2f}")
    for t in ["prev", "D_head", "dir_frac", "rope_imag_frac", "freq_centroid"]:
        rho, p = spearmanr(df[t], df["dInd_rope"]); print(f"   corr(dInd, {t:15s}) = {rho:+.3f} (p={p:.1e})")
    cols = ["layer", "head", "dInd_rope", "prev", "D_head", "rope_imag_frac"]
    print("\n top-10 heads by dInd:")
    print(df.sort_values("dInd_rope", ascending=False)[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
