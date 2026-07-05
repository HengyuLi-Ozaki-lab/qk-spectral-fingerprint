"""P2 (targeted-task ablation, spec §6.1). General CE dilutes head-specific effects;
here we kill each head's M_A and measure the delta on an INDUCTION task (loss on the
2nd copy of repeated-random sequences, which requires prev-token→induction routing).

If M_A's prev-token role (report F2) is real, dICL_sym should track prev/induction
scores more sharply than the general-CE dCE_sym did.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import load_model
from observables import _ablation_hooks

DEV = "cuda"


@torch.no_grad()
def induction_ce(model, toks, seq_len, fwd_hooks=None):
    logits = model.run_with_hooks(toks, return_type="logits", fwd_hooks=fwd_hooks or [])
    lp = torch.log_softmax(logits.float(), dim=-1)
    ps = torch.arange(seq_len + 1, 2 * seq_len)                 # induction-predictable positions
    tgt = toks[:, ps + 1]                                       # [b, len(ps)]
    ce = -lp[:, ps].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [b, len(ps)]
    return float(ce.mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n_seq", type=int, default=300)
    ap.add_argument("--seq_len", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_parquet(C.CACHE / f"{args.model}_head_full.parquet").reset_index(drop=True)
    model = load_model(args.model, DEV)

    g = torch.Generator(device="cpu").manual_seed(0)
    r = torch.randint(0, model.cfg.d_vocab, (args.n_seq, args.seq_len), generator=g)
    bos = torch.full((args.n_seq, 1), model.tokenizer.bos_token_id)
    toks = torch.cat([bos, r, r], dim=1).to(DEV)

    base = induction_ce(model, toks, args.seq_len)
    print(f"[P2-targeted] baseline induction CE = {base:.4f}")
    vals = np.full(len(df), np.nan)
    for idx, row in df.iterrows():
        l, h = int(row["layer"]), int(row["head"])
        vals[idx] = induction_ce(model, toks, args.seq_len, _ablation_hooks(model, l, h, "sym")) - base
        if h == 0:
            print(f"  layer {l:2d} dICL(head0)={vals[idx]:+.4f}", flush=True)
    df["dICL_sym"] = vals
    df.to_parquet(C.CACHE / f"{args.model}_head_full.parquet")

    from scipy.stats import spearmanr
    print("\n[P2-targeted] does killing M_A hurt the INDUCTION task per head?")
    for t in ["prev", "ind", "dup", "dir_frac", "D_head", "dCE_sym", "MA_fro" if "MA_fro" in df else "imag_mass"]:
        if t in df:
            rho, p = spearmanr(df[t], df["dICL_sym"]); print(f"   corr(dICL_sym, {t:12s}) = {rho:+.3f}  (p={p:.2e})")
    print(f"\n   dICL_sym: median {df.dICL_sym.median():+.4f}  max {df.dICL_sym.max():+.4f}  (baseline {base:.3f})")
    cols = ["layer", "head", "dICL_sym", "prev", "ind", "dir_frac", "dCE_sym"]
    print("\n   top-8 heads whose M_A the induction task needs:")
    print(df.sort_values("dICL_sym", ascending=False)[cols].head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
