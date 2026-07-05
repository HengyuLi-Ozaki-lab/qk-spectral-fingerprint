"""Candidate A — Pythia checkpoint spectral natural history (INTERVENTION_PLAN §1).

Per checkpoint (TL checkpoint_value=step, native), compute:
  - weight-space per head: dir_frac, D_head, rope_imag_frac (reuse fast low-rank + rope paths)
  - behavioral per head: prefix-matching (induction) + prev-token, on repeated-random
  - global: ICL score (loss@late - loss@early) as the Olsson phase-change marker
  - K-composition: r(5.2 -> each induction head) to time the circuit WIRING
Pre-registered primary outcome (locked 2026-07-03, INTERVENTION_PLAN §5):
  lead/lag of the eventual top prev head's rope_imag_frac/D_head rise vs its prefix-matching rise.

Disk-safe: TL caches each checkpoint under HF hub; we delete after processing.
GPU pinned by caller (CUDA_VISIBLE_DEVICES). BLAS threads pinned. float64 spectral.
"""
from __future__ import annotations
import argparse, gc, json, shutil
import numpy as np, torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from extract import centering_matrix
from metrics import kspace_spectrum
from rope import rope_params, head_rope_metrics
from p4_rope import thin_svd

DEV = "cuda"
RANK_TOL = 1e-9


def spectral_row(WQh, WKh, npair, theta):
    Uk, Sk, Vk = thin_svd(WQh, WKh)
    s = kspace_spectrum(Sk, Vk.T @ Uk)
    rm = head_rope_metrics(WQh, WKh, npair, theta)
    return dict(dir_frac=s["dir_frac"], D_head=s["D_head"],
                rope_imag_frac=rm["rope_imag_frac"], freq_centroid=rm["freq_centroid"])


@torch.no_grad()
def behavioral(model, n_seq=150, seq_len=64, seed=0):
    """prev-token + induction (prefix-matching) per head, on [BOS,r,r]; plus global ICL score."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    bos = model.tokenizer.bos_token_id
    bos = model.tokenizer.eos_token_id if bos is None else (bos if bos is not None else 0)
    prev = np.zeros((L, H)); ind = np.zeros((L, H)); seen = 0
    p = torch.arange(seq_len + 1, 2 * seq_len + 1)
    tprev, tind = p - 1, p - seq_len + 1
    icl_num = 0.0
    for i in range(0, n_seq, 50):
        bs = min(50, n_seq - i)
        r = torch.randint(0, model.cfg.d_vocab, (bs, seq_len), generator=g)
        toks = torch.cat([torch.full((bs, 1), bos), r, r], dim=1).to(DEV)
        logits, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_pattern"))
        for l in range(L):
            A = cache[f"blocks.{l}.attn.hook_pattern"]
            prev[l] += A[:, :, p, tprev].mean(dim=(0, 2)).float().cpu().numpy() * bs
            ind[l] += A[:, :, p, tind].mean(dim=(0, 2)).float().cpu().numpy() * bs
        # ICL score: mean NLL on 2nd copy - 1st copy (negative = induction working)
        lp = torch.log_softmax(logits.float(), -1)
        tgt = toks[:, 1:]
        nll = -lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [bs, 2L]
        first = nll[:, 1:seq_len].mean(); second = nll[:, seq_len + 1:2 * seq_len].mean()
        icl_num += float(second - first) * bs
        seen += bs; del cache, logits
    return prev / seen, ind / seen, icl_num / seen


@torch.no_grad()
def kcomp(model, target_lh, prev_lh, norm="LN"):
    d = model.cfg.d_model
    Cm = (torch.eye(d, device=DEV) - 1.0 / d) if norm == "LN" else torch.eye(d, device=DEV)
    l2, h2 = target_lh; l1, h1 = prev_lh
    if l1 >= l2:
        return float("nan")
    M = Cm @ (model.W_Q[l2, h2].float() @ model.W_K[l2, h2].float().T) @ Cm
    OV = model.W_V[l1, h1].float() @ model.W_O[l1, h1].float()
    return float(torch.linalg.norm(M @ OV.T) / (torch.linalg.norm(M) * torch.linalg.norm(OV) + 1e-30))


def process_step(model_key, tl_name, step, prev_lh, ind_lh, norm, rope):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        tl_name, checkpoint_value=step, fold_ln=True,
        center_writing_weights=(norm == "LN"), center_unembed=(norm == "LN"), device=DEV)
    model.eval()
    cfg = model.cfg
    npair, theta = (rope_params(cfg)[1], rope_params(cfg)[2]) if rope else (1, np.array([1.0]))
    Cm = centering_matrix(cfg.d_model) if norm == "LN" else None
    WQ = model.W_Q.detach().double().cpu().numpy(); WK = model.W_K.detach().double().cpu().numpy()
    fold = (lambda W: Cm @ W) if Cm is not None else (lambda W: W)

    prev, ind, icl = behavioral(model)
    rows = []
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            sr = spectral_row(fold(WQ[l, h]), fold(WK[l, h]), npair, theta)
            sr.update(step=step, layer=l, head=h, prev=float(prev[l, h]), ind=float(ind[l, h]))
            rows.append(sr)
    # K-composition of the eventual prev head into eventual induction heads
    kc = {f"kcomp_{il}_{ih}": kcomp(model, (il, ih), prev_lh, norm) for (il, ih) in ind_lh}
    summ = dict(step=step, icl_score=icl, **kc)
    # eventual-prev-head spectral+behavioral trajectory
    pl, ph = prev_lh
    prow = next(r for r in rows if r["layer"] == pl and r["head"] == ph)
    summ.update({f"prev_{k}": prow[k] for k in ["dir_frac", "D_head", "rope_imag_frac", "prev", "ind"]})
    del model; gc.collect(); torch.cuda.empty_cache()
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--prev", default="5,2"); ap.add_argument("--ind", default="8,6;10,9;11,14")
    ap.add_argument("--steps", default="")   # comma list; default = log grid below
    args = ap.parse_args()
    tl = C.MODELS[args.model]["tl_name"]; norm = C.MODELS[args.model]["norm"]; rope = C.MODELS[args.model]["rope"]
    prev_lh = tuple(int(x) for x in args.prev.split(","))
    ind_lh = [tuple(int(x) for x in p.split(",")) for p in args.ind.split(";")]
    if args.steps:
        steps = [int(x) for x in args.steps.split(",")]
    else:
        steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000,
                 32000, 48000, 64000, 96000, 128000, 143000]
    outdir = C.CACHE / f"ckpt_{args.model}"; outdir.mkdir(exist_ok=True)
    import pandas as pd
    summ_all = []
    for step in steps:
        sj = outdir / f"step{step}_summary.json"
        if (outdir / f"step{step}.parquet").exists() and sj.exists():
            summ_all.append(json.loads(sj.read_text()))
            print(f"[step {step:>6}] cached, skip", flush=True); continue
        try:
            rows, summ = process_step(args.model, tl, step, prev_lh, ind_lh, norm, rope)
        except Exception as e:
            print(f"[step {step}] FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True); continue
        pd.DataFrame(rows).to_parquet(outdir / f"step{step}.parquet")
        sj.write_text(json.dumps(summ))
        summ_all.append(summ)
        pd.DataFrame(summ_all).sort_values("step").to_parquet(C.CACHE / f"{args.model}_ckpt_summary.parquet")
        kcs = " ".join(f"{k.split('_',1)[1]}={v:.2f}" for k, v in summ.items() if k.startswith("kcomp"))
        print(f"[step {step:>6}] icl={summ['icl_score']:+.3f} | prev-head 5.2: "
              f"prev={summ['prev_prev']:.2f} ind={summ['prev_ind']:.2f} "
              f"rope_imag={summ['prev_rope_imag_frac']:.3f} D={summ['prev_D_head']:.3f} | Kcomp {kcs}", flush=True)
        # disk hygiene: delete ALL cached revisions of this repo properly (blobs+snapshots+refs)
        try:
            from huggingface_hub import scan_cache_dir
            sc = scan_cache_dir()
            for repo in sc.repos:
                if repo.repo_id == f"EleutherAI/{args.model}":
                    hashes = [r.commit_hash for r in repo.revisions]
                    if hashes:
                        sc.delete_revisions(*hashes).execute()
        except Exception as e:
            print(f"  (cache cleanup warning: {e})", flush=True)
    print(f"\n[A] summary -> {C.CACHE / (args.model + '_ckpt_summary.parquet')}  ({len(summ_all)} checkpoints)")


if __name__ == "__main__":
    main()
