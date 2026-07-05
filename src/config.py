"""Project-wide paths, model registry, and numerical conventions.

Spectral work uses float64 (spec §4). GPU work is pinned to GPU1 at the shell
level via CUDA_VISIBLE_DEVICES=1 (shared workstation — do not touch GPU0).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
FIGS = RESULTS / "figures"
for _d in (RESULTS, CACHE, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# Model registry. GPT-2 small first: learned-absolute positions, LayerNorm,
# no RoPE confound (spec §9). RoPE track (P4) uses Pythia / Llama.
MODELS = {
    "gpt2":        dict(tl_name="gpt2",              norm="LN",  rope=False),
    "gpt2-medium": dict(tl_name="gpt2-medium",       norm="LN",  rope=False),
    "pythia-160m": dict(tl_name="pythia-160m",       norm="LN",  rope=True),   # checkpoint-dynamics replicate
    "pythia-410m": dict(tl_name="pythia-410m",       norm="LN",  rope=True),   # d_head=64 (=GPT-2) + RoPE → control
    "pythia-1.4b": dict(tl_name="pythia-1.4b",       norm="LN",  rope=True),
    "pythia-2.8b": dict(tl_name="pythia-2.8b",       norm="LN",  rope=True),
    # full-RoPE + RMSNorm + GQA (third architecture point)
    "llama-3-8b":  dict(tl_name="meta-llama/Meta-Llama-3-8B", norm="RMS", rope=True, dtype="bfloat16"),
    # learned-absolute anchors ≥1B (REVIEW_P1 pending item 2)
    "opt-1.3b":    dict(tl_name="facebook/opt-1.3b",          norm="LN",  rope=False),
    "gpt-neo-1.3b":dict(tl_name="EleutherAI/gpt-neo-1.3B",    norm="LN",  rope=False),
    # ALiBi third positional scheme (position enters as score bias, zero positional info in QK)
    "bloom-1b1":   dict(tl_name="bigscience/bloom-1b1",       norm="LN",  rope=False),
}

SPECTRAL_DTYPE = np.float64      # all decompositions
RECON_RTOL = 1e-4                # logit-match tolerance (spec §1)
GAUGE_ATOL = 1e-9                # gauge-invariance tolerance (spec §11)
SEED = 0
