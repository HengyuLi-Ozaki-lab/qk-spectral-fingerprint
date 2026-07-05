# Fingerprint, Not Blueprint — code & data

Reproducibility bundle for:

- **Fingerprint, Not Blueprint: How Positional Schemes Set the Default Spectral Algebra of Attention**
  (`paper/qk_spectral.tex|.pdf`)
- **Steering Is Cheaper Than Banning: Assistance-Based Selection of Attention's Spectral Solutions**
  (`paper/steering_note.tex|.pdf`, companion note)

Li Hengyu, Institute for Solid State Physics, The University of Tokyo
(lihengyu@issp.u-tokyo.ac.jp)

## What is in here

| path | contents |
|---|---|
| `src/` | all pipelines. Main paper: `extract, decompose, metrics, observables, semantics, rope, p4_rope, p5_rope_ablation, p6_placebo, p7_llama_ablation, p8_remaining, p2b_targeted, p9_checkpoint_dynamics, p10_training_intervention`. Note: `h1_hinge, h1_analyze, h2_m21`. Figures: `paper_figs_v2.py` (every main-paper figure from the cached tables, CPU-only), `paper_kernel_refresh.py` |
| `results/cache/*_head_full.parquet` | per-head spectral + behavioral tables for the seven models (GPT-2, OPT-1.3B, GPT-Neo-1.3B, BLOOM-1b1, Pythia-410m, Pythia-1.4B, Llama-3-8B) |
| `results/cache/pythia-*_ckpt_*.parquet` | checkpoint natural-history trajectories (22 checkpoints × 2 models) |
| `results/cache/trainB/` | constrained-training grid: per-eval trajectories, per-head weight metrics, kernels (all seeds) |
| `results/h1/`, `results/h2/` | steering-note experiments: assistance hinge, selection pilot, dose–response/dissection grid (per-run trajectory tables + summary/gate tables; model weights omitted — regenerable from `src/h1_hinge.py`) |
| `INTERVENTION_PLAN.md` | pre-registration document for the main paper's dynamics (Q1–Q3) and intervention (P1–P4) claims, with outcomes |
| `PREREGISTRATIONS_STEERING.md` | frozen predictions for the note (P-A1/A2/A3, P-M21a–d, P-A5), with outcomes; falsified predictions are reported in the papers as findings |
| `environment/pip-freeze.txt` | exact Python environment (torch 2.7.1+cu118, transformer_lens 3.5.0) |

## Reproduce

```bash
PY=python  # conda env matching environment/pip-freeze.txt
# 1) all main-paper figures from the cached tables (CPU, ~1 min)
$PY src/paper_figs_v2.py
# 2) §6 statistics (model-level exact permutation p=0.029; top-k sensitivity; leave-one-model-out)
#    are direct computations on results/cache/*_head_full.parquet (percentile convention r/n)
# 3) per-head tables from scratch (GPU): e.g.
$PY src/p4_rope.py --model pythia-410m
# 4) checkpoint natural history (GPU; disk-safe HF-cache cycling built in)
$PY src/p9_checkpoint_dynamics.py --model pythia-410m
# 5) constrained-training grid of §10 (GPU, ~4 min/run)
$PY src/p10_training_intervention.py --grid full
# 6) steering-note experiments
$PY src/h1_hinge.py --help
```

Model revisions: HF defaults as of 2026-07 (TransformerLens `from_pretrained`; Pythia checkpoints via
`checkpoint_value=step`). Head detectors (previous-token, duplicate, induction prefix-matching on
repeated-random sequences; K-composition) are defined in `src/observables.py` / `src/extract.py`.

## License

Code: MIT (see `LICENSE`). Data tables: CC-BY-4.0.
