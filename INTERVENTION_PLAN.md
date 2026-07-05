# Intervention Experiments — Comprehensive Investigation & Evaluation

> Purpose: the single highest-leverage addition to the QK paper is an intervention/dynamics result that upgrades it
> from *characterization* ("prev heads are rotational under RoPE") to *controllable mechanism* ("the positional
> scheme dictates which spectral solutions training can reach — and blocking the rotational channel blocks/reroutes
> induction"). This kills the main top-venue objection ("RoPE is rotations — expected"). Two prior-art agents swept
> the space (2026-07-03); both target cells are **OPEN**. Feasibility verified on our hardware. Date: 2026-07-03.

---

## 0. Prior-art verdicts (condensed; full agent reports in session transcript)

**Cell 1 — training-dynamics of QK spectral structure: OPEN.** Nobody tracks a per-head, weight-space,
complex/rotational QK quantity across training and ties it to induction emergence. Near-miss map:

| work | per-head | weight-QK | complex spectrum | public ckpts | tied to head function/phase change |
|---|---|---|---|---|---|
| Tigges+ NeurIPS'24 (2407.10827) | ✓ | ✗ behavioral | ✗ | ✓ all 154 Pythia | ✓ (**induction ≈ 2×10⁹ tokens** — our alignment target) |
| Xu 2026 (2606.02378) ⚠ | ✓ | ✗ activations | ✗ (real SVs) | ✓ (10 coarse) | classes ✓, phase ✗ |
| Saponati ICML'25 | ✗ per-layer | ✓ | ✗ norm ratio | ✗ own runs | ✗ |
| Wang+ rLLC (2410.02984) | ✓ | ✗ loss-landscape | ✗ | ✗ toy | ✓ head differentiation |
| **Jamil & Kapadia 2026 (2605.18826)** ⚠ | ✓ | ✓ | **✓ eig(M)** | **✗ static only** | **✗ none** |

⚠ **Action item executed alongside this plan**: Jamil & Kapadia is now *verified* to compute per-head complex
eigenvalues of `M` on final weights (GPT-2/BERT/Pythia-410m) with **no behavioral anchoring, no nulls, no
positional-scheme contrast, no training axis**. They erode the literal phrase "the QK operator's own complex
spectrum remains unstudied" — main paper related-work updated to cite & differentiate (our four differentiators all
stand: behavior anchoring, matched nulls, causal ablation, cross-scheme design).

**Cell 2 — constrained-QK autoregressive training × induction formation: OPEN.** Nobody has trained causal LMs
with a symmetry/rotational constraint on `M` (or Im(M_t) suppression) and measured induction formation. Nearest:
- Symmetric-QK causal training **exists but unanalyzed**: Reformer (ICLR'20, shared-QK, enwik8 — trains fine, even
  slightly faster) and Kayyam+ 2026 (2606.04032: shared-QK causal LMs at 300M/1.2B → only **+4.9% ppl**). ⇒ the
  constraint is trainable; nobody looked inside.
- Encoder-only precedents: Courtois+ ACL-F'24 (symmetric dot-product BERT, explicitly "no conclusion for decoders");
  Saponati ICML'25 (symmetric **init** only, encoders only); Trockman & Kolter ICML'23 (mimetic init, vision).
- Training-time circuit interventions **by other means**: Singh+ ICML'24 (activation clamping through training —
  our methodological template), Olsson'22 smeared-key, Sahin+ 2025 Hapax (loss masking), Chen+ ICLR'24 "Sudden
  Drops" (regularize an interpretable attention property during MLM — the exact template, wrong objective/property).
- RoPE frequency surgery exists (p-RoPE ICLR'25, FoPE ICML'25, HoPE) but removes/replaces **frequency subspaces**;
  none suppress the **Im(M_t) phase component of the QK operator**, none measure induction formation.
- ⚠ **Strategic warning**: the *observational* PE×circuit-timing space is being colonized fast — Huang+ 2026
  (2601.20796: 2-layer decoders, APE/RoPE/ALiBi × induction formation; finds "RoPE struggles to form strong
  prev-token heads") and Xu 2026. **The weight-space constraint intervention is the defensible novelty — lead with
  it, and use Huang's base-rate finding as a design control, not a scoop.**

---

## 1. Candidate A — Pythia checkpoint spectral natural history (observational)

**Design.** Pythia-410m (+160m replicate), ~25 checkpoints (all early log-spaced 0,1,2,…,512 + every ~10k):
per checkpoint, load via TL `checkpoint_value=step` (native support, verified) → reuse the exact existing pipeline:
- weight-space: `D_head`, `dir_frac`, `rope_imag_frac`, `freq_centroid`, `hi_freq_dir` per head (+ matched null for
  a subsample);
- behavioral: prefix-matching (induction), prev-token score, ICL score (Olsson loss-at-500 − loss-at-50), 2nd-copy
  loss on repeated-random;
- alignment: the known behavioral transition ≈ 2×10⁹ tokens (Tigges) sits inside Pythia's dense early grid
  (step 1000 ≈ 2×10⁹ tokens at Pythia's 2M-token batches — the log-spaced 0–512 checkpoints straddle it; Aoyama
  2511.16893 notes OLMo's grid starts *too late*, Pythia's does not).

**Questions (pre-registered).**
- Q1: does head 5.2's (and the induction heads') `rope_imag_frac`/`D_head` **structured rise** lead, coincide with,
  or lag the behavioral prefix-matching rise? (Chen+ "structure onset precedes capability onset" template.)
- Q2: population-level: when does the **suppression** happen (median D_head z: 0 at init → −11 final)? Is
  "retain-vs-suppress" a phase-change-time event or gradual?
- Q3: does the *eventual* prev head already differ from peers **before** the behavioral transition (predictive
  weight signature)?

**Predictions.** At init all heads sit at the Ginibre null (verified analytically + empirically in the paper);
the differential (prev heads retain + structure, others suppress) emerges around the phase change; `rope_imag_frac`
of the eventual circuit heads rises **before or at** prefix-matching onset (structure→capability). Falsifier: the
spectral trajectory is flat/noisy or strictly lags behavior → rotational imprint is a *consequence*, not a
precursor — still publishable as the honest ordering.

**Cost.** Downloads ~25×0.9 GB sequential (delete-after-extract; 12 GB disk OK), GPU minutes per checkpoint.
**~1 GPU-day, mostly automated.** Feasibility de-risked: TL checkpoint loading verified; NeoX fused-QKV unpack
verified at 0 error (backup raw-safetensors path).

**Risks (from the literature).** Head-identity turnover across training (Tigges) → track head-slots AND function
labels separately; two-transitions caveat (Xu, Chen) → don't assume a single aligned event; per-checkpoint nulls
needed only for a subsample (D_head init ≈ null everywhere makes the *differential* the signal).

**Payoff.** A time axis for the paper's core finding + potentially a *predictive* weight signature. Novel per
agent-1 verdict (nearest neighbors are activations/behavioral/static). **Verdict: DO — cheap, near-certain value.**

### ✅ EXECUTED 2026-07-03 — results (Pythia-410m + 160m replicate, 22 checkpoints each)
- **Three-act natural history, replicated across both models:**
  (I) *Silence* (0–0.5B tokens): everything at the Ginibre null (behavior 0.01, rif=0.500, pop D=0.61, K-comp
  baseline). (II) *Sharp formation* (1–4B): prev behavior 0.37→0.95 between steps 512→1000 (**identical timing and
  even identical 0.37 waypoint in both models**); induction+ICL+K-comp wiring follow in the same window (Olsson
  ordering, Tigges ~2×10⁹ timing confirmed). (III) *Slow differentiation* (4–300B): population median D_head
  suppresses below null (0.61→0.395 / 0.61→0.453) while the prev head retains; absolute rif consolidates 0.50→0.578
  (same endpoint both models).
- **Pre-registered answers:** Q1 — the *percentile* rotational signature locks **within the same 512→1000 window as
  behavior** (simultaneous at our resolution; in both models the prev head's rif percentile is *low* (0.03–0.04)
  mid-formation at step 512, snapping to 0.90–0.99 by step 1000); the *absolute* consolidation clearly **lags**
  (100× longer). Q2 — suppression is a **post-formation** process (2–30B). Q3 — **NO predictive signature**
  (percentile at/below chance before formation) — honest pre-registered negative.
- **Implication (the A→B bridge):** the static "retain-vs-suppress" profile is the *end state* of
  differential-suppression dynamics that follow circuit formation. Observation cannot order structure vs function
  within the formation window — **only the constrained-training intervention (B) can settle necessity.** Figures:
  `results/figures/{pythia-410m_ckpt_dynamics, ckpt_dynamics_2models}.png`; data
  `results/cache/{pythia-410m,pythia-160m}_ckpt_{summary,analysis}.parquet` + per-step tables in `ckpt_*/`.
- *Ops note:* first run filled the disk (HF blobs not freed by snapshot deletion — root cause fixed with
  `scan_cache_dir().delete_revisions`); reruns are disk-safe (peak ≤1 checkpoint).

---

## 2. Candidate B — from-scratch constrained-training intervention (the flagship)

**Design (2 positional schemes × 3 QK constraints × ≥3 seeds).**
- **Models:** 2–4-layer decoder-only, d_model 256, 8 heads (d_k 32), context 256 — the Bietti/Reddy/Edelman scale
  where induction emergence is well-characterized and fast.
- **Data:** synthetic bigram+ICL corpus (à la Bietti "Birth of a Transformer": global bigrams + in-context repeated
  pairs) so induction is *the* solution and its formation time is crisp; secondary run on a small natural corpus
  (TinyStories/pile slice) for realism.
- **Positional schemes:** RoPE (full), learned-absolute (APE). (ALiBi arm optional third.)
- **Constraints (the new axis):**
  1. **free** (control);
  2. **sym-M**: penalty λ‖M_A‖²_F per head (soft; sweep λ) — or hard `W_K=W_Q G` tying as a robustness variant;
  3. **Im-suppressed (RoPE arm only)**: penalty λ Σ_t‖Im(M_t)‖²_F — *the surgical one*. **Key algebra (easy to get
     wrong): forcing M symmetric does NOT zero Im(M_t)** — static `M = Σ_t Re(M_t) + M_nonrot` contains only the
     Re parts, so the RoPE-phase channel Im(M_t) is a *separate* degree of freedom that must be penalized directly.
     (Hard variant: tie each rotary pair's columns, `c=κa, d=κb` ⇒ Im(M_t)=κ(abᵀ)−κ(abᵀ)=0.)
- **Outcomes:** induction formation step (prefix-matching crossing threshold; 2nd-copy loss drop), final induction
  strength, prev-head formation, relative-position kernel shape, final LM loss; plus the D_head/rope_imag_frac
  trajectories (ties A and B together).

**Pre-registered predictions.**
- P1 (free arms): replicate the known phase change; per Huang 2026, RoPE-free may form prev heads somewhat more
  slowly than APE-free — this is the **base-rate control**, and our effects are measured as constraint-minus-free
  *within* scheme (Huang's observation is thereby controlled, not confounded).
- P2 (**RoPE + Im-suppressed**): prev-token attention via RoPE phase becomes impossible; the cos-only kernel is
  even in Δ, so Δ=−1 selectivity must be built another way (an even kernel *can* bump at |Δ|=1 on the causal side
  but it is a strictly harder solution). Predict: **induction strongly delayed or rerouted**, kernel shape
  qualitatively different, possibly formed via content/nonrotary channel. This is the money arm.
- P3 (APE + sym-M): the position-embedding-matching solution is symmetric ⇒ predict **mild or no delay** (training-
  time constraints allow re-routing, unlike our post-hoc ablations — the contrast between "ablation kills it" (F2)
  and "constrained training routes around it" is itself informative).
- P4 (RoPE + sym-M): static-M symmetry doesn't block the phase channel ⇒ predict **small effect**, cleanly
  dissociating "antisymmetric M" from "RoPE phase" — a control only our decomposition makes possible.
- Falsifiers stated in advance: if P2 shows no delay at any λ, the rotational channel is dispensable at training
  time → the paper's causal claim gets scoped to trained-solution fragility (still coherent with F2-style ablation
  results, but the "mechanism dictates solutions" framing dies). Report either way.

**Cost.** ~2 schemes × 3 constraints × 3 seeds × (2 λ values where applicable) ≈ 18–30 runs × 20–60 min on A100 ≈
**1.5–3 GPU-days** + ~2 days engineering (training loop, constraint hooks, eval). Shared-GPU etiquette: run
serially, check `nvidia-smi`, checkpoint frequently.

**Risks.** (i) Toy-scale criticism — mitigate by pairing with A (real-model dynamics) and C (real-model
intervention); (ii) hyperparameter sensitivity of emergence times — mitigate with seeds + λ sweeps + the
within-scheme difference design; (iii) constraint leakage (optimizer finds Im(M_t)≈0-violating minima under soft
penalty) — monitor the penalized quantity; (iv) Huang 2026 base rates — handled by within-scheme contrasts.

**Payoff.** The headline upgrade: *"the positional mechanism dictates which spectral solutions are reachable;
blocking the rotational channel blocks/reroutes induction under RoPE but symmetrizing M under APE is nearly free."*
This is the experiment that answers "isn't this expected?" with a causal training-time result. **Verdict: DO as the
flagship, after A.**

### ✅ EXECUTED 2026-07-03 — results (18 runs, 2L attn-only d=128, per-seq random-map task; +seeds 3–4 running)
Infrastructure: task validated (ce_pred 4.16→0.23, prev 0.87 / induction 0.89 form at step ~1000); constraints
enforced hard (imag λ10 → Im share **0.0000**; sym λ10 → dir_frac 0.004–0.006); LM floor unaffected in every arm.
**Pre-registered scorecard (honest):**
- **P1 ✓** free arms replicate; APE forms faster than RoPE (600 vs 967±58) — Huang-consistent base rates.
- **P2 ⚠** strong version (block) FALSIFIED — Im-suppression delays only +38–52% and the circuit **reroutes**
  (final capability & head scores intact with rif=0.006); weak version (delay+reroute) confirmed.
- **P3 ✗ FALSIFIED — biggest surprise**: APE+sym is the LARGEST delay in the grid (600→1600, **2.7×**). The fast
  APE solution is antisymmetric embedding-matching; the symmetric variant (nearby-kernel + causal mask) is
  reachable but much slower to find. (Consistent with trained LLMs *ending* at the symmetric profile — big models
  converge there with 1000× more data; the toy shows the search cost.)
- **P4 ✗ formation-time / ✓ MECHANISM**: RoPE+sym delays comparably (+59%), **but the weight-level dissociation is
  clean and decisive**: sym-arm's prev head keeps **full phase usage (rif 0.519) with a fully symmetric static M
  (dir_frac 0.004)** — directional prev-attention through a symmetric M, impossible under APE; imag-arm reroutes the
  other way (rif 0.006, kernel shape changes, peak stays Δ=−1). Kernels: sym-M kernel ≈ free kernel exactly
  (phase untouched); Im-sup kernel qualitatively different. Figure: `results/figures/trainB_main.png`.
**Unified B verdict:** *no spectral channel is necessary (all arms reroute — degenerate solution space); every
constraint has a quantifiable search cost that reveals the scheme's DEFAULT solution; sym-M and Im(M_t) are
independent channels, either sufficient under RoPE.* Combined with A and the main paper: LLM profiles = end-state
default solutions; post-hoc ablations = trained-circuit dependence; training-time constraints = alternatives
reachable at measurable cost. **"The positional scheme sets the default spectral algebra of the solution, not a
hard constraint."**

### ✅ FINAL STATISTICS (n=5 seeds/cell, 2026-07-03)
Formation steps: ape_free **600±0**; rope_free **940±55**; rope_imag λ10 **1280±164**; rope_sym **1560±167**;
ape_sym **1740±241** (rope_imag λ1: 1467±208, n=3). Pre-registered contrasts (MWU exact one-sided, BH-FDR over 4):

| contrast | p | q_BH | rank-biserial | HL shift |
|---|---|---|---|---|
| P1 rope_free > ape_free | 0.0040 | 0.016 | **+1.00** | +300 steps |
| P3 ape_sym > ape_free | 0.0040 | 0.008 | **+1.00** | **+1200 steps (2.9×)** |
| P4 rope_sym > rope_free | 0.0040 | 0.005 | **+1.00** | +600 steps |
| P2 rope_imag > rope_free | 0.0079 | 0.008 | +0.92 | +400 steps |

All four significant with (near-)maximal effect sizes; weight signatures replicate on the new seeds (rope_sym
prev head: rif 0.523 / dir_frac 0.005; rope_imag: rif 0.005 / dir_frac 0.315). Figures:
`trainB_main.png` + `trainB_formation_n5.png`.

---

## 3. Candidate C — continued-pretraining knock-out on Pythia-410m (bridge)

**Design.** Continue pretraining Pythia-410m (~50–100M tokens) with (a) penalty λΣ‖Im(M_t)‖² on all heads, or (b)
targeted penalty on head 5.2 only; controls: matched-norm random-direction penalty + no-penalty continued training.
Track induction/prefix-matching, LM loss, and whether **another head takes over** (circuit plasticity — connects to
Tigges' head-turnover result).

**Cost.** 410m training on one A100-40GB: feasible (batch ~128×1024 with grad-accum), ~0.5–1 GPU-day.
**Risks.** Continued-training is messy (LR schedule mismatch, forgetting confounds — hence the matched controls);
interpretation weaker than B (post-formation plasticity ≠ formation).
**Payoff.** Real-scale bridge for the toy-scale criticism; the "does the circuit reroute?" question is novel.
**Verdict: OPTIONAL — run only if A+B land and reviewer-proofing at real scale is needed.**

---

## 4. Recommendation & sequencing

| | novelty | cost | risk | payoff | verdict |
|---|---|---|---|---|---|
| **A** checkpoint natural history | OPEN (weight-spectral axis) | ~1 GPU-day | low | time axis + possible predictive signature | **DO FIRST** |
| **B** constrained training 2×3 | OPEN (the unclaimed grid) | 1.5–3 GPU-days + eng. | medium | causal training-time claim — the top-venue lever | **FLAGSHIP** |
| **C** continued-training knock-out | OPEN | 0.5–1 GPU-day | medium-high | real-scale bridge | optional |

Sequence: **A now** (automatable, informs B's trajectory metrics) → **B** (pre-register the predictions above
verbatim) → C if needed. Publication packaging: A+B extend the current paper (§9 becomes "From characterization to
control") or, if B's results are rich, split as a companion paper ("Positional mechanisms dictate the spectral
solutions of attention") — decide after B.

**Immediate side-action (done with this plan):** cite & differentiate Jamil & Kapadia 2605.18826 in the main paper
(static per-head eig(M) exists; no behavior/nulls/causality/schemes) and in `PRIOR_ART.md`.

## 5. Pre-registration lock

The predictions P1–P4 and Q1–Q3 above are frozen as of 2026-07-03, before any checkpoint download or training run.
Primary outcome for A: lead/lag of `rope_imag_frac` rise vs prefix-matching rise for the eventual top prev head
(410m: expected 5.2, subject to turnover handling). Primary outcome for B: induction-formation step, constraint
arms vs free arm, within scheme, ≥3 seeds, BH-FDR over the 4 contrasts. No metric shopping; negative results
reported plainly.
