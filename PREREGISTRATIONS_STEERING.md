# Pre-registered predictions — steering note

Frozen in the project plan before the corresponding runs were launched (assistance hinge + selection
pilot: 2026-07-03; dose–response/dissection grid: 2026-07-05). Falsified predictions are reported in
the note as findings. Verbatim text:

## Assistance hinge & selection pilot (before any assisted run)

- **P-A1 (acceleration)**: default-aligned assist-init speeds formation vs free in *both* schemes
  (one-sided MWU, n=5, per scheme; effect visible at ≥1 λ). *Falsifier*: no arm speeds → hinge fails
  → pivot. **Outcome: FALSIFIED as frozen** (algebra-init non-significant in both schemes); the
  solution-init positive control speeds both schemes at rank-biserial +1.0.
- **P-A2 (specificity)**: cross-scheme placebo assist (RoPE-style under APE and vice versa) gives no
  speedup (or slows). *Falsifier*: placebo speeds too → "assistance" is a generic init-scale
  artifact, not algebra steering. **Outcome: PASSED.**
- **P-A3 (selection)**: weak non-default assist raises P(non-default implementation) vs free (Fisher
  exact, thresholds rif 0.25 / dir_frac 0.35 pre-set) at ≤10% formation-time cost. **Outcome:
  selection clause PASSED maximally (5/5 vs 0/5, both schemes); cost clause PASSED for RoPE (−23%),
  FALSIFIED for APE (+30%).**

## Dose–response / dissection grid (before the 108-run grid)

- **P-M21a (dose)**: median formation step is non-increasing in f (rope) and α (ape), with endpoints
  consistent with the hinge (f=0/α=0 ≈ free-level; f=1/α=1 reproduces the solution-arm speedup).
  Test: one-sided trend (Jonckheere or Spearman on cell medians), per scheme, BH over 2. *Falsifier*:
  flat everywhere, or non-monotone with an interior slowdown. **Outcome: CONFIRMED** (rope graded,
  ape threshold).
- **P-M21b (dissection)**: the implementation flip is carried by the reg component: reg-only flips
  ≥4/5; init-only flips ≤1/5 (both schemes; Fisher vs free). *Falsifier per scheme*: init-only flips
  ≥3/5. **Outcome: CONFIRMED maximally** (reg-only 5/5 both schemes — RoPE at zero formation cost;
  init-only 0/5).
- **P-M21c (non-monotonicity)**: rope_imag ban cost is non-monotone over λ∈{0.3,1,3,10} with
  cost(1) > cost(10) replicating at n=5 (MWU one-sided λ1 vs λ10). *Falsifier*: monotone/flat — the
  n=3 flag was noise. **Outcome: FALSIFIED (flag was noise); replacement descriptive finding: the ban
  cost is a plateau (binds from λ=0.3, flat across 33×).**
- **P-M21d (anti-assist dose)**: toward-default reg cost increases monotonically in λ in both schemes
  (trend test). *Falsifier*: non-monotone or null at small λ. **Outcome: CONFIRMED** (APE
  superlinear with capability damage).
- **P-A5 (scaffold freeze)**: freezing the planted heads preserves ≥half of the solution-init speedup
  (scaffold = representational/gradient-shaping, not plant adaptation). *Falsifier*: speedup vanishes
  under freezing. **Outcome: CONFIRMED with surprise — freezing doubles the speedup** (500 vs 700
  unfrozen vs 900 free).

P1-paper pre-registrations (dynamics Q1–Q3; intervention P1–P4) are in `INTERVENTION_PLAN.md`.
