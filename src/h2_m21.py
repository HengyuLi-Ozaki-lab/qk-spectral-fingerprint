"""M2.1 — dose–response & dissection (H2_EXECUTION_BRIEF §A; frozen P-M21a–d/P-A5 in §3).

Blocks: A1 dissection (nd_init / nd_reg / [nd_both = reuse M1.3]), A2 solution-fraction
dose (rope solfrac f, ape solmix α), A3 rope_imag λ grid (all fresh, n=5), A4 anti-assist
reg λ{0.3,3} (λ{1,10} reused from H1), A5 frozen-scaffold probe (restore-after-step),
A6 ban-weight service for M2.3 (scheduled FIRST; handoff via results/h1/LOG.md).

Design decisions + operationalization sharpenings S-M21-1…7: results/h2/LOG.md Step-0
entry (recorded pre-launch). Reuse-equivalence: same construction + same seed reproduced
trainB bit-exactly in H1; reused cells are listed in the LOG.

Subcommands: gate | sanity | run --shard i --nshards K | analyze
Runs → results/h2/runs/ (all with save_weights — M2.3 raw material).
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, itertools, json, math, time
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG
import p10_training_intervention as p10
from h1_hinge import (NPAIR, THETA, SEED_HEADS, LAYER, formation_step, _params,
                      _norm_match, _rope_pair_rotation, _sym_plant, _algebra_phases,
                      _recover_phases, penalty_toward_real, penalty_toward_symM,
                      penalty_assist_imag, penalty_assist_sym)

OUT2 = CFG.RESULTS / "h2" / "runs"
OUT2.mkdir(parents=True, exist_ok=True)
H1RUNS = CFG.RESULTS / "h1" / "runs"
FIGS2 = CFG.RESULTS / "h2" / "figures"
FIGS2.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ constructions

def solfrac_phases(seed, h, f):
    """S-M21-2: base = H1 algebra draw; uniform-random subset of round(f·16) pairs
    overwritten with θ_t. f=0 ≡ H1 algebra, f=1 ≡ H1 solution, exactly."""
    base = _algebra_phases(seed, h).copy()
    k = int(round(f * NPAIR))
    idx = np.sort(np.random.default_rng(999_000_000 + seed * 1000 + h)
                  .choice(NPAIR, k, replace=False))
    base[idx] = THETA[idx]
    return base, [int(i) for i in idx]


def _ape_plant_factors(model):
    """Rank-32 factors of M = Σ_k p_k p_{k−1}ᵀ (corrected orientation, H1 LOG C1)."""
    Wp = model.W_pos.detach().double().cpu().numpy()
    M = Wp[1:].T @ Wp[:-1]
    dh = model.cfg.d_head
    U, s, Vt = np.linalg.svd(M)
    return U[:, :dh] * np.sqrt(s[:dh]), Vt[:dh].T * np.sqrt(s[:dh])


def ape_solmix_init(model, alpha):
    """S-M21-3: per matrix W_new = rescale(α·Ŵ_plant + (1−α)·Ŵ_free) to ‖W_free‖."""
    Uq, Vk = _ape_plant_factors(model)
    PQ, PK = _params(model)
    meta = {}
    with torch.no_grad():
        for h in SEED_HEADS:
            for P, F in ((PQ, Uq), (PK, Vk)):
                W = P[h]
                free = W.clone()
                plant = torch.as_tensor(F, dtype=W.dtype, device=W.device)
                mix = alpha * plant / plant.norm() + (1 - alpha) * free / free.norm()
                W.copy_(_norm_match(mix, free.norm()))
            meta[f"h{h}"] = dict(alpha=alpha)
    return meta


def make_init(arm, scheme, seed, extra):
    holder = {}
    if arm in ("free", "nd_reg", "imag", "sym", "assist_reg"):
        return None, holder

    def fn(model):
        if arm == "nd_init":
            if scheme == "rope":
                holder.update(_rope_pair_rotation(
                    model, {h: np.zeros(NPAIR) for h in SEED_HEADS}))
            else:
                holder.update(_sym_plant(model, seed))
        elif arm == "solfrac":
            ph, subsets = {}, {}
            for h in SEED_HEADS:
                p, idx = solfrac_phases(seed, h, extra["f"])
                ph[h] = p
                subsets[f"h{h}"] = idx
            holder.update(_rope_pair_rotation(model, ph))
            holder["planted_pairs"] = subsets
            holder["f"] = extra["f"]
        elif arm == "solmix":
            holder.update(ape_solmix_init(model, extra["alpha"]))
        elif arm == "sol_frozen":
            holder.update(_rope_pair_rotation(model, {h: THETA.copy() for h in SEED_HEADS}))
            holder["frozen"] = True
        else:
            raise ValueError(arm)
        holder["arm"], holder["scheme"], holder["seed"] = arm, scheme, seed
    return fn, holder


def extra_penalties(scheme):
    return {
        "nd_init": None, "solfrac": None, "solmix": None, "sol_frozen": None,
        "nd_reg": penalty_toward_real if scheme == "rope" else penalty_toward_symM,
        "assist_reg": penalty_assist_imag if scheme == "rope" else penalty_assist_sym,
    }


# ------------------------------------------------------------------ frozen runner (A5)

def run_frozen(scheme, constraint, lam, seed, steps=6000, bs=64, lr=1e-3, eval_every=100,
               outdir=None, wd=0.01, init_fn=None, extra_pens=None, save_weights=False,
               tag_suffix="", freeze=True):
    """Verbatim p10.run loop + restore-after-step of the seeded heads' W_Q/W_K slices
    (S-M21-4: exact freeze incl. decoupled WD). freeze=False must replicate p10.run —
    verified by the `sanity` subcommand before the frozen arm launches."""
    import torch.nn.functional as F
    model = p10.make_model(scheme, seed)
    if init_fn is not None:
        init_fn(model)
    PQ, PK = _params(model)
    saved = {h: (PQ[h].detach().clone(), PK[h].detach().clone()) for h in SEED_HEADS}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200))
    rng = torch.Generator().manual_seed(10_000 + seed)
    registry = dict(free=None, sym=p10.penalty_sym, imag=p10.penalty_imag)
    if extra_pens:
        registry.update(extra_pens)
    pen_fn = registry[constraint]
    hist = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            model.eval()
            ev = p10.evaluate(model, torch.Generator().manual_seed(99))
            pen_now = float(pen_fn(model).item()) if pen_fn else float("nan")
            wm = p10.weight_metrics(model)
            hist.append(dict(step=step,
                             **{k: v for k, v in ev.items()
                                if k not in ("prev_all", "ind_all", "kernels")},
                             penalty=pen_now,
                             prev_all=json.dumps(ev["prev_all"]),
                             ind_all=json.dumps(ev["ind_all"]),
                             kernels=json.dumps(ev["kernels"]), wm=json.dumps(wm)))
            model.train()
        if step == steps:
            break
        x, _ = p10.gen_batch(bs, rng)
        x = x.to(p10.DEV)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, p10.V), x[:, 1:].reshape(-1))
        if pen_fn is not None:
            loss = loss + lam * pen_fn(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if freeze:
            with torch.no_grad():
                for h in SEED_HEADS:
                    PQ[h].copy_(saved[h][0])
                    PK[h].copy_(saved[h][1])
    import pandas as pd
    df = pd.DataFrame(hist)
    tag = f"{scheme}_{constraint}_lam{lam}_seed{seed}{tag_suffix}"
    if outdir:
        df.to_parquet(outdir / f"{tag}.parquet")
        if save_weights:
            torch.save(model.state_dict(), outdir / f"{tag}.pt")
    return df, tag


# ------------------------------------------------------------------ worklist

F_INTERIOR = (0.25, 0.5, 0.75)
A_INTERIOR = (0.25, 0.5, 0.75)
L_A3 = (0.3, 1.0, 3.0, 10.0)
L_A4_NEW = (0.3, 3.0)


def worklist():
    """(scheme, constraint, lam, seed, extra). A6 + free anchors first (H1-B polls A6)."""
    combos = []
    for s in range(3):                                        # A6 — service, FIRST
        combos.append(("rope", "imag", 10.0, s, {}))
        combos.append(("ape", "sym", 10.0, s, {}))
    for scheme in ("rope", "ape"):                            # free anchors (regression)
        for s in range(5):
            combos.append((scheme, "free", 0.0, s, {}))
    for scheme in ("rope", "ape"):                            # A1 dissection
        for s in range(5):
            combos.append((scheme, "nd_init", 0.0, s, {}))
            combos.append((scheme, "nd_reg", 1.0, s, {}))
    for f in F_INTERIOR:                                      # A2 rope interior
        for s in range(5):
            combos.append(("rope", "solfrac", 0.0, s, dict(f=f, suffix=f"_f{f:g}")))
    for a in A_INTERIOR:                                      # A2 ape interior
        for s in range(5):
            combos.append(("ape", "solmix", 0.0, s, dict(alpha=a, suffix=f"_a{a:g}")))
    for lam in L_A3:                                          # A3 (λ10 s0-2 = A6 tags, skipped)
        for s in range(5):
            combos.append(("rope", "imag", lam, s, {}))
    for lam in L_A4_NEW:                                      # A4 new λ
        for scheme in ("rope", "ape"):
            for s in range(5):
                combos.append((scheme, "assist_reg", lam, s, {}))
    for s in range(5):                                        # A5 frozen
        combos.append(("rope", "sol_frozen", 0.0, s, dict(frozen=True)))
    # dedupe (A3 λ10 seeds 0-2 duplicate A6 rope entries)
    seen, out = set(), []
    for c in combos:
        key = (c[0], c[1], c[2], c[3], c[4].get("suffix", ""))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def cmd_run(args):
    combos = worklist()
    shard = combos[args.shard::args.nshards]
    print(f"[shard {args.shard}/{args.nshards}] {len(shard)} runs", flush=True)
    for scheme, cons, lam, seed, extra in shard:
        suffix = extra.get("suffix", "")
        tag = f"{scheme}_{cons}_lam{lam}_seed{seed}{suffix}"
        if (OUT2 / f"{tag}.parquet").exists():
            print(f"[skip] {tag}", flush=True)
            continue
        fn, meta = make_init(cons, scheme, seed, extra)
        t0 = time.time()
        if extra.get("frozen"):
            df, tag = run_frozen(scheme, cons, lam, seed, outdir=OUT2, init_fn=fn,
                                 extra_pens=extra_penalties(scheme), save_weights=True,
                                 freeze=True)
        else:
            df, tag = p10.run(scheme, cons, lam, seed, steps=6000, outdir=OUT2,
                              init_fn=fn, extra_penalties=extra_penalties(scheme),
                              save_weights=True, tag_suffix=suffix)
        wall = time.time() - t0
        fstep = formation_step(df)
        last = df.iloc[-1]
        meta.update(tag=tag, formation_step=fstep, wall_s=round(wall, 1),
                    final_ce_pred=float(last.ce_pred), final_prev=float(last.prev_best),
                    final_ind=float(last.ind_best), **{k: v for k, v in extra.items()
                                                       if k != "suffix"})
        (OUT2 / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print(f"[done] {tag}: formation={fstep} ce={last.ce_pred:.3f} "
              f"prev={last.prev_best:.2f} ({wall/60:.1f} min)", flush=True)


# ------------------------------------------------------------------ gates

@torch.no_grad()
def _snap(model):
    model.eval()
    ev = p10.evaluate(model, torch.Generator().manual_seed(99))
    wm = p10.weight_metrics(model)
    return ev, wm


def cmd_gate(args):
    import pandas as pd
    rows = []
    for scheme, arm, seeds, extras in (
            ("rope", "nd_init", range(5), [{}]),
            ("ape", "nd_init", range(5), [{}]),
            ("rope", "solfrac", range(5), [dict(f=f) for f in F_INTERIOR]),
            ("ape", "solmix", range(5), [dict(alpha=a) for a in A_INTERIOR]),
    ):
        for extra in extras:
            for seed in seeds:
                ref = p10.make_model(scheme, seed)
                Pq, Pk = _params(ref)
                refn = {h: (float(Pq[h].norm()), float(Pk[h].norm())) for h in SEED_HEADS}
                del ref
                torch.cuda.empty_cache()
                model = p10.make_model(scheme, seed)
                fn, meta = make_init(arm, scheme, seed, extra)
                fn(model)
                ev, wm = _snap(model)
                PQ, PK = _params(model)
                for h in SEED_HEADS:
                    i = LAYER * 4 + h
                    ker = np.array(ev["kernels"][i])
                    row = dict(scheme=scheme, arm=arm, seed=seed, head=h,
                               f=extra.get("f", np.nan), alpha=extra.get("alpha", np.nan),
                               qn=float(PQ[h].norm()) / refn[h][0],
                               kn=float(PK[h].norm()) / refn[h][1],
                               rif=wm[i]["rope_imag_frac"], dir_frac=wm[i]["dir_frac"],
                               k_m1=float(ker[7]), k_0=float(ker[8]),
                               ce0=ev["ce_pred"], planted=np.nan)
                    if scheme == "rope":
                        rec = _recover_phases(model, h)
                        if arm == "solfrac":
                            intended, idx = solfrac_phases(seed, h, extra["f"])
                            err = np.abs(((rec - intended + np.pi) % (2 * np.pi)) - np.pi)
                            row["planted"] = len(idx) if float(err.max()) < 1e-4 else -1
                            row["phase_err0"] = float(err.max())
                        if arm == "nd_init":
                            row["phase_err0"] = float(np.abs(
                                ((rec + np.pi) % (2 * np.pi)) - np.pi).max())
                    rows.append(row)
                del model
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_parquet(CFG.RESULTS / "h2" / "gates_m21.parquet")
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- gate checks ---")
    ok = True
    nm = float(np.abs(df[["qn", "kn"]].values - 1).max())
    print(f"norm match: max|ratio−1| = {nm:.2e}  {'PASS' if nm < 1e-4 else 'FAIL'}")
    ok &= nm < 1e-4
    ce = df.ce0
    print(f"ce_pred@init ∈ [{ce.min():.3f}, {ce.max():.3f}]  "
          f"{'PASS' if ce.min() >= math.log(64) - 0.05 and ce.max() <= 5.0 else 'FAIL'}")
    sf = df[df.arm == "solfrac"]
    bad = sf[np.abs(sf.planted - (sf.f * NPAIR).round()) > 0]
    print(f"solfrac planted-count == f·16: {'PASS' if len(bad) == 0 else f'FAIL ({len(bad)} rows)'}")
    ok &= len(bad) == 0
    rifmin = float(df[df.scheme == "rope"].rif.min())
    print(f"rope rif@init min = {rifmin:.4f}  {'PASS' if rifmin >= 0.4 else 'FAIL'}")
    ok &= rifmin >= 0.4
    nd = df[(df.arm == "nd_init") & (df.scheme == "rope")]
    print(f"nd_init rope φ̂=0: max err = {nd.phase_err0.max():.2e}; dir_frac max = "
          f"{nd.dir_frac.max():.2e}")
    sm = df[df.arm == "solmix"].groupby("alpha").k_m1.median()
    sm_full = pd.concat([pd.Series({0.0: 0.01, 1.0: 2.1}), sm]).sort_index()  # H1 refs
    mono = bool(np.all(np.diff(sm_full.values) >= -0.05))
    print(f"solmix k(−1)@init medians (incl. H1 refs α=0/1): "
          f"{ {round(k,2): round(v,3) for k, v in sm_full.items()} }  "
          f"monotone(±0.05): {'PASS' if mono else 'FAIL'} | α=.5 > 0.3: "
          f"{'PASS' if sm.get(0.5, 0) > 0.3 else 'FAIL'}")
    ok &= mono and sm.get(0.5, 0) > 0.3
    print(f"\nGATES {'ALL PASS' if ok else 'FAILURE — do not launch'}")


def cmd_sanity(args):
    """A5 runner gate: run_frozen(freeze=False) must reproduce H1 rope solution seed0."""
    fn, _ = make_init("sol_frozen", "rope", 0, {})
    df, _ = run_frozen("rope", "sol_frozen", 0.0, 0, outdir=None, init_fn=fn,
                       extra_pens=extra_penalties("rope"), freeze=False)
    fs = formation_step(df)
    print(f"[sanity] run_frozen(freeze=False, rope solution seed0): formation={fs} "
          f"(H1 reference 700) → {'PASS' if fs == 700 else 'FAIL'}")


# ------------------------------------------------------------------ analysis

def _load(dirpath, scheme, cons, lam, seeds, suffix=""):
    import pandas as pd
    out = []
    for s in seeds:
        f = dirpath / f"{scheme}_{cons}_lam{lam}_seed{s}{suffix}.parquet"
        if not f.exists():
            print(f"[warn] missing {f.name}")
            continue
        df = pd.read_parquet(f)
        last = df.iloc[-1]
        wm = json.loads(last.wm)
        pa = json.loads(last.prev_all)
        best = int(np.argmax(pa))
        ker = np.array(json.loads(last.kernels)[best])
        out.append(dict(scheme=scheme, cons=cons, lam=lam, seed=s, suffix=suffix,
                        fstep=formation_step(df), final_ce=float(last.ce_pred),
                        final_prev=float(last.prev_best), final_ind=float(last.ind_best),
                        prev_head=best, seeded=best in (0, 1),
                        rif=wm[best]["rope_imag_frac"], dirf=wm[best]["dir_frac"],
                        kpeak=int(np.arange(-8, 1)[int(np.argmax(ker))])))
    import pandas as pd
    return pd.DataFrame(out)


def classify(row):
    if row.scheme == "rope":
        if row.rif >= 0.25:
            return "default"
        if row.rif <= 0.10 and row.kpeak == -1:
            return "nondefault"
    else:
        if row.dirf >= 0.35:
            return "default"
        if row.dirf <= 0.10:
            return "nondefault"
    return "unclassified"


def spearman_exact(x, y, alternative):
    """Exact permutation p for Spearman rho on ≤6 points (ties via average ranks)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    def rho(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-300))
    obs = rho(rx, ry)
    cnt = tot = 0
    for perm in itertools.permutations(ry):
        r = rho(rx, np.array(perm))
        cnt += (r <= obs + 1e-12) if alternative == "less" else (r >= obs - 1e-12)
        tot += 1
    return obs, cnt / tot


def cmd_analyze(args):
    import pandas as pd
    from scipy.stats import fisher_exact
    from h1_analyze import mwu_exact, bh_fdr
    E = {}                                                    # cells
    E["free_rope"] = _load(OUT2, "rope", "free", 0.0, range(5))
    E["free_ape"] = _load(OUT2, "ape", "free", 0.0, range(5))
    print("=" * 88, "\nREGRESSION — in-batch free anchors vs trainB")
    for k, ref in (("free_rope", (940, 55)), ("free_ape", (600, 0))):
        g = E[k].fstep
        print(f"  {k}: {g.mean():.0f}±{g.std(ddof=1):.0f} {sorted(g)} vs {ref[0]}±{ref[1]}"
              f" → {'PASS' if abs(g.mean() - ref[0]) <= 50 else 'FAIL'}")
    print("\nREGRESSION — A6/A3 replicas vs trainB per-seed formations")
    tb = {("rope", "imag"): {0: 1300, 1: 1400, 2: 1300, 3: 1000, 4: 1400},
          ("ape", "sym"): {0: 1400, 1: 1600, 2: 1800}}
    for (sc, cons), refs in tb.items():
        got = _load(OUT2, sc, cons, 10.0, sorted(refs))
        gg = dict(zip(got.seed, got.fstep))
        ok = all(gg.get(s) == v for s, v in refs.items())
        print(f"  {sc}_{cons}_λ10: {gg} vs trainB {refs} → {'PASS' if ok else 'MISMATCH (list)'}")

    # ---------------- P-M21a dose curves
    print("=" * 88, "\nP-M21a — solution-fraction dose (cell medians; endpoints reused from H1)")
    rope_cells = {0.0: _load(H1RUNS, "rope", "assist_init_algebra", 0.0, range(5))}
    for f in F_INTERIOR:
        rope_cells[f] = _load(OUT2, "rope", "solfrac", 0.0, range(5), f"_f{f:g}")
    rope_cells[1.0] = _load(H1RUNS, "rope", "assist_init_solution", 0.0, range(5))
    ape_cells = {0.0: E["free_ape"]}
    for a in A_INTERIOR:
        ape_cells[a] = _load(OUT2, "ape", "solmix", 0.0, range(5), f"_a{a:g}")
    ape_cells[1.0] = _load(H1RUNS, "ape", "assist_init_solution", 0.0, range(5))
    ps = []
    for name, cells, free_med in (("rope", rope_cells, float(E["free_rope"].fstep.median())),
                                  ("ape", ape_cells, float(E["free_ape"].fstep.median()))):
        xs = sorted(cells)
        med = [float(cells[x].fstep.median()) for x in xs]
        rho, p = spearman_exact(np.array(xs), np.array(med), "less")
        ps.append(p)
        print(f"  {name}: medians {dict(zip(xs, med))} (free {free_med:.0f}) | "
              f"Spearman ρ={rho:+.2f} one-sided p={p:.4f}")
        for x in xs[1:]:
            pp, rb, hl = mwu_exact(cells[x].fstep.values, cells[xs[0]].fstep.values)
            print(f"    vs {name}[{xs[0]}]: cell {x:g} p={pp:.4f} rb={rb:+.2f} HL={hl:+.0f}"
                  f"  (descriptive)")
    q = bh_fdr(ps)
    print(f"  BH q over 2 schemes: rope {q[0]:.4f}, ape {q[1]:.4f}")

    # ---------------- P-M21b dissection
    print("=" * 88, "\nP-M21b — dissection (implementation flips vs free)")
    for scheme in ("rope", "ape"):
        cells = dict(
            free=E[f"free_{scheme}"],
            init_only=_load(OUT2, scheme, "nd_init", 0.0, range(5)),
            reg_only=_load(OUT2, scheme, "nd_reg", 1.0, range(5)),
            both=_load(H1RUNS, scheme, "assist_nondefault", 1.0, range(5)))
        for k, t in cells.items():
            t["impl"] = t.apply(classify, axis=1)
            flips = int((t.impl == "nondefault").sum())
            tab = [[flips, len(t) - flips],
                   [int((cells['free'].impl == 'nondefault').sum()),
                    len(cells["free"]) - int((cells['free'].impl == 'nondefault').sum())]]
            fp = fisher_exact(tab, alternative="greater")[1] if k != "free" else float("nan")
            print(f"  {scheme} {k:9s}: flips {flips}/{len(t)} | fstep "
                  f"{t.fstep.mean():5.0f}±{0 if len(t) < 2 else t.fstep.std(ddof=1):3.0f} | "
                  f"ce {t.final_ce.mean():.3f} | Fisher p={fp:.4f}")

    # ---------------- P-M21c constraint non-monotonicity
    print("=" * 88, "\nP-M21c — rope_imag λ grid (all fresh n=5)")
    cells = {lam: _load(OUT2, "rope", "imag", lam, range(5)) for lam in L_A3}
    med = {lam: float(cells[lam].fstep.median()) for lam in L_A3}
    print(f"  medians: {med}")
    p_c, rb, hl = mwu_exact(cells[10.0].fstep.values, cells[1.0].fstep.values)
    print(f"  MWU one-sided cost(λ1) > cost(λ10): p={p_c:.4f} rb={rb:+.2f} HL={hl:+.0f}")
    rho, ptr = spearman_exact(np.array(L_A3), np.array([med[l] for l in L_A3]), "greater")
    print(f"  descriptive 4-pt Spearman (increasing): ρ={rho:+.2f} p={ptr:.4f}")

    # ---------------- P-M21d anti-assist dose
    print("=" * 88, "\nP-M21d — toward-default reg λ grid (λ1/λ10 reused from H1)")
    ps = []
    for scheme in ("rope", "ape"):
        cells = {0.3: _load(OUT2, scheme, "assist_reg", 0.3, range(5)),
                 1.0: _load(H1RUNS, scheme, "assist_reg", 1.0, range(5)),
                 3.0: _load(OUT2, scheme, "assist_reg", 3.0, range(5)),
                 10.0: _load(H1RUNS, scheme, "assist_reg", 10.0, range(5))}
        med = [float(cells[l].fstep.median()) for l in (0.3, 1.0, 3.0, 10.0)]
        rho, p = spearman_exact(np.array((0.3, 1.0, 3.0, 10.0)), np.array(med), "greater")
        ps.append(p)
        print(f"  {scheme}: medians {dict(zip((0.3, 1., 3., 10.), med))} | "
              f"Spearman(增) ρ={rho:+.2f} p={p:.4f} | final ce λ3: "
              f"{cells[3.0].final_ce.mean():.3f} λ10(H1): {cells[10.0].final_ce.mean():.3f}")
    q = bh_fdr(ps)
    print(f"  BH q: rope {q[0]:.4f}, ape {q[1]:.4f}")

    # ---------------- P-A5 scaffold
    print("=" * 88, "\nP-A5 — frozen-scaffold probe (rope)")
    fro = _load(OUT2, "rope", "sol_frozen", 0.0, range(5))
    sol = _load(H1RUNS, "rope", "assist_init_solution", 0.0, range(5))
    fr_med, so_med, fr_free = (float(fro.fstep.median()), float(sol.fstep.median()),
                               float(E["free_rope"].fstep.median()))
    sp_frozen, sp_norm = fr_free - fr_med, fr_free - so_med
    rho_ratio = sp_frozen / (sp_norm + 1e-300)
    p_f, rb_f, hl_f = mwu_exact(fro.fstep.values, E["free_rope"].fstep.values)
    print(f"  medians: free {fr_free:.0f} | solution(H1) {so_med:.0f} | frozen {fr_med:.0f}"
          f" {sorted(fro.fstep)}")
    print(f"  speedup: normal {sp_norm:.0f}, frozen {sp_frozen:.0f} → ρ = {rho_ratio:.2f} "
          f"(P-A5 needs ≥0.5) | frozen-vs-free MWU p={p_f:.4f} rb={rb_f:+.2f}")

    # figures
    make_figs(rope_cells, ape_cells, cells_a3={l: _load(OUT2, 'rope', 'imag', l, range(5))
                                               for l in L_A3})
    print("[analyze] done")


def make_figs(rope_cells, ape_cells, cells_a3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, cells, name in (
            (axes[0], rope_cells, "rope: solution fraction f"),
            (axes[1], ape_cells, "ape: plant amplitude α")):
        xs = sorted(cells)
        for x in xs:
            v = cells[x].fstep.values
            ax.plot(np.full(len(v), x) + np.random.default_rng(0).uniform(-.02, .02, len(v)),
                    v, "o", color="tab:blue", alpha=0.7)
        ax.plot(xs, [float(cells[x].fstep.median()) for x in xs], "k-o", lw=2, ms=4)
        ax.set_xlabel(name)
        ax.set_ylabel("formation step")
        ax.set_title(f"A2 dose ({name.split(':')[0]})")
    xs = sorted(cells_a3)
    for x in xs:
        v = cells_a3[x].fstep.values
        axes[2].plot(np.full(len(v), x) * (1 + np.random.default_rng(0).uniform(-.03, .03, len(v))),
                     v, "o", color="tab:red", alpha=0.7)
    axes[2].plot(xs, [float(cells_a3[x].fstep.median()) for x in xs], "k-o", lw=2, ms=4)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("rope_imag λ")
    axes[2].set_ylabel("formation step")
    axes[2].set_title("A3 constraint dose (non-monotonicity check)")
    fig.tight_layout()
    fig.savefig(FIGS2 / "m21_dose.png", dpi=140)
    print(f"[fig] {FIGS2 / 'm21_dose.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "sanity", "run", "analyze"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(4)
    {"gate": cmd_gate, "sanity": cmd_sanity, "run": cmd_run,
     "analyze": cmd_analyze}[args.cmd](args)
