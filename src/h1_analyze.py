"""H1 Workstream A — M1.2/M1.3 analysis (pre-specified in brief §3–§4; ROADMAP §3 frozen).

Primary family (M1.2): {rope: algebra vs free, ape: algebra vs free}, one-sided exact
MWU (enumerated — handles ties exactly), BH-FDR over the 2 tests.
Secondary (reported, not headline): solution vs free, reg λ1/λ10 vs free, placebos vs
free. Effect sizes: rank-biserial + Hodges–Lehmann shift. Capability table for every arm.

Usage: python src/h1_analyze.py m12 | m13
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import itertools, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG
from h1_hinge import formation_step, worklist

RUNS = CFG.RESULTS / "h1" / "runs"
FIGS = CFG.RESULTS / "h1" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
TRAINB_REF = {("ape", "free"): (600, 0), ("rope", "free"): (940, 55)}


def load_phase(phase):
    rows = []
    for scheme, cons, lam, seed, extra in worklist(phase):
        tag = f"{scheme}_{cons}_lam{lam}_seed{seed}{extra.get('suffix', '')}"
        f = RUNS / f"{tag}.parquet"
        if not f.exists():
            print(f"[warn] missing {tag}")
            continue
        df = pd.read_parquet(f)
        last = df.iloc[-1]
        wm = json.loads(last.wm)
        prev_all = json.loads(last.prev_all)
        best = int(np.argmax(prev_all))
        ker = np.array(json.loads(last.kernels)[best])         # offsets −8…0
        arm = cons if cons == "free" or lam == 0.0 else f"{cons}_l{lam:g}"
        rows.append(dict(
            scheme=scheme, arm=arm, cons=cons, lam=lam, seed=seed, tag=tag,
            fstep=formation_step(df),
            final_ce=float(last.ce_pred), final_prev=float(last.prev_best),
            final_ind=float(last.ind_best),
            prev_head=best, prev_head_seeded=bool(best in (0, 1)),
            prev_head_rif=wm[best]["rope_imag_frac"], prev_head_dir=wm[best]["dir_frac"],
            prev_head_score=float(prev_all[best]),
            prev_kernel_argmax=int(np.arange(-8, 1)[int(np.argmax(ker))]),
        ))
    return pd.DataFrame(rows)


def mwu_exact(x, y):
    """One-sided exact MWU by full enumeration: H1 = x stochastically SMALLER than y
    (x = assist formation steps, y = free). Returns p, rank-biserial, HL shift.
    rb > 0 means x < y (speedup); HL < 0 means x faster by |HL| steps."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    U = sum((xi < yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    pooled = np.concatenate([x, y])
    n = nx + ny
    cnt = 0
    tot = 0
    for idx in itertools.combinations(range(n), nx):
        xs = pooled[list(idx)]
        ys = pooled[[i for i in range(n) if i not in idx]]
        u = sum((xi < yj) + 0.5 * (xi == yj) for xi in xs for yj in ys)
        cnt += (u >= U)                       # as-or-more-extreme toward "x smaller"
        tot += 1
    p = cnt / tot
    rb = 2.0 * U / (nx * ny) - 1.0
    hl = float(np.median([xi - yj for xi in x for yj in y]))
    return p, rb, hl


def bh_fdr(ps):
    m = len(ps)
    order = np.argsort(ps)
    q = np.empty(m)
    prev = 1.0
    for rank_from_end, i in enumerate(order[::-1]):
        rank = m - rank_from_end
        prev = min(prev, ps[i] * m / rank)
        q[i] = prev
    return q


def analyze_m12():
    t = load_phase("m12")
    t.to_parquet(CFG.RESULTS / "h1" / "m12_summary.parquet")
    print(f"loaded {len(t)}/62 runs\n")

    # ---- regression gate: free arms vs trainB
    print("=" * 78, "\nREGRESSION GATE — free arms vs trainB reference")
    gate_ok = True
    for scheme in ("rope", "ape"):
        g = t[(t.scheme == scheme) & (t.arm == "free")].fstep
        m_ref, s_ref = TRAINB_REF[(scheme, "free")]
        ok = abs(g.mean() - m_ref) <= max(2 * (s_ref if s_ref > 0 else 25), 50)
        gate_ok &= ok
        print(f"  {scheme}_free: {g.mean():.0f}±{g.std(ddof=1):.0f} (n={len(g)}) "
              f"vs trainB {m_ref}±{s_ref}  {'PASS' if ok else 'FAIL'}  {sorted(g.tolist())}")
    if not gate_ok:
        print("  !! REGRESSION FAILED — stop, debug before interpreting anything")
        return

    # ---- formation table
    print("=" * 78, "\nFORMATION STEPS (per arm)")
    summ = t.groupby(["scheme", "arm"]).fstep.agg(["mean", "std", "count", list])
    print(summ.to_string())

    # ---- primary family
    print("=" * 78, "\nPRIMARY FAMILY (P-A1): assist_init_algebra vs free, one-sided exact MWU, BH-FDR/2")
    prim = []
    for scheme in ("rope", "ape"):
        x = t[(t.scheme == scheme) & (t.arm == "assist_init_algebra")].fstep.values
        y = t[(t.scheme == scheme) & (t.arm == "free")].fstep.values
        p, rb, hl = mwu_exact(x, y)
        prim.append(dict(scheme=scheme, contrast="algebra<free", p=p, rb=rb, hl=hl,
                         x_mean=x.mean(), y_mean=y.mean()))
    qs = bh_fdr([r["p"] for r in prim])
    for r, q in zip(prim, qs):
        r["q_BH"] = q
        print(f"  {r['scheme']}: algebra {r['x_mean']:.0f} vs free {r['y_mean']:.0f} | "
              f"p={r['p']:.4f} q={q:.4f} rb={r['rb']:+.2f} HL={r['hl']:+.0f}")

    # ---- secondary contrasts
    print("=" * 78, "\nSECONDARY (reported, not headline): arm vs free within scheme")
    sec = []
    for scheme in ("rope", "ape"):
        y = t[(t.scheme == scheme) & (t.arm == "free")].fstep.values
        for arm in ("assist_init_solution", "assist_reg_l1", "assist_reg_l10",
                    "placebo_cross", "placebo_random"):
            x = t[(t.scheme == scheme) & (t.arm == arm)].fstep.values
            if len(x) == 0:
                continue
            p, rb, hl = mwu_exact(x, y)
            p_slow, rb_s, _ = mwu_exact(y, x)      # opposite direction (slowdown)
            sec.append(dict(scheme=scheme, arm=arm, n=len(x), mean=x.mean(),
                            p_speed=p, p_slow=p_slow, rb=rb, hl=hl))
            print(f"  {scheme} {arm:22s} n={len(x)} mean={x.mean():7.0f} "
                  f"p_speed={p:.4f} p_slow={p_slow:.4f} rb={rb:+.2f} HL={hl:+.0f}")

    # ---- capability table (assistance must not cost capability)
    print("=" * 78, "\nFINAL CAPABILITY (ce_pred / prev_best / ind_best; free reference)")
    cap = t.groupby(["scheme", "arm"]).agg(
        ce=("final_ce", "mean"), ce_sd=("final_ce", "std"),
        prev=("final_prev", "mean"), ind=("final_ind", "mean"),
        seeded_frac=("prev_head_seeded", "mean")).round(3)
    print(cap.to_string())

    # ---- verdict inputs
    print("=" * 78, "\nVERDICT INPUTS")
    a1 = [r for r, q in zip(prim, qs) if q <= 0.05 and r["rb"] > 0]
    any_speed = bool(a1) or any(
        s["p_speed"] <= 0.05 and s["arm"].startswith(("assist_reg", "assist_init"))
        for s in sec)
    placebo_speed = [s for s in sec if s["arm"].startswith("placebo") and s["p_speed"] <= 0.05]
    print(f"  P-A1 primary significant (q≤0.05): {[r['scheme'] for r in a1] or 'NONE'}")
    print(f"  any assist arm (init or reg) speeds (p≤0.05): {any_speed}")
    print(f"  placebo speedups (P-A2 violation if any): "
          f"{[(s['scheme'], s['arm']) for s in placebo_speed] or 'NONE'}")

    make_figure_m12(t)
    return t


def make_figure_m12(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = ["free", "assist_init_algebra", "assist_init_solution",
            "assist_reg_l1", "assist_reg_l10", "placebo_cross", "placebo_random"]
    labels = ["free", "init:algebra", "init:solution", "reg λ1", "reg λ10",
              "placebo:cross", "placebo:rand"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, scheme in zip(axes, ("rope", "ape")):
        sub = t[t.scheme == scheme]
        for i, arm in enumerate(arms):
            v = sub[sub.arm == arm].fstep.values
            if len(v) == 0:
                continue
            jit = (np.random.default_rng(0).uniform(-0.12, 0.12, len(v)))
            ax.plot(np.full(len(v), i) + jit, v, "o", ms=6, alpha=0.75,
                    color="tab:blue" if "assist" in arm else
                          ("tab:grey" if arm == "free" else "tab:orange"))
            ax.hlines(np.median(v), i - 0.25, i + 0.25, color="k", lw=2)
        fm = sub[sub.arm == "free"].fstep.median()
        ax.axhline(fm, color="grey", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(range(len(arms)), labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{scheme} (free median {fm:.0f})")
        ax.set_ylabel("formation step" if scheme == "rope" else "")
    fig.suptitle("M1.2 assistance hinge — formation steps (6000-step runs, n=5/3 seeds)")
    fig.tight_layout()
    out = FIGS / "m12_formation.png"
    fig.savefig(out, dpi=140)
    print(f"[fig] {out}")


def analyze_m13():
    t12 = load_phase("m12")
    parts = [t12[t12.arm == "free"], load_phase("m13")]
    try:
        parts.append(load_phase("m13ape"))
    except Exception:
        pass
    t = pd.concat(parts)
    print(t[["scheme", "arm", "seed", "fstep", "final_ce", "prev_head", "prev_head_rif",
             "prev_head_dir", "prev_kernel_argmax", "final_prev"]].to_string(index=False))
    # classification (frozen thresholds, ROADMAP P-A3): rope — phase-carried rif≥0.25 vs
    # cos-only rif≤0.10 with kernel peak still Δ=−1; ape — antisym dir≥0.35 vs sym ≤0.10.
    def classify(row):
        if row.scheme == "rope":
            if row.prev_head_rif >= 0.25:
                return "default(phase)"
            if row.prev_head_rif <= 0.10 and row.prev_kernel_argmax == -1:
                return "nondefault(cos)"
        else:
            if row.prev_head_dir >= 0.35:
                return "default(antisym)"
            if row.prev_head_dir <= 0.10:
                return "nondefault(sym)"
        return "unclassified"
    t["impl"] = t.apply(classify, axis=1)
    print("\nimplementation counts:")
    ct = t.groupby(["scheme", "arm", "impl"]).size().unstack(fill_value=0)
    print(ct.to_string())
    # Fisher exact: nondefault vs not, assist vs free (one-sided, assist raises nondefault)
    from scipy.stats import fisher_exact
    for scheme in sorted(t[t.arm != "free"].scheme.unique()):
        free = t[(t.arm == "free") & (t.scheme == scheme)]
        asst = t[(t.arm == "assist_nondefault_l1") & (t.scheme == scheme)]
        if len(asst) == 0:
            continue
        tab = [[int((asst.impl.str.startswith("nondefault")).sum()),
                int((~asst.impl.str.startswith("nondefault")).sum())],
               [int((free.impl.str.startswith("nondefault")).sum()),
                int((~free.impl.str.startswith("nondefault")).sum())]]
        odds, p = fisher_exact(tab, alternative="greater")
        print(f"\n{scheme}: Fisher exact (nondefault | assist vs free): table={tab} p={p:.4f}")
        cost = asst.fstep.mean() / free.fstep.mean() - 1
        print(f"  formation-time cost of assisted selection: {cost:+.1%} (P-A3 bound ≤ +10%)")
        print(f"  capability: assist ce {asst.final_ce.mean():.3f} vs free {free.final_ce.mean():.3f}; "
              f"assist prev {asst.final_prev.mean():.3f} vs free {free.final_prev.mean():.3f}")
    t.to_parquet(CFG.RESULTS / "h1" / "m13_summary.parquet")


if __name__ == "__main__":
    {"m12": analyze_m12, "m13": analyze_m13}[sys.argv[1] if len(sys.argv) > 1 else "m12"]()
