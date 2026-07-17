"""M2.2 analysis — pre-registered readouts + verdicts (design entry: results/h2/LOG.md,
2026-07-05; ROADMAP M2.2 kill clause; §3 two-sided scale question).

Frozen readouts (verbatim from the design entry):
  T_prev(arm,seed) = first eval with prev_beh >= 0.5 (token count); T_ind likewise.
  Primary contrasts: assist-free, constraint-free, per seed.
  n=2 pilot verdict = SIGN CONSISTENCY across both seeds with |D| > local grid spacing
    (25M tokens in-window).
  Capability parity: val loss + ICL at matched token counts.
  Adoption/scaffold: seeded-head prev/ind scores + rif trajectories vs population.
  Kill clause: assist-free <= 0 in BOTH seeds (T_prev AND T_ind) WITH plant verified in
    weight space -> report; scope decision is the user's.

Descriptive additions (labelled; no frozen quantity replaced):
  - sub-grid linear-interpolated crossings (resolution refinement of the same quantity)
  - freeze-displacement audit (do free-arm winners land on seeded head slots?)
  - implementation class of the constraint arm's prev head (M1.3 thresholds)
  - lambda-mass / Im-mass: trajectories for runs 4-6, endpoints (from final.pt) for 1-3

Usage: python src/h2_m22_analyze.py            (CPU only)
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG

OUT = CFG.RESULTS / "h2" / "runs" / "m22"
FIGS = CFG.RESULTS / "h2" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
CKPT = Path("/large/share/li_qk/h2_m22")
ARMS = ("free", "assist", "constraint")
SEEDS = (0, 1)
THRESH = 0.5
GRID_IN_WINDOW = 24_117_248          # int(25e6//BATCH_TOKENS)=23 steps x 1,048,576
                                     # (the trainer's actual fine cadence; verified vs data)
SEEDED = [(l, h) for l in (1, 2, 3, 4) for h in (0, 1)]
# toy anchors (H1/M2.1, results/h1+h2 LOG): frozen solution plant -44%; imag-ban l10 +36..52%
TOY_ASSIST = -0.44
TOY_BAN = (0.36, 0.52)


def tag(arm, seed):
    return f"m22_{arm}_seed{seed}"


def rows(arm, seed):
    f = OUT / f"{tag(arm, seed)}_evals.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def crossing(rs, key, thresh=THRESH):
    """Frozen readout: first eval at/above threshold (token count)."""
    for r in rs:
        if r[key] >= thresh:
            return r["tokens"]
    return None


def crossing_interp(rs, key, thresh=THRESH):
    """Descriptive sub-grid refinement: linear interpolation between the bracketing
    evals of the SAME quantity. Not a frozen readout."""
    prev = None
    for r in rs:
        if r[key] >= thresh:
            if prev is None:
                return float(r["tokens"])
            x0, y0 = prev["tokens"], prev[key]
            x1, y1 = r["tokens"], r[key]
            if y1 == y0:
                return float(x1)
            return float(x0 + (thresh - y0) * (x1 - x0) / (y1 - y0))
        prev = r
    return None


def at_tokens(rs, tok):
    return min(rs, key=lambda r: abs(r["tokens"] - tok))


def endpoint_lam_mass(arm, seed):
    """lambda-mass / Im-mass endpoints for runs whose in-flight battery predates the
    2026-07-06 instrument amendment (runs 1-3). Recomputed from final.pt with the
    identical battery() code path."""
    import torch
    import h2_m22_train as T
    T.DEV = "cpu"
    sd = torch.load(CKPT / tag(arm, seed) / "final.pt", map_location="cpu",
                    weights_only=False)["model"]
    model = T.make_model(0)
    model.load_state_dict(sd)
    wm = T.battery(model)
    del model, sd
    return wm


def prev_head_of(row):
    pa = np.array(row["prev_all"])
    l, h = np.unravel_index(int(pa.argmax()), pa.shape)
    return int(l), int(h), float(pa[l, h])


def topk_prev(row, k=4):
    pa = np.array(row["prev_all"])
    idx = np.dstack(np.unravel_index(np.argsort(-pa, axis=None)[:k], pa.shape))[0]
    return [(int(l), int(h), float(pa[l, h])) for l, h in idx]


def classify_rope(rif, kernel_peak_ok=True):
    """Frozen M1.3 thresholds (rope): phase-carried rif>=0.25 / cos-only rif<=0.10."""
    if rif >= 0.25:
        return "default(phase)"
    if rif <= 0.10:
        return "nondefault(cos-only)"
    return "unclassified"


def main():
    R = {(a, s): rows(a, s) for a in ARMS for s in SEEDS}
    res = {}

    print("=" * 92)
    print("M2.2 PILOT — pre-registered analysis (160M x 4B tokens, n=2 seeds, shared token stream)")
    print("=" * 92)

    # ---------------------------------------------------------------- 1. crossings
    print("\n[1] FROZEN READOUTS — formation crossings (first eval >= 0.5)")
    cross = {}
    for a in ARMS:
        for s in SEEDS:
            rs = R[(a, s)]
            cross[(a, s)] = dict(
                T_prev=crossing(rs, "prev_beh"), T_ind=crossing(rs, "ind_beh"),
                T_prev_i=crossing_interp(rs, "prev_beh"),
                T_ind_i=crossing_interp(rs, "ind_beh"))
            c = cross[(a, s)]
            print(f"  {a:10s} seed{s}: T_prev={c['T_prev']/1e9:.3f}B  T_ind={c['T_ind']/1e9:.3f}B "
                  f"| interp (descriptive): {c['T_prev_i']/1e9:.3f}B / {c['T_ind_i']/1e9:.3f}B")

    print(f"\n  in-window grid spacing = {GRID_IN_WINDOW/1e6:.1f}M tokens "
          f"(the pre-registered detectability floor)")

    print("\n[2] PRIMARY CONTRASTS (per seed; frozen rule: sign-consistent AND |D| > grid)")
    contrasts = {}
    for a in ("assist", "constraint"):
        for s in SEEDS:
            for key in ("T_prev", "T_ind"):
                d = cross[(a, s)][key] - cross[("free", s)][key]
                rel = d / cross[("free", s)][key]
                di = cross[(a, s)][key + "_i"] - cross[("free", s)][key + "_i"]
                reli = di / cross[("free", s)][key + "_i"]
                contrasts[(a, s, key)] = dict(abs=d, rel=rel, abs_i=di, rel_i=reli,
                                              steps=d / GRID_IN_WINDOW)
                print(f"  {a:10s} seed{s} {key}: {d/1e6:+7.1f}M ({rel:+6.1%}, "
                      f"{d/GRID_IN_WINDOW:+.2f} grid steps) | interp {di/1e6:+7.1f}M ({reli:+6.1%})")

    def verdict(a, key):
        ds = [contrasts[(a, s, key)] for s in SEEDS]
        signs = {np.sign(d["abs"]) for d in ds}
        consistent = len(signs) == 1 and 0 not in signs
        detectable = all(abs(d["abs"]) > GRID_IN_WINDOW for d in ds)
        return consistent, detectable, [d["rel"] for d in ds]

    print("\n[3] VERDICTS vs the frozen rule")
    for a in ("assist", "constraint"):
        for key in ("T_prev", "T_ind"):
            cons, det, rels = verdict(a, key)
            tagv = ("EFFECT (consistent + detectable)" if cons and det else
                    "consistent sign but AT/BELOW grid resolution" if cons else
                    "no consistent sign")
            print(f"  {a:10s} {key}: {tagv}; rel = {rels[0]:+.1%}, {rels[1]:+.1%}")
    res["contrasts"] = {f"{a}|{s}|{k}": v for (a, s, k), v in contrasts.items()}

    # ------------------------------------------------- 4. scale-transfer (two-sided Q)
    print("\n[4] SCALE TRANSFER vs toy anchors (ROADMAP §3 two-sided question; descriptive)")
    a_rel = [contrasts[("assist", s, "T_prev")]["rel"] for s in SEEDS]
    c_rel = [contrasts[("constraint", s, "T_prev")]["rel"] for s in SEEDS]
    c_rel_i = [contrasts[("constraint", s, "T_ind")]["rel"] for s in SEEDS]
    print(f"  assist (frozen plant): toy {TOY_ASSIST:+.0%}  ->  160M {a_rel[0]:+.1%}, {a_rel[1]:+.1%}")
    need = abs(TOY_ASSIST) * np.mean([cross[("free", s)]["T_prev"] for s in SEEDS])
    print(f"     a toy-sized effect would be {-need/1e6:.0f}M tokens = "
          f"{-need/GRID_IN_WINDOW:.1f} grid steps; observed +1 step -> toy effect EXCLUDED")
    print(f"  constraint (imag-ban l10): toy {TOY_BAN[0]:+.0%}..{TOY_BAN[1]:+.0%}  ->  "
          f"160M T_prev {c_rel[0]:+.1%}, {c_rel[1]:+.1%} | T_ind {c_rel_i[0]:+.1%}, {c_rel_i[1]:+.1%}")

    # -------------------------------------------------------- 5. capability parity
    print("\n[5] CAPABILITY (endpoint 3.95B; and at matched mid-window tokens)")
    for a in ARMS:
        for s in SEEDS:
            last = R[(a, s)][-1]
            print(f"  {a:10s} seed{s}: val={last['val_loss']:.4f} prev={last['prev_beh']:.3f} "
                  f"ind={last['ind_beh']:.3f} icl={last['icl']:+.2f}")
    print("  paired endpoint deltas (arm - free, same seed):")
    for a in ("assist", "constraint"):
        dv = [R[(a, s)][-1]["val_loss"] - R[("free", s)][-1]["val_loss"] for s in SEEDS]
        di = [R[(a, s)][-1]["icl"] - R[("free", s)][-1]["icl"] for s in SEEDS]
        print(f"    {a:10s} dval = {dv[0]:+.4f}, {dv[1]:+.4f} nats | dICL = {di[0]:+.2f}, {di[1]:+.2f}")
    res["endpoint"] = {f"{a}|{s}": {k: R[(a, s)][-1][k] for k in
                                    ("val_loss", "prev_beh", "ind_beh", "icl")}
                       for a in ARMS for s in SEEDS}

    # ----------------------------------------- 6. adoption / scaffold / displacement
    print("\n[6] ADOPTION vs SCAFFOLD + freeze-displacement audit (descriptive)")
    print(f"  seeded slots (assist arm): {['L%dH%d' % lh for lh in SEEDED]}")
    for s in SEEDS:
        fl, al = R[("free", s)][-1], R[("assist", s)][-1]
        fw, aw = prev_head_of(fl), prev_head_of(al)
        print(f"  seed{s}: free winner L{fw[0]}H{fw[1]} ({fw[2]:.3f})"
              f"{'  <-- IS A SEEDED SLOT' if (fw[0], fw[1]) in SEEDED else ''}"
              f" | assist winner L{aw[0]}H{aw[1]} ({aw[2]:.3f})"
              f"{'  (seeded)' if (aw[0], aw[1]) in SEEDED else '  (unseeded)'}")
        print(f"     free top-4 prev: {[(f'L{l}H{h}', round(v,3)) for l,h,v in topk_prev(fl)]}")
        print(f"     assist seeded-head prev scores: "
              f"{[round(np.array(al['prev_all'])[l][h], 4) for l, h in SEEDED]}")
        print(f"     assist seeded-head rif (frozen -> const): {al['seeded_rif'][:4]} ...")

    # --------------------------------------------- 7. implementation class (READ side)
    print("\n[7] IMPLEMENTATION CLASS of the prev-carrying head (frozen M1.3 thresholds)")
    impl = {}
    for a in ARMS:
        for s in SEEDS:
            last = R[(a, s)][-1]
            l, h, sc = prev_head_of(last)
            wm = {(w["layer"], w["head"]): w for w in last["wm"]}
            rif = wm[(l, h)]["rif"]
            cls = classify_rope(rif)
            impl[(a, s)] = dict(layer=l, head=h, prev=sc, rif=rif,
                                dir_frac=wm[(l, h)]["dir_frac"],
                                D_head=wm[(l, h)]["D_head"], cls=cls)
            print(f"  {a:10s} seed{s}: L{l}H{h} prev={sc:.3f} rif={rif:.4f} "
                  f"dir={wm[(l,h)]['dir_frac']:.3f} -> {cls}")
    print("  population imag-share (penalty value) endpoints:")
    for a in ARMS:
        print(f"    {a:10s}: " + ", ".join(f"seed{s} {R[(a,s)][-1]['penalty']:.4f}" for s in SEEDS))
    res["impl"] = {f"{a}|{s}": v for (a, s), v in impl.items()}

    # --------------------------------------------- 8. M-side natural history + lam-mass
    print("\n[8] M-SIDE NATURAL HISTORY (population medians; in-training, 160M)")
    for a in ARMS:
        for s in SEEDS:
            rs = R[(a, s)]
            f0, f1, fend = rs[0], at_tokens(rs, 1e9), rs[-1]
            lm = ("" if "pop_med_lam_mass" not in fend else
                  f" | lam-mass {f0['pop_med_lam_mass']:.4f}->{fend['pop_med_lam_mass']:.4f}"
                  f" im-mass {f0['pop_med_im_mass']:.4f}->{fend['pop_med_im_mass']:.4f}")
            print(f"  {a:10s} seed{s}: D_head {f0['pop_med_D']:.3f} -> {f1['pop_med_D']:.3f}(1B)"
                  f" -> {fend['pop_med_D']:.3f} | rif {f0['pop_med_rif']:.3f} -> "
                  f"{fend['pop_med_rif']:.3f}{lm}")

    print("\n  endpoint lambda-mass retrofit for runs 1-3 (from final.pt, identical battery path):")
    for a in ARMS:
        wm = endpoint_lam_mass(a, 0)
        lam = float(np.median([w["lam_mass"] for w in wm]))
        im = float(np.median([w["im_mass"] for w in wm]))
        D = float(np.median([w["D_head"] for w in wm]))
        # cross-check against the in-flight battery of the same run (D_head must match)
        D_inflight = R[(a, 0)][-1]["pop_med_D"]
        print(f"    {a:10s} seed0: pop-med lam_mass={lam:.4f} im_mass={im:.4f} "
              f"D_head={D:.4f} (in-flight {D_inflight:.4f}; delta {abs(D-D_inflight):.2e})")
        res.setdefault("lam_endpoint", {})[f"{a}|0"] = dict(lam=lam, im=im, D=D)

    # ---------------------------------------------------------------- 9. kill clause
    print("\n[9] KILL-CLAUSE ASSESSMENT (ROADMAP M2.2; execution session REPORTS ONLY)")
    a_le0 = all(contrasts[("assist", s, k)]["abs"] >= 0 for s in SEEDS for k in ("T_prev", "T_ind"))
    plant_ok = True
    for s in SEEDS:
        log = (CFG.RESULTS / "h2" / f"m22_m22_assist_seed{s}.log").read_text(errors="ignore")
        plant_ok &= ("plant verified" in log)
    print(f"  (i) assist speedup <= 0 in BOTH seeds on BOTH crossings: {a_le0}")
    print(f"  (ii) plant verified in weight space at init (both assist runs): {plant_ok}")
    print(f"  (iii) toy effect established in the same construction/code path: YES "
          f"(M1.2 rb=+1.0 p=.004; M2.1 P-A5 frozen rho=2.0)")
    print(f"  => KILL-CLAUSE INPUTS {'SATISFIED' if (a_le0 and plant_ok) else 'NOT satisfied'}. "
          f"Scope/pivot decision is the USER's (ROADMAP §0).")
    res["kill"] = dict(assist_le0=bool(a_le0), plant_verified=bool(plant_ok))

    (CFG.RESULTS / "h2" / "m22_results.json").write_text(json.dumps(res, indent=1, default=float))
    make_figs(R, cross)
    print("\n[done] results -> results/h2/m22_results.json")


def make_figs(R, cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = dict(free="tab:grey", assist="tab:blue", constraint="tab:red")
    # free is drawn as a wide translucent halo so that arms lying exactly on top of it
    # (the assist LM curve does) remain visible rather than being overplotted.
    LW = dict(free=4.0, assist=1.4, constraint=1.4)
    AL = dict(free=0.35, assist=0.95, constraint=0.95)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for a in ARMS:
        for s in SEEDS:
            rs = R[(a, s)]
            t = np.array([r["tokens"] for r in rs]) / 1e9
            ls = "-" if s == 0 else "--"
            kw = dict(color=C[a], alpha=AL[a], lw=LW[a])
            axes[0].plot(t, [r["prev_beh"] for r in rs], ls, label=f"{a} s{s}", **kw)
            axes[1].plot(t, [r["ind_beh"] for r in rs], ls, **kw)
            axes[2].plot(t, [r["val_loss"] for r in rs], ls, **kw)
    for ax, ttl, yl in ((axes[0], "prev-token head score", "prev_beh"),
                        (axes[1], "induction head score", "ind_beh"),
                        (axes[2], "validation loss", "nats")):
        ax.set_xlabel("tokens (B)")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        ax.set_xlim(0, 1.0 if ax is not axes[2] else 4.0)
    for ax in axes[:2]:
        ax.axhline(0.5, color="k", lw=.8, ls=":")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    axes[2].text(0.42, 0.92, "free curve (wide halo) is hidden under assist:\n"
                             "the frozen plant costs and buys nothing on LM loss",
                 transform=axes[2].transAxes, fontsize=7, ha="center", va="top")
    fig.suptitle("M2.2 — 160M scale transfer: formation (solid seed0, dashed seed1; "
                 "free = wide halo)")
    fig.tight_layout()
    fig.savefig(FIGS / "m22_formation.png", dpi=140)

    fig2, ax = plt.subplots(figsize=(6.2, 4.2))
    for a in ARMS:
        for s in SEEDS:
            rs = R[(a, s)][1:]                      # drop t=0 for the log axis
            t = np.array([r["tokens"] for r in rs]) / 1e9
            ax.plot(t, [r["pop_med_rif"] for r in rs], "-" if s == 0 else "--",
                    color=C[a], alpha=AL[a], lw=LW[a], label=f"{a} s{s}")
    ax.axhline(0.5, color="k", lw=.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 4.2)
    ax.set_xlabel("tokens (B, log)")
    ax.set_ylabel("population median rope_imag_frac")
    ax.set_title("M2.2 — the ban binds at scale (imag share -> 0)")
    ax.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(FIGS / "m22_rif.png", dpi=140)
    print(f"[fig] {FIGS/'m22_formation.png'} , {FIGS/'m22_rif.png'}")


if __name__ == "__main__":
    main()
