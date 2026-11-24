"""A (confirmed design): posture-dependence of identical joint motions, within patient.

Usage: python -I 06_A_paired_robust.py <extracted/REHAB dir> <out dir>

Pre-specified (2026-11-23, after the exploratory run 05_A_paired.py):
  * 8 pairs, easier -> harder by FMA theory (see PAIRS); primary hypothesis: kinematic magnitude
    decreases within patient for the harder posture.
  * Main metric M1 = full range of the a-priori channel (raw 50 Hz, median filter 5, unwrap), capped 180.
  * Main sample = folders 001-120 (assumed one assessment per patient).
  * Test: Wilcoxon signed-rank, Holm correction across the 8 pairs; effect size = matched-pairs
    rank-biserial r (negative = smaller in harder posture).
  * Exclusions: signal < 25 samples; pairs where both values are 0 (no movement in either).
Sensitivity: M2 robust range (p5-p95), M3 peak angular speed (p95), M4 processed data (d02) full range;
  replication sample folders 121-222 (assumed discharge assessments).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import wilcoxon, rankdata

FS = 50.0
CH = {5: ("S1", 2), 8: ("S1", 2), 11: ("S1", 2), 12: ("S1", 2), 13: ("S1", 2), 14: ("S1", 2),
      15: ("glove", 1), 16: ("glove", 1), 17: ("glove", 1), 18: ("glove", 1)}
PAIRS = [(5, 11), (8, 11), (11, 14), (11, 12), (11, 13), (14, 13), (15, 17), (16, 18)]
LABEL = {5: "supination (syn)", 8: "pronation (syn)", 11: "pro/sup sh0 e90", 14: "pro/sup flex30-90 e0",
         12: "pronation abd90 e0", 13: "pro/sup flex90-180 e0", 15: "dorsi e90", 17: "dorsi e0",
         16: "flex/ext e90", 18: "flex/ext e0"}


def ang(x):
    return medfilt(np.degrees(np.unwrap(np.radians(x))), 5)


def metrics(x, src, col, processed):
    if len(x) < 25:
        return dict(M1=np.nan, M2=np.nan, M3=np.nan)
    if np.std(x[:, col]) < 1e-3:
        return dict(M1=0.0, M2=0.0, M3=0.0)
    a = ang(x[:, col])
    m1 = min(float(a.max() - a.min()), 180.0)
    m2 = min(float(np.percentile(a, 95) - np.percentile(a, 5)), 180.0)
    if processed:
        return dict(M1=m1, M2=m2, M3=np.nan)
    if src == "S1":
        w = np.linalg.norm(x[:, 9:12], axis=1)  # deg/s
    else:
        w = np.abs(np.diff(a)) * FS
    return dict(M1=m1, M2=m2, M3=float(np.percentile(w, 95)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def rank_biserial(a, b):
    d = np.asarray(b) - np.asarray(a)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main():
    root = Path(sys.argv[1]) / "Rehab_assessment"
    out = Path(sys.argv[2])
    rows = []
    for sub, processed in [("d01_raw_data", False), ("d02_processed_data", True)]:
        for k in range(1, 223):
            for m, (src, col) in CH.items():
                f = root / sub / f"{k:03d}" / f"{m:03d}_{1 if src == 'S1' else 3}.npy"
                if not f.exists():  # folder 195 has an extra dated subfolder in d02
                    f = next((root / sub / f"{k:03d}").rglob(f.name))
                x = np.load(f, allow_pickle=False)
                for name, v in metrics(x, src, col, processed).items():
                    key = "M4" if (processed and name == "M1") else (None if processed else name)
                    if key:
                        rows.append(dict(assess=k, move=m, metric=key, value=v))
    long = pd.DataFrame(rows)
    long.to_csv(out / "A_paired_metrics_long.csv", index=False, encoding="utf-8-sig")

    results = []
    for metric in ["M1", "M2", "M3", "M4"]:
        for sample, lo, hi in [("001-120", 1, 120), ("121-222", 121, 222)]:
            W = long[(long.metric == metric) & long.assess.between(lo, hi)].pivot(
                index="assess", columns="move", values="value")
            ps, block = [], []
            for a, b in PAIRS:
                ok = W[[a, b]].dropna()
                ok = ok[(ok[a] > 0) | (ok[b] > 0)]
                p = wilcoxon(ok[a], ok[b]).pvalue
                ps.append(p)
                block.append(dict(metric=metric, sample=sample, pair=f"{a:03d}->{b:03d}",
                                  label=f"{LABEL[a]} -> {LABEL[b]}", n=len(ok),
                                  med_a=round(ok[a].median(), 1), med_b=round(ok[b].median(), 1),
                                  frac_drop=round(float((ok[b] < ok[a]).mean()), 2),
                                  r_rb=round(rank_biserial(ok[a], ok[b]), 2), p=p))
            for row, padj in zip(block, holm(ps)):
                row["p_holm"] = padj
                row["sig"] = "*" if padj < 0.05 else ""
            results += block
    R = pd.DataFrame(results)
    R.to_csv(out / "A_paired_robust.csv", index=False, encoding="utf-8-sig")

    show = R.copy()
    show["p"] = show.p.map(lambda v: f"{v:.2g}")
    show["p_holm"] = show.p_holm.map(lambda v: f"{v:.2g}")
    main_ = show[(show.metric == "M1") & (show["sample"] == "001-120")]
    print("MAIN (M1 full range, folders 001-120, Holm over 8 pairs)")
    print(main_.drop(columns=["metric", "sample"]).to_string(index=False))

    print("\nSENSITIVITY: rank-biserial r (Holm-significant marked *)")
    piv = R.assign(cell=R.r_rb.map(lambda v: f"{v:+.2f}") + R.sig).pivot_table(
        index="pair", columns=["sample", "metric"], values="cell", aggfunc="first")
    print(piv.to_string())

    # gimbal check for 013: forearm pitch near +-90 makes roll unstable
    print("\nmedian |S1 pitch| during forearm items (deg), raw, folders 001-120:")
    for m in [5, 8, 11, 12, 13, 14]:
        vals = []
        for k in range(1, 121):
            x = np.load(root / "d01_raw_data" / f"{k:03d}" / f"{m:03d}_1.npy", allow_pickle=False)
            if len(x) >= 25 and np.std(x[:, 0]) > 1e-3:
                vals.append(np.median(np.abs(x[:, 0])))
        v = np.array(vals)
        print(f"  {m:03d} {LABEL[m]:24s} median {np.median(v):5.1f}  share>70deg {np.mean(v > 70):.2f}")


if __name__ == "__main__":
    main()
