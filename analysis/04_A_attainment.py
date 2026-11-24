"""A: label-free kinematic difficulty from a-priori angle targets, compared with Rasch difficulty.

Usage: python -I 04_A_attainment.py <extracted/REHAB dir> <out dir>

Targets were fixed BEFORE looking at item-level results (2026-11-23):
  * FMA definitions in the dataset's Table 2 / FMA-UE manual: shoulder abduction 90 deg (002),
    shoulder flexion 90 deg (010), wrist dorsiflexion 15 deg (015, 017; FMA wrist "stability").
  * Otherwise normative active ROM (AAOS, Greene & Heckman 1994):
    shoulder ER 90, IR 70, elbow flexion 150, pronation 80, supination 80,
    pronation+supination arc 160, wrist flexion+extension arc 150 (80+70).
Items without a single-joint angle target are excluded from A:
  000 elevation, 001 retraction (scapular, not measured by arm IMUs), 009 hand to lumbar (functional),
  019 circumduction, 020-025 hand/grasp (glove finger calibration unknown), 026 finger-to-nose (timing).

Measurement: raw 50 Hz signal, 5-sample median filter, unwrap, full range (max - min).
Axis: within the a-priori sensor, the axis with the largest median range across all 222 assessments
for that movement (label-free choice).
Flat (constant) signals = no movement -> ROM 0 in the main analysis; excluded in a sensitivity run.
Signals shorter than 0.5 s are excluded.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import spearmanr, kendalltau
from girth import pcm_mml

FS = 50.0
# item: (source, target_deg, basis)
TARGETS = {
    2: ("S2", 90, "FMA def"), 3: ("S1", 90, "AAOS ER"), 4: ("elbow", 150, "AAOS"),
    5: ("S1", 80, "AAOS sup"), 6: ("S1", 70, "AAOS IR"), 7: ("elbow", 150, "AAOS"),
    8: ("S1", 80, "AAOS pro"), 10: ("S2", 90, "FMA def"), 11: ("S1", 160, "AAOS arc"),
    12: ("S1", 80, "AAOS pro"), 13: ("S1", 160, "AAOS arc"), 14: ("S1", 160, "AAOS arc"),
    15: ("glove", 15, "FMA def"), 16: ("glove", 150, "AAOS arc"), 17: ("glove", 15, "FMA def"),
    18: ("glove", 150, "AAOS arc"),
}
NAMES = {2: "sh abduction", 3: "sh ext rot", 4: "elbow flex", 5: "supination", 6: "sh add/IR",
         7: "elbow ext", 8: "pronation", 10: "sh flex 90", 11: "pro/sup elbow90",
         12: "pronation abd90", 13: "pro/sup flex90-180", 14: "pro/sup flex30-90",
         15: "wrist dorsi e90", 16: "wrist flex/ext e90", 17: "wrist dorsi e0", 18: "wrist flex/ext e0"}


def full_range(x):
    if len(x) < 5:
        return np.nan
    y = medfilt(np.degrees(np.unwrap(np.radians(x))), 5)
    return float(y.max() - y.min())


def candidate_ranges(f, m, src):
    """Return dict axis -> range, flat flag, duration."""
    if src in ("S1", "S2", "elbow"):
        s1 = np.load(f / f"{m:03d}_1.npy", allow_pickle=False)
        s2 = np.load(f / f"{m:03d}_2.npy", allow_pickle=False)
        dur = max(len(s1), len(s2)) / FS
        if src == "S1":
            x, cols = s1, {"pitch": 0, "yaw": 1, "roll": 2}
        elif src == "S2":
            x, cols = s2, {"pitch": 1, "yaw": 2, "roll": 3}
        else:
            x, cols = s2, {"elbow": 0}
    else:
        x = np.load(f / f"{m:03d}_3.npy", allow_pickle=False)
        dur = len(x) / FS
        cols = {"g0": 0, "g1": 1}
    flat = bool(np.all(np.std(x[:, list(cols.values())], axis=0) < 1e-3))
    return {a: (0.0 if flat else full_range(x[:, c])) for a, c in cols.items()}, flat, dur


def main():
    root = Path(sys.argv[1]) / "Rehab_assessment"
    out = Path(sys.argv[2])
    lab = np.load(root / "Label.npy", allow_pickle=False)
    R = root / "d01_raw_data"

    recs = []
    for k in range(1, 223):
        for m, (src, tgt, basis) in TARGETS.items():
            rngs, flat, dur = candidate_ranges(R / f"{k:03d}", m, src)
            for a, v in rngs.items():
                recs.append(dict(assess=k, move=m, axis=a, rom=v, flat=flat, dur=dur,
                                 score=int(lab[k - 1, m])))
    long = pd.DataFrame(recs)
    long = long[long.dur >= 0.5]
    # label-free axis choice: largest median range across all 222 assessments
    axis = long.groupby(["move", "axis"]).rom.median().unstack().idxmax(axis=1)
    sel = long[long.apply(lambda r: r.axis == axis[r.move], axis=1)].copy()
    sel["target"] = sel.move.map(lambda m: TARGETS[m][1])
    sel.to_csv(out / "A_rom_by_assessment.csv", index=False, encoding="utf-8-sig")

    items = sorted(TARGETS)
    base = sel[sel.assess <= 120]

    def kin_difficulty(df, scale=1.0, mode="attain", drop_flat=False):
        d = df[~df.flat] if drop_flat else df
        out_ = []
        for m in items:
            r = d[d.move == m].rom.values / (TARGETS[m][1] * scale)
            out_.append(1 - np.mean(np.clip(r, 0, 1)) if mode == "attain" else 1 - np.mean(r >= 0.9))
        return np.array(out_)

    def rasch_difficulty(L):
        res = pcm_mml(L.T.astype(int))
        return np.nanmean(np.asarray(res["Difficulty"]), axis=1)

    L120 = lab[:120]
    rasch_all = rasch_difficulty(L120)
    rasch = rasch_all[items]

    table = pd.DataFrame(dict(item=[f"{m:03d}" for m in items], name=[NAMES[m] for m in items],
                              sensor=[TARGETS[m][0] for m in items], axis=[axis[m] for m in items],
                              target=[TARGETS[m][1] for m in items], basis=[TARGETS[m][2] for m in items],
                              median_rom=[round(base[base.move == m].rom.median(), 1) for m in items],
                              rasch=rasch.round(2)))
    table["kin_attain"] = kin_difficulty(base).round(3)
    table["kin_pass90"] = kin_difficulty(base, mode="pass").round(3)
    table["rasch_rank"] = table.rasch.rank().astype(int)
    table["kin_rank"] = table.kin_attain.rank().astype(int)
    table.to_csv(out / "A_attainment_items.csv", index=False, encoding="utf-8-sig")
    print(table.sort_values("rasch").to_string(index=False))

    print("\nSpearman / Kendall, Rasch vs kinematic difficulty (16 items, folders 001-120)")
    variants = {
        "main: attainment, targets x1.0": kin_difficulty(base),
        "targets x0.8": kin_difficulty(base, 0.8),
        "targets x1.2": kin_difficulty(base, 1.2),
        "pass rate (>=90% of target)": kin_difficulty(base, mode="pass"),
        "flat signals excluded": kin_difficulty(base, drop_flat=True),
        "all 222 assessments": kin_difficulty(sel),
    }
    for name, kd in variants.items():
        print(f"  {name:32s} rho {spearmanr(rasch, kd).statistic:+.2f}  tau {kendalltau(rasch, kd).statistic:+.2f}")
    for lbl, sub in [("IMU items (002-014)", [i for i, m in enumerate(items) if m <= 14]),
                     ("glove items (015-018)", [i for i, m in enumerate(items) if m >= 15])]:
        kd = variants["main: attainment, targets x1.0"]
        print(f"  within {lbl:24s} rho {spearmanr(rasch[sub], kd[sub]).statistic:+.2f}")

    # bootstrap over assessments (resample folders 001-120, recompute both difficulties)
    rng = np.random.default_rng(20261123)
    B, rhos = 300, []
    by_assess = {k: g for k, g in base.groupby("assess")}
    keys = np.array(sorted(by_assess))
    for _ in range(B):
        idx = rng.choice(keys, size=len(keys), replace=True)
        bd = pd.concat([by_assess[k] for k in idx])
        try:
            rb = rasch_difficulty(L120[idx - 1])[items]
        except Exception:
            continue
        rhos.append(spearmanr(rb, kin_difficulty(bd)).statistic)
    rhos = np.array(rhos)
    print(f"\nbootstrap ({len(rhos)} reps): rho median {np.median(rhos):+.2f}, "
          f"95% CI [{np.percentile(rhos, 2.5):+.2f}, {np.percentile(rhos, 97.5):+.2f}]")

    # theory order (FMA subscale order) vs Rasch and vs kinematics
    theory = np.array([{2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 10: 3, 11: 3,
                        12: 4, 13: 4, 14: 4, 15: 5, 16: 5, 17: 5, 18: 5}[m] for m in items])
    print("theory subscale order vs Rasch: tau "
          f"{kendalltau(theory, rasch).statistic:+.2f}; vs kinematic: tau "
          f"{kendalltau(theory, variants['main: attainment, targets x1.0']).statistic:+.2f}")


if __name__ == "__main__":
    main()
