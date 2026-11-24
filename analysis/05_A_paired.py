"""A (revised): within-patient comparison of the SAME joint motion measured by the SAME sensor axis
under different postures, as FMA theory orders them. Sensor calibration cancels within patient.

Usage: python -I 05_A_paired.py <A_rom_by_assessment.csv dir/extracted REHAB> <out dir>

Families (channel fixed a priori, not chosen from results):
  forearm rotation (S1 roll): 005 supination (flexor synergy), 008 pronation (extensor synergy),
      011 pro/sup sh0 e90 (mixed), 014 pro/sup flex30-90 e0, 012 pronation abd90 e0,
      013 pro/sup flex90-180 e0 (isolated)
  wrist dorsiflexion (glove g1): 015 elbow 90 -> 017 elbow 0
  wrist flexion/extension (glove g1): 016 elbow 90 -> 018 elbow 0
FMA theory: synergy < mixed < isolated in difficulty; elbow extended harder than elbow flexed.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import wilcoxon, spearmanr, kendalltau
from girth import pcm_mml

FS = 50.0
FAMILIES = {
    "forearm rotation (S1 roll)": ("S1", 2, [5, 8, 11, 14, 12, 13]),
    "wrist dorsiflexion (glove)": ("glove", 1, [15, 17]),
    "wrist flex/ext (glove)": ("glove", 1, [16, 18]),
}
THEORY = {5: 1, 8: 1, 11: 2, 14: 3, 12: 3, 13: 3, 15: 1, 17: 2, 16: 1, 18: 2}
NAMES = {5: "supination (syn)", 8: "pronation (syn)", 11: "pro/sup sh0 e90", 14: "pro/sup flex30-90 e0",
         12: "pronation abd90 e0", 13: "pro/sup flex90-180 e0", 15: "dorsi e90", 17: "dorsi e0",
         16: "flex/ext e90", 18: "flex/ext e0"}


def full_range(x):
    if len(x) < 5:
        return np.nan
    y = medfilt(np.degrees(np.unwrap(np.radians(x))), 5)
    return float(y.max() - y.min())


def main():
    root = Path(sys.argv[1]) / "Rehab_assessment"
    out = Path(sys.argv[2])
    lab = np.load(root / "Label.npy", allow_pickle=False)
    R = root / "d01_raw_data"
    res = pcm_mml(lab[:120].T.astype(int))
    rasch = np.nanmean(np.asarray(res["Difficulty"]), axis=1)

    rows = []
    for k in range(1, 121):
        for fam, (src, col, items) in FAMILIES.items():
            for m in items:
                x = np.load(R / f"{k:03d}" / f"{m:03d}_{1 if src == 'S1' else 3}.npy", allow_pickle=False)
                flat = bool(np.std(x[:, col]) < 1e-3)
                rom = np.nan if len(x) < 25 else (0.0 if flat else full_range(x[:, col]))
                rows.append(dict(assess=k, family=fam, move=m, rom=rom, score=int(lab[k - 1, m])))
    d = pd.DataFrame(rows)
    d.to_csv(out / "A_paired_rom.csv", index=False, encoding="utf-8-sig")
    W = d.pivot_table(index="assess", columns="move", values="rom")

    print("Item summary (folders 001-120): median ROM, Rasch difficulty, theory level")
    summ = []
    for fam, (_, _, items) in FAMILIES.items():
        for m in items:
            summ.append(dict(family=fam, item=f"{m:03d}", name=NAMES[m], theory=THEORY[m],
                             median_rom=round(W[m].median(), 1), mean_score=round(lab[:120, m].mean(), 2),
                             rasch=round(rasch[m], 2)))
    S = pd.DataFrame(summ)
    print(S.to_string(index=False))
    S.to_csv(out / "A_paired_items.csv", index=False, encoding="utf-8-sig")

    print("\nPairwise within-patient tests (easier -> harder by FMA theory)")
    pairs = [(5, 11), (8, 11), (11, 14), (11, 12), (11, 13), (14, 13), (15, 17), (16, 18)]
    pr = []
    for a, b in pairs:
        ok = W[[a, b]].dropna()
        ok = ok[(ok[a] > 0) | (ok[b] > 0)]  # drop patients with no movement on both
        diff = ok[b] - ok[a]
        p = wilcoxon(ok[a], ok[b]).pvalue if len(ok) > 10 else np.nan
        sa, sb = lab[ok.index - 1, a], lab[ok.index - 1, b]
        pr.append(dict(pair=f"{a:03d}->{b:03d}", label=f"{NAMES[a]} -> {NAMES[b]}", n=len(ok),
                       med_rom_a=round(ok[a].median(), 1), med_rom_b=round(ok[b].median(), 1),
                       frac_rom_drop=round((diff < 0).mean(), 2), wilcoxon_p=float(f"{p:.2g}"),
                       rasch_a=round(rasch[a], 2), rasch_b=round(rasch[b], 2),
                       frac_score_drop=round((sb < sa).mean(), 2)))
    P = pd.DataFrame(pr)
    print(P.to_string(index=False))
    P.to_csv(out / "A_paired_tests.csv", index=False, encoding="utf-8-sig")

    fr = FAMILIES["forearm rotation (S1 roll)"][2]
    med = np.array([W[m].median() for m in fr])
    th = np.array([THEORY[m] for m in fr])
    print("\nForearm-rotation family (6 items):")
    print(f"  theory vs Rasch      tau {kendalltau(th, rasch[fr]).statistic:+.2f}")
    print(f"  theory vs -ROM       tau {kendalltau(th, -med).statistic:+.2f}")
    print(f"  Rasch  vs -ROM       rho {spearmanr(rasch[fr], -med).statistic:+.2f}")


if __name__ == "__main__":
    main()
