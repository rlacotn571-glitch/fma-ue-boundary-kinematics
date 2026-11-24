"""TRIPOD+AI support: missing feature values (item 11) and per-item AUC 95% CIs (item 23a).

Per-item CIs: cluster bootstrap over assessments with m = 112 (worst-case n_eff), 1,000 resamples,
from the fixed out-of-fold predictions of 16_v4.py (v5_B_oof.csv).
Usage: python -I 20_tripod_support.py <results dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

res = Path(sys.argv[1])
feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
F = feat[~flat & ~feat.short.astype(bool)]
IMU_F = ["S1_pitch_rom", "S1_yaw_rom", "S1_roll_rom", "S2_pitch_rom", "S2_yaw_rom", "S2_roll_rom",
         "elbow_rom", "S1_w_peak", "S2_w_peak", "S1_sparc", "dur_s"]
GLV_F = ["g_col0_rom", "g_col1_rom", "g_w_peak", "g_sparc", "f1_rom", "f12_rom", "fall_rom", "fall_max", "dur_s"]
imu = F[(F.move <= 14) | (F.move == 26)][IMU_F]; glv = F[(F.move >= 15) & (F.move <= 25)][GLV_F]
n_cells = imu.size + glv.size; n_nan = int(imu.isna().sum().sum() + glv.isna().sum().sum())
rows_nan = int(imu.isna().any(axis=1).sum() + glv.isna().any(axis=1).sum())
print(f"missing feature values: {n_nan} of {n_cells} cells ({n_nan/n_cells:.2%}); rows with any missing: {rows_nan} of {len(F)}")
print("  by feature:", {**imu.isna().sum()[imu.isna().sum() > 0].to_dict(), **glv.isna().sum()[glv.isna().sum() > 0].to_dict()})

D = pd.read_csv(res / "v5_B_oof.csv")
ids = np.array(sorted(D.assess.unique())); G = {a: g for a, g in D.groupby("assess")}
rng = np.random.default_rng(20261124)


def auc(y, s, a, b):
    keep = (y == a) | (y == b)
    yy = (y[keep] == b)
    return roc_auc_score(yy, s[keep]) if 0 < yy.sum() < len(yy) else np.nan


boot = {m: {"0v1": [], "1v2": []} for m in range(27)}
for _ in range(1000):
    Bd = pd.concat([G[i] for i in rng.choice(ids, 112, replace=True)])
    for m, g in Bd.groupby("move"):
        y, s = g.score.values, g.E.values
        boot[m]["0v1"].append(auc(y, s, 0, 1)); boot[m]["1v2"].append(auc(y, s, 1, 2))
rows = []
for m, g in D.groupby("move"):
    y, s = g.score.values, g.E.values
    r = dict(item=m)
    for k, (a, b) in {"0v1": (0, 1), "1v2": (1, 2)}.items():
        r[f"cv_{k}"] = auc(y, s, a, b)
        lo, hi = np.nanpercentile(boot[m][k], [2.5, 97.5]); r[f"cv_{k}_lo"], r[f"cv_{k}_hi"] = lo, hi
    rows.append(r)
C = pd.DataFrame(rows); C.to_csv(res / "v5_B_item_auc_ci.csv", index=False)
print(C.round(3).to_string(index=False))
print(f"items whose worst-case 95% CIs for 0v1 and 1v2 do not overlap: {(C.cv_1v2_hi < C.cv_0v1_lo).sum()}/27")
