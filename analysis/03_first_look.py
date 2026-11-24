"""QC + first look: primary kinematic feature vs FMA item score (folders 001-120).

Usage: python -I 03_first_look.py <features csv> <Label.npy> <out dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
from girth import pcm_mml

feat = pd.read_csv(sys.argv[1])
lab = np.load(sys.argv[2], allow_pickle=False)
out = Path(sys.argv[3])

NAMES = ["sh elevation", "sh retraction", "sh abduction", "sh ext rot", "elbow flex", "supination",
         "sh add/IR", "elbow ext", "pronation", "hand to lumbar", "sh flex 90", "pro/sup elbow90",
         "pronation abd90", "pro/sup flex90-180", "pro/sup flex30-90", "wrist dorsi e90",
         "wrist flex/ext e90", "wrist dorsi e0", "wrist flex/ext e0", "wrist circum",
         "mass flexion", "mass extension", "thumb add", "pinch", "cylinder", "sphere", "finger-nose"]

# ---------- QC ----------
feat["flat"] = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
print("QC (all 222 assessments):")
print("  short (<0.5 s):", int(feat.short.sum()), " flat sensor:", int(feat.flat.sum()))
print("  score distribution among short:", feat[feat.short].score.value_counts().sort_index().to_dict())
print("  score distribution among flat :", feat[feat.flat].score.value_counts().sort_index().to_dict())
print("  glove files with >0 angle jumps:", int((feat.g_pitch_jumps > 0).sum()))

d = feat[(feat.assess <= 120) & ~feat.short & ~feat.flat].copy()
print(f"\nanalysis set: folders 001-120, {len(d)} rows of {120*27}")

# ---------- B: primary feature vs score ----------
FEATS = [c for c in feat.columns if c.endswith(("_rom", "_rom_max", "_peak", "_mean", "_sparc",
                                                 "_npeaks", "_max", "_min")) or c in ("dur_s", "comp_ratio")]
rows = []
for m in range(27):
    s = d[d.move == m]
    y = s.score.values
    x = s.primary.values
    ok = ~np.isnan(x)
    rho = spearmanr(x[ok], y[ok]).statistic
    sign = 1 if rho >= 0 else -1
    def auc(mask_pos, mask_set):
        yy = mask_pos[mask_set & ok]
        return roc_auc_score(yy, sign * x[mask_set & ok]) if 0 < yy.sum() < len(yy) else np.nan
    a01 = auc(y >= 1, np.ones_like(ok))
    a12 = auc(y == 2, y >= 1)
    best = max(((abs(spearmanr(s[c], y, nan_policy="omit").statistic), c) for c in FEATS
                if s[c].notna().sum() > 30 and s[c].nunique() > 3), default=(np.nan, ""))
    med = s.groupby("score").primary.median().reindex([0, 1, 2]).round(1).tolist()
    rows.append(dict(item=f"{m:03d}", name=NAMES[m], n=int(ok.sum()), primary=s.primary_name.iloc[0],
                     rho=round(rho, 2), auc_0v12=round(a01, 2), auc_1v2=round(a12, 2),
                     med_by_score=med, best_feat=best[1], best_rho=round(best[0], 2)))
B = pd.DataFrame(rows)
B.to_csv(out / "firstlook_B_item_feature.csv", index=False, encoding="utf-8-sig")
print("\n[B] primary feature vs score (folders 001-120)")
print(B.to_string(index=False))

# ---------- A: kinematic difficulty vs Rasch difficulty ----------
res = pcm_mml(lab[:120].T.astype(int))
rasch = np.nanmean(np.asarray(res["Difficulty"]), axis=1)
kin = []
for m in range(27):
    v = d[d.move == m]
    col = "S1_rom_max" if m == 26 else v.primary_name.iloc[0]
    x = v[col].dropna().values
    ref = np.percentile(x, 95)  # best performers on this movement as reference, labels not used
    kin.append(1 - np.mean(np.clip(x / ref, 0, 1)))
kin = np.array(kin)
A = pd.DataFrame(dict(item=[f"{m:03d}" for m in range(27)], name=NAMES, rasch=rasch.round(2),
                      kin_deficit=kin.round(3)))
A["rasch_rank"] = A.rasch.rank().astype(int)
A["kin_rank"] = A.kin_deficit.rank().astype(int)
A.to_csv(out / "firstlook_A_hierarchy.csv", index=False, encoding="utf-8-sig")
print("\n[A] Rasch difficulty vs kinematic deficit (higher = harder)")
print(A.sort_values("rasch").to_string(index=False))
for lbl, idx in [("all 27", range(27)), ("arm 000-014,026", list(range(15)) + [26]),
                 ("glove 015-025", range(15, 26))]:
    idx = list(idx)
    r = spearmanr(rasch[idx], kin[idx]).statistic
    t = kendalltau(rasch[idx], kin[idx]).statistic
    print(f"  {lbl:18s} Spearman {r:+.2f}  Kendall {t:+.2f}")
