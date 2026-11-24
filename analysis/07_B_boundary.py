"""B: kinematic consistency of FMA-UE item score boundaries.

Usage: python -I 07_B_boundary.py <features_v1.csv> <out dir>

Pre-specified 2026-11-23 (before running):
  Data: features_v1.csv; exclude flat (constant) signals and signals < 0.5 s.
  Feature sets (fixed, same for all items of a sensor type, so no per-item feature picking):
    IMU items (000-014, 026): S1_pitch/yaw/roll_rom, S2_pitch/yaw/roll_rom, elbow_rom, S1_w_peak,
                              S2_w_peak, S1_sparc, dur_s
    glove items (015-025):    g_col0_rom, g_col1_rom, g_w_peak, g_sparc, f1_rom, f12_rom, fall_rom,
                              fall_max, dur_s
  Model: per item, multinomial logistic regression (L2, C=1) on standardized features
         (log1p for ranges/speeds/duration), 5-fold stratified CV repeated 10x; out-of-fold
         probabilities averaged over repeats.
  Main sample: folders 001-120.
  H1 (boundary): cross-validated AUC for 1-vs-2 is lower than for 0-vs-(1,2) across the 27 items
                 (Wilcoxon signed-rank over items).
  H2 (discordance): an item score is "kinematically discordant" if |observed - E[score|kinematics]| >= 1.
                 Discordance is more frequent at score 1 than at scores 0 or 2 (chi-square), and,
                 because labels are the ceiling of two raters' mean, discordant scores are more
                 often ABOVE than below the kinematic expectation (binomial test vs 0.5).
  Sensitivity: (a) threshold 0.75 instead of 1; (b) train on 001-120, predict 121-222
               (pooled CV over all 222 is not done: no patient IDs, so leakage cannot be prevented).
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, chi2_contingency, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
IMU_F = ["S1_pitch_rom", "S1_yaw_rom", "S1_roll_rom", "S2_pitch_rom", "S2_yaw_rom", "S2_roll_rom",
         "elbow_rom", "S1_w_peak", "S2_w_peak", "S1_sparc", "dur_s"]
GLV_F = ["g_col0_rom", "g_col1_rom", "g_w_peak", "g_sparc", "f1_rom", "f12_rom", "fall_rom",
         "fall_max", "dur_s"]
NAMES = ["sh elevation", "sh retraction", "sh abduction", "sh ext rot", "elbow flex", "supination",
         "sh add/IR", "elbow ext", "pronation", "hand to lumbar", "sh flex 90", "pro/sup e90",
         "pronation abd90", "pro/sup flex90-180", "pro/sup flex30-90", "wrist dorsi e90",
         "wrist flex/ext e90", "wrist dorsi e0", "wrist flex/ext e0", "wrist circum",
         "mass flexion", "mass extension", "thumb add", "pinch", "cylinder", "sphere", "finger-nose"]


def prep(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if c.endswith(("_rom", "_peak", "_max")) or c == "dur_s":
            X[c] = np.log1p(X[c].clip(lower=0))
    return X.values


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000))


def oof_proba(X, y, seed=20261123):
    P = np.zeros((len(y), 3))
    cnt = np.zeros(len(y))
    n_splits = min(5, np.bincount(y, minlength=3)[np.bincount(y, minlength=3) > 0].min())
    n_splits = max(2, n_splits)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=seed)
    for tr, te in rskf.split(X, y):
        m = model().fit(X[tr], y[tr])
        p = np.zeros((len(te), 3))
        p[:, m.classes_] = m.predict_proba(X[te])
        P[te] += p
        cnt[te] += 1
    return P / cnt[:, None]


def fit_predict(Xtr, ytr, Xte):
    m = model().fit(Xtr, ytr)
    p = np.zeros((len(Xte), 3))
    p[:, m.classes_] = m.predict_proba(Xte)
    return p


def aucs(y, P):
    E = P @ np.array([0, 1, 2])
    a01 = roc_auc_score(y >= 1, E) if 0 < (y >= 1).sum() < len(y) else np.nan
    s = y >= 1
    a12 = roc_auc_score(y[s] == 2, E[s]) if 0 < (y[s] == 2).sum() < s.sum() else np.nan
    return a01, a12, E


def summarize(D, thr, tag):
    D = D.copy()
    D["resid"] = D.score - D.E
    D["disc"] = D.resid.abs() >= thr
    tab = pd.crosstab(D.score, D.disc)
    rate = D.groupby("score").disc.mean()
    chi = chi2_contingency(tab)
    up = int((D.disc & (D.resid > 0)).sum())
    dn = int((D.disc & (D.resid < 0)).sum())
    bt = binomtest(up, up + dn, 0.5) if up + dn > 0 else None
    print(f"\n[{tag}] discordance threshold |obs-E| >= {thr}: {int(D.disc.sum())} of {len(D)} "
          f"({D.disc.mean():.1%})")
    print("  rate by observed score:", {int(k): f"{v:.1%}" for k, v in rate.items()},
          f" chi2 p={chi.pvalue:.2g}")
    if bt:
        print(f"  direction: above expectation {up}, below {dn}  binomial p={bt.pvalue:.2g}")
    return D


def main():
    feat = pd.read_csv(sys.argv[1])
    out = Path(sys.argv[2])
    flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
    feat = feat[~flat & ~feat.short.astype(bool)]

    item_rows, oof_rows, ext_rows = [], [], []
    for m in range(27):
        cols = IMU_F if (m <= 14 or m == 26) else GLV_F
        d = feat[feat.move == m]
        tr = d[d.assess <= 120]
        te = d[d.assess > 120]
        Xtr, ytr = prep(tr, cols), tr.score.values.astype(int)
        P = oof_proba(Xtr, ytr)
        a01, a12, E = aucs(ytr, P)
        Pext = fit_predict(Xtr, ytr, prep(te, cols))
        e01, e12, Eext = aucs(te.score.values.astype(int), Pext)
        item_rows.append(dict(item=f"{m:03d}", name=NAMES[m], n=len(tr),
                              n0=int((ytr == 0).sum()), n1=int((ytr == 1).sum()), n2=int((ytr == 2).sum()),
                              cv_auc_0v12=round(a01, 3), cv_auc_1v2=round(a12, 3),
                              ext_auc_0v12=round(e01, 3), ext_auc_1v2=round(e12, 3)))
        oof_rows.append(pd.DataFrame(dict(assess=tr.assess.values, move=m, score=ytr, E=E,
                                          p0=P[:, 0], p1=P[:, 1], p2=P[:, 2])))
        ext_rows.append(pd.DataFrame(dict(assess=te.assess.values, move=m,
                                          score=te.score.values.astype(int), E=Eext)))
    I = pd.DataFrame(item_rows)
    I.to_csv(out / "B_item_auc.csv", index=False, encoding="utf-8-sig")
    print(I.to_string(index=False))

    ok = I.dropna(subset=["cv_auc_0v12", "cv_auc_1v2"])
    w = wilcoxon(ok.cv_auc_0v12, ok.cv_auc_1v2)
    print(f"\nH1 (CV, 001-120): median AUC 0v12 = {ok.cv_auc_0v12.median():.3f}, "
          f"1v2 = {ok.cv_auc_1v2.median():.3f}; 1v2 lower in {(ok.cv_auc_1v2 < ok.cv_auc_0v12).sum()}/"
          f"{len(ok)} items; Wilcoxon p={w.pvalue:.2g}")
    okx = I.dropna(subset=["ext_auc_0v12", "ext_auc_1v2"])
    print(f"   external (train 001-120 -> 121-222): median 0v12 {okx.ext_auc_0v12.median():.3f}, "
          f"1v2 {okx.ext_auc_1v2.median():.3f}")

    D = pd.concat(oof_rows, ignore_index=True)
    D = summarize(D, 1.0, "H2 main, CV 001-120")
    summarize(D, 0.75, "sensitivity threshold 0.75")
    X = summarize(pd.concat(ext_rows, ignore_index=True), 1.0, "sensitivity external 121-222")
    D.to_csv(out / "B_oof_predictions.csv", index=False, encoding="utf-8-sig")

    by_item = D.groupby("move").agg(disc_rate=("disc", "mean"), n_disc=("disc", "sum")).reset_index()
    by_item["name"] = by_item.move.map(lambda m: NAMES[m])
    by_item = by_item.sort_values("disc_rate", ascending=False)
    by_item.to_csv(out / "B_discordance_by_item.csv", index=False, encoding="utf-8-sig")
    print("\nDiscordance by item (top 8):")
    print(by_item.head(8).to_string(index=False))
    by_asm = D.groupby("assess").disc.sum()
    print(f"\nDiscordant items per assessment: median {by_asm.median():.0f}, "
          f"max {by_asm.max()} (assessments with >=5: {(by_asm >= 5).sum()})")


if __name__ == "__main__":
    main()
