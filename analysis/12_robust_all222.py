"""Robust re-analysis WITHOUT any assumption about folder order (v2 design, 2026-11-23).

Why: the folder order (which assessments are baseline vs discharge, which belong to the same patient) is
undocumented. The paper reports upper-limb FMA gains of ~7.5 (control) and ~11.3 (experimental) points
(/66), but folders 001-120 vs 121-222 differ by only 1.9 points (/54), so "001-120 = baseline" is not
supported. Design v2:
  * unit = assessment; main sample = all 222 assessments;
  * the 222 come from 120 patients (120 baseline + 102 discharge), so a patient contributes at most 2
    assessments. Worst case (the two assessments of a patient perfectly dependent) = 120 independent
    units. Conservative inference = information of n_eff = 120:
      - test statistics scaled by sqrt(120/222) ("worst-case p");
      - confidence intervals by m-out-of-n bootstrap (resample 120 of the 222 assessments);
      - regression SEs multiplied by sqrt(222/120);
  * leakage-free check for B: AUC of the pre-specified primary feature per item, no model fitting;
  * split replication: folders 001-111 vs 112-222 analysed separately (arbitrary split; may share patients).

Usage: python -I 12_robust_all222.py <results dir>
Inputs: A_paired_metrics_long.csv, features_v1.csv ; Label from features.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata, norm, chi2_contingency, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from girth import pcm_mml

warnings.filterwarnings("ignore")
res = Path(sys.argv[1])
N_ALL, N_EFF = 222, 120
SCALE = np.sqrt(N_EFF / N_ALL)
rng = np.random.default_rng(20261123)


def worst_p(p):
    """two-sided p rescaled to n_eff (z * sqrt(n_eff/n))."""
    z = norm.isf(p / 2)
    return float(2 * norm.sf(z * SCALE))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k]); adj[k] = min(run, 1)
    return adj


def rrb(a, b):
    d = np.asarray(b) - np.asarray(a); d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d)); return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


out_lines = []
def say(s=""):
    print(s); out_lines.append(s)


# ======================= A =======================
PAIRS = [(11, 12), (16, 18), (11, 13), (15, 17), (14, 13), (11, 14), (8, 11), (5, 11)]
long = pd.read_csv(res / "A_paired_metrics_long.csv")
rowsA = []
for metric in ["M1", "M2", "M3", "M4"]:
    for sample, lo, hi in [("all 222", 1, 222), ("split 001-111", 1, 111), ("split 112-222", 112, 222)]:
        W = long[(long.metric == metric) & long.assess.between(lo, hi)].pivot(index="assess", columns="move", values="value")
        block = []
        for a, b in PAIRS:
            ok = W[[a, b]].dropna(); ok = ok[(ok[a] > 0) | (ok[b] > 0)]
            p = wilcoxon(ok[a], ok[b]).pvalue
            r = rrb(ok[a].values, ok[b].values)
            row = dict(metric=metric, sample=sample, pair=f"{a:03d}->{b:03d}", n=len(ok), r=r, p=p,
                       frac_drop=float((ok[b] < ok[a]).mean()))
            if sample == "all 222":
                row["p_worst"] = worst_p(p)
                if metric == "M1":
                    vals = ok.values
                    m = int(round(len(vals) * N_EFF / N_ALL))
                    bs = [rrb(*vals[rng.choice(len(vals), m, replace=True)].T) for _ in range(2000)]
                    row["ci_lo"], row["ci_hi"] = np.percentile(bs, [2.5, 97.5])
            block.append(row)
        for row, ph in zip(block, holm([r_["p"] for r_ in block])):
            row["p_holm"] = ph
        if sample == "all 222":
            for row, ph in zip(block, holm([r_["p_worst"] for r_ in block])):
                row["p_holm_worst"] = ph
        rowsA += block
A = pd.DataFrame(rowsA)
A.to_csv(res / "v2_A_pairs.csv", index=False)
say("=== A (v2): all 222 assessments, M1 full range ===")
m = A[(A.metric == "M1") & (A["sample"] == "all 222")]
for _, r in m.iterrows():
    say(f"  {r.pair}  n={r.n}  r={r.r:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}]  drop={r.frac_drop:.0%}  "
        f"Holm p={r.p_holm:.2g}  worst-case Holm p={r.p_holm_worst:.2g}")
say("  robustness: count of 12 analyses (4 metrics x 3 samples) with Holm p<0.05 | same sign as main")
for pr in m.pair:
    s = A[A.pair == pr]
    main_sign = np.sign(m[m.pair == pr].r.iloc[0])
    say(f"    {pr}: {int((s.p_holm < 0.05).sum())}/12 significant, {int((np.sign(s.r) == main_sign).sum())}/12 same sign")

# Rasch on all 222 (descriptive)
feat = pd.read_csv(res / "features_v1.csv")
lab = feat.pivot(index="assess", columns="move", values="score").sort_index().values
rasch = np.nanmean(np.asarray(pcm_mml(lab.T.astype(int))["Difficulty"]), axis=1)
pd.DataFrame(dict(item=[f"{i:03d}" for i in range(27)], rasch_222=rasch, mean_score=lab.mean(0))).to_csv(
    res / "v2_rasch_222.csv", index=False)
say("  Rasch (222) difficulty for pair items: " + ", ".join(f"{i:03d}:{rasch[i]:+.2f}" for i in [5, 8, 11, 12, 13, 14, 15, 16, 17, 18]))

# ======================= B =======================
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
F = feat[~flat & ~feat.short.astype(bool)].copy()
say(f"\n=== B (v2): all 222, analysis rows {len(F)} of {len(feat)} (flat {int(flat.sum())}, short {int((~flat & feat.short.astype(bool)).sum())}) ===")

# B1 leakage-free: primary feature AUC, fixed positive direction, no fitting
rowsB = []
for mv in range(27):
    d = F[F.move == mv]; y = d.score.values; x = d.primary.values; ok = ~np.isnan(x)
    a01 = roc_auc_score(y[ok] >= 1, x[ok]); s = ok & (y >= 1)
    a12 = roc_auc_score(y[s] == 2, x[s])
    rowsB.append(dict(item=mv, primary=d.primary_name.iloc[0], uni_0v12=a01, uni_1v2=a12))
U = pd.DataFrame(rowsB)

# B2 CV model on all 222 (same fixed feature sets as 07)
IMU_F = ["S1_pitch_rom", "S1_yaw_rom", "S1_roll_rom", "S2_pitch_rom", "S2_yaw_rom", "S2_roll_rom",
         "elbow_rom", "S1_w_peak", "S2_w_peak", "S1_sparc", "dur_s"]
GLV_F = ["g_col0_rom", "g_col1_rom", "g_w_peak", "g_sparc", "f1_rom", "f12_rom", "fall_rom", "fall_max", "dur_s"]


def prep(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if c.endswith(("_rom", "_peak", "_max")) or c == "dur_s":
            X[c] = np.log1p(X[c].clip(lower=0))
    return X.values


oof = []
for mv in range(27):
    cols = IMU_F if (mv <= 14 or mv == 26) else GLV_F
    d = F[F.move == mv]; X, y = prep(d, cols), d.score.values.astype(int)
    P = np.zeros((len(y), 3)); cnt = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=20261123).split(X, y):
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        mdl.fit(X[tr], y[tr]); p = np.zeros((len(te), 3)); p[:, mdl.classes_] = mdl.predict_proba(X[te])
        P[te] += p; cnt[te] += 1
    P /= cnt[:, None]; E = P @ np.array([0, 1, 2])
    U.loc[U.item == mv, "cv_0v12"] = roc_auc_score(y >= 1, E)
    s = y >= 1
    U.loc[U.item == mv, "cv_1v2"] = roc_auc_score(y[s] == 2, E[s])
    oof.append(pd.DataFrame(dict(assess=d.assess.values, move=mv, score=y, p0=P[:, 0], p1=P[:, 1], p2=P[:, 2])))
U.to_csv(res / "v2_B_item_auc.csv", index=False)
for lbl, c0, c1 in [("leakage-free univariate (primary feature)", "uni_0v12", "uni_1v2"), ("CV model", "cv_0v12", "cv_1v2")]:
    w = wilcoxon(U[c0], U[c1])
    say(f"  {lbl}: median 0v12 {U[c0].median():.3f}, 1v2 {U[c1].median():.3f}; 1v2 lower in "
        f"{(U[c1] < U[c0]).sum()}/27; Wilcoxon over items p={w.pvalue:.2g}")
# split replication (CV AUC computed within each half would need refits; use univariate, no fitting)
for lbl, lo, hi in [("split 001-111", 1, 111), ("split 112-222", 112, 222)]:
    a0, a1 = [], []
    for mv in range(27):
        d = F[(F.move == mv) & F.assess.between(lo, hi)]; y = d.score.values; x = d.primary.values; ok = ~np.isnan(x)
        a0.append(roc_auc_score(y[ok] >= 1, x[ok])); s = ok & (y >= 1); a1.append(roc_auc_score(y[s] == 2, x[s]))
    a0, a1 = np.array(a0), np.array(a1)
    say(f"  {lbl} univariate: median 0v12 {np.median(a0):.3f}, 1v2 {np.median(a1):.3f}, 1v2 lower in {(a1 < a0).sum()}/27")
bad = U[(U.uni_0v12 < 0.7)]
say(f"  items where the pre-specified primary feature fails (univariate AUC 0v12 < 0.7): "
    + ", ".join(f"{int(r['item']):03d} ({r['primary']}, {r['uni_0v12']:.2f})" for _, r in bad.iterrows()))

# H2 discordance (D1, D2) on CV predictions
D = pd.concat(oof, ignore_index=True)
Pm = D[["p0", "p1", "p2"]].values; y = D.score.values
D["p_obs"] = Pm[np.arange(len(y)), y]; D["argmax"] = Pm.argmax(1)
D["D1"] = D.p_obs < 0.10; D["D2"] = (D.argmax != y) & (Pm.max(1) >= 0.70)
ent = -(Pm * np.log(np.clip(Pm, 1e-12, 1))).sum(1) / np.log(3); D["entropy"] = ent
say("  entropy by score: " + ", ".join(f"{k}:{v:.2f}" for k, v in D.groupby("score").entropy.mean().items()))
for col in ["D1", "D2"]:
    tab = pd.crosstab(D.score, D[col]); chi = chi2_contingency(tab)
    chi_w = chi2_contingency((tab * N_EFF / N_ALL).round())  # conservative: counts scaled to n_eff
    up = int((D[col] & (D.argmax < y)).sum()); dn = int((D[col] & (D.argmax > y)).sum())
    bt = binomtest(up, up + dn).pvalue
    bt_w = binomtest(int(round(up * N_EFF / N_ALL)), int(round((up + dn) * N_EFF / N_ALL))).pvalue
    say(f"  {col}: rate by score " + ", ".join(f"{k}:{v:.1%}" for k, v in D.groupby('score')[col].mean().items())
        + f"; chi2 p={chi.pvalue:.2g} (worst-case {chi_w.pvalue:.2g}); above/below kinematic guess {up}/{dn}, "
        f"binomial p={bt:.2g} (worst-case {bt_w:.2g})")
excess = D[D.score == 1].D1.mean() - D[D.score != 1].D1.mean(); null = []
for _ in range(1000):
    yy = D.groupby("move").score.transform(lambda s: rng.permutation(s.values)).values
    pp = Pm[np.arange(len(yy)), yy] < 0.10; null.append(pp[yy == 1].mean() - pp[yy != 1].mean())
say(f"  D1 score-1 excess {excess:+.3f}; permutation null mean {np.mean(null):+.3f}; p={np.mean(np.array(null) >= excess):.3f}")
D.to_csv(res / "v2_B_oof.csv", index=False)

# ======================= H3 =======================
SH, EL, FO, WR = [0, 1, 2, 3, 6, 10], [4, 7], [5, 8, 11, 12, 13, 14], [15, 16, 17, 18, 19]
def coupling(r, mv):
    L = np.log
    if mv in SH: return L((r.elbow_rom + 1) / (r.S2_rom_max + 1))
    if mv in EL: return L((r.S2_rom_max + 1) / (r.elbow_rom + 1))
    if mv in FO: return L((r.S2_rom_max + 1) / (r.S1_roll_rom + 1))
    if mv in (9, 26): return L((r.S2_rom_max + 1) / (r.S1_rom_max + 1))
    if mv in WR: return L((r.fall_rom + 1) / (r.g_pitch_rom + 1))
    return np.nan
parts = []
for mv, g in F[F.score.isin([1, 2])].groupby("move"):
    g = g.copy(); imu = mv <= 14 or mv == 26
    rc = ["S1_rom_max", "S2_rom_max", "elbow_rom"] if imu else ["g_pitch_rom", "fall_rom", "f12_rom"]
    R = np.log1p(g[rc].clip(lower=0).fillna(g[rc].median()))
    pc = PCA(1).fit_transform(StandardScaler().fit_transform(R))[:, 0]
    g["magnitude"] = pc if np.corrcoef(pc, R.sum(1))[0, 1] > 0 else -pc
    g["coupling"] = g.apply(lambda r: coupling(r, mv), axis=1)
    g["sparc"] = g.S1_sparc if imu else g.g_sparc
    g["peaks_ps"] = (g.S1_npeaks if imu else g.g_npeaks) / g.dur_s.clip(lower=0.5)
    g["logdur"] = np.log(g.dur_s.clip(lower=0.5))
    for c in ["magnitude", "coupling", "sparc", "peaks_ps", "logdur"]:
        s_ = g[c]; g[c + "_z"] = (s_ - s_.mean()) / s_.std() if s_.notna().sum() > 5 and s_.std() > 0 else np.nan
    parts.append(g)
Q = pd.concat(parts); Q["y2"] = (Q.score == 2).astype(int); Q.to_csv(res / "v2_H3_features.csv", index=False)
say("\n=== H3 (v2, exploratory): all 222, score 2 vs 1 ===")
infl = np.sqrt(N_ALL / N_EFF)
def fit_report(sub, terms, label):
    sub = sub.dropna(subset=terms)
    f = smf.logit("y2 ~ " + " + ".join(terms) + " + C(move)", data=sub).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": sub.assess.values})
    out = []
    for t in terms:
        b, se = f.params[t], f.bse[t] * infl
        p = 2 * norm.sf(abs(b / se))
        out.append(f"{t.replace('_z','')} OR {np.exp(b):.2f} [{np.exp(b-1.96*se):.2f},{np.exp(b+1.96*se):.2f}] p={p:.2g}")
    say(f"  {label} (n={len(sub)}; worst-case SE): " + " | ".join(out))
    return f
fit_report(Q, ["magnitude_z", "coupling_z", "sparc_z", "peaks_ps_z", "logdur_z"], "pooled, coupling-defined items")
for lbl, ids in [("wrist", WR), ("forearm", FO), ("elbow", EL), ("009/026", [9, 26]), ("shoulder", SH)]:
    fit_report(Q[Q.move.isin(ids)], ["magnitude_z", "coupling_z", "peaks_ps_z"], lbl)

(res / "v2_summary.txt").write_text("\n".join(out_lines), encoding="utf-8")
