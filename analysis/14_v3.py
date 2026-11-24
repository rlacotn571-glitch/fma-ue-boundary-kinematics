"""v3 analysis (after adversarial review, 2026-11-23).

Changes from v2 (12_robust_all222.py):
  A  only like-for-like pairs under the canonical FMA-UE definitions (Fugl-Meyer 1975):
       011 -> 014  pronation/supination, elbow 90 deg + shoulder 0 (mixed)  vs elbow 0 + shoulder flexed 30-90 (isolated)
       015 -> 017  wrist dorsiflexion, elbow 90 vs elbow 0
       016 -> 018  wrist flexion/extension, elbow 90 vs elbow 0
     012 and 013 are shoulder tasks with the forearm held pronated / mid-position in the canonical FMA-UE,
     so forearm-roll comparisons with them are invalid and are dropped. Synergy items (005, 008) vs 011
     differ in task goal and are dropped. Processed signals (variance-normalised by the providers) dropped.
     Measures: M1 full range, M2 robust range, M3 peak speed, M5 rotation path (integral of |angular speed|,
     frame-free; S1 gyro norm for 011/014, |d pitch/dt| for glove items).
  Dependence (v5 correction): worst-case n_eff = 222^2 / max sum(m_i^2). The visits of the 7 erroneous
  sessions are not reported, so the worst case is all 109 completers with 2 assessments and 4 of the 11
  dropouts with 1: 222^2/(109*4 + 4) = 112.0. (Earlier versions used 102 x 2 + 18 x 1 = 115.7.)
  B  adjacent boundaries 0v1 and 1v2 (plus 0v12); ordinal (cumulative, two L2 binary logits) CV model;
     inference on the item-median AUC difference by cluster bootstrap over assessments (m = n and m = 116).
     Discordance: calibration null (labels simulated from the out-of-fold probabilities), cluster bootstrap.
  H3 non-target joint range entered separately from target range (no ratio, no PCA), item fixed effects.

Usage: python -I 14_v3.py <extracted/REHAB dir> <results dir>
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import wilcoxon, rankdata, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
root, res = Path(sys.argv[1]) / "Rehab_assessment" / "d01_raw_data", Path(sys.argv[2])
FS, N_ALL = 50.0, 222
N_EFF = N_ALL ** 2 / (109 * 4 + 4)  # worst case: all 109 completers twice, 4 of 11 dropouts once (7 erroneous sessions of unknown visit)
M_EFF = int(round(N_EFF))
SC = np.sqrt(N_EFF / N_ALL)
rng = np.random.default_rng(20261123)
lines = []
def say(s=""):
    print(s); lines.append(s)
say(f"worst-case n_eff = {N_EFF:.1f}")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); run = 0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k]); a[k] = min(run, 1)
    return a


def rrb(a, b):
    d = np.asarray(b) - np.asarray(a); d = d[d != 0]
    if len(d) == 0: return np.nan
    r = rankdata(np.abs(d)); return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def ang(x):
    return medfilt(np.degrees(np.unwrap(np.radians(x))), 5)


# ============ A ============
PAIRS = [(11, 14), (15, 17), (16, 18)]
CH = {11: ("S1", 2), 14: ("S1", 2), 15: ("G", 1), 16: ("G", 1), 17: ("G", 1), 18: ("G", 1)}
rec = []
for k in range(1, 223):
    for m, (src, col) in CH.items():
        x = np.load(root / f"{k:03d}" / f"{m:03d}_{1 if src == 'S1' else 3}.npy", allow_pickle=False)
        if len(x) < 25:
            continue
        if np.std(x[:, col]) < 1e-3:
            v = dict(M1=0.0, M2=0.0, M3=0.0, M5=0.0)
        else:
            a = ang(x[:, col])
            w = np.linalg.norm(x[:, 9:12], axis=1) if src == "S1" else np.abs(np.diff(a)) * FS
            v = dict(M1=min(a.max() - a.min(), 180.0), M2=min(np.percentile(a, 95) - np.percentile(a, 5), 180.0),
                     M3=float(np.percentile(w, 95)), M5=float(np.sum(w) / FS))
        for kk, vv in v.items():
            rec.append(dict(assess=k, move=m, metric=kk, value=vv))
L = pd.DataFrame(rec); L.to_csv(res / "v5_A_metrics.csv", index=False)
rows = []
for metric in ["M1", "M2", "M3", "M5"]:
    for sample, lo, hi in [("all", 1, 222), ("half1", 1, 111), ("half2", 112, 222)]:
        W = L[(L.metric == metric) & L.assess.between(lo, hi)].pivot(index="assess", columns="move", values="value")
        blk = []
        for a, b in PAIRS:
            ok = W[[a, b]].dropna(); ok = ok[(ok[a] > 0) | (ok[b] > 0)]
            p = wilcoxon(ok[a], ok[b]).pvalue
            row = dict(metric=metric, sample=sample, pair=f"{a:03d}->{b:03d}", n=len(ok),
                       r=rrb(ok[a].values, ok[b].values), frac_drop=float((ok[b] < ok[a]).mean()), p=p)
            if sample == "all":
                row["p_worst"] = float(2 * norm.sf(norm.isf(p / 2) * SC))
                vals = ok.values
                bs = [rrb(*vals[rng.choice(len(vals), int(round(len(vals) * N_EFF / N_ALL)))].T) for _ in range(2000)]
                row["ci_lo"], row["ci_hi"] = np.percentile(bs, [2.5, 97.5])
            blk.append(row)
        for row, ph in zip(blk, holm([r["p"] for r in blk])): row["p_holm"] = ph
        if sample == "all":
            for row, ph in zip(blk, holm([r["p_worst"] for r in blk])): row["p_holm_worst"] = ph
        rows += blk
A = pd.DataFrame(rows); A.to_csv(res / "v5_A_pairs.csv", index=False)
say("=== A v3 (3 like-for-like pairs, Holm over 3) ===")
for _, r in A[(A["sample"] == "all")].iterrows():
    say(f"  {r.metric} {r.pair} n={r.n} r={r.r:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}] drop={r.frac_drop:.0%} "
        f"Holm p={r.p_holm:.2g} worst Holm p={r.p_holm_worst:.2g}")
for pr in A.pair.unique():
    s = A[A.pair == pr]; sg = np.sign(s[(s.metric == "M1") & (s["sample"] == "all")].r.iloc[0])
    say(f"  {pr}: Holm p<0.05 in {(s.p_holm < 0.05).sum()}/12; same sign in {(np.sign(s.r) == sg).sum()}/12")

# ============ B ============
feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
F = feat[~flat & ~feat.short.astype(bool)].copy()
IMU_F = ["S1_pitch_rom", "S1_yaw_rom", "S1_roll_rom", "S2_pitch_rom", "S2_yaw_rom", "S2_roll_rom",
         "elbow_rom", "S1_w_peak", "S2_w_peak", "S1_sparc", "dur_s"]
GLV_F = ["g_col0_rom", "g_col1_rom", "g_w_peak", "g_sparc", "f1_rom", "f12_rom", "fall_rom", "fall_max", "dur_s"]


def prep(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if c.endswith(("_rom", "_peak", "_max")) or c == "dur_s":
            X[c] = np.log1p(X[c].clip(lower=0))
    return X.values


def mdl():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def oof_ordinal(X, y, reps=10):
    P = np.zeros((len(y), 3)); c = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            g1 = mdl().fit(X[tr], (y[tr] >= 1).astype(int)).predict_proba(X[te])[:, 1]
            g2 = mdl().fit(X[tr], (y[tr] >= 2).astype(int)).predict_proba(X[te])[:, 1]
            g2 = np.minimum(g2, g1)
            P[te] += np.column_stack([1 - g1, g1 - g2, g2]); c[te] += 1
    return P / c[:, None]


def aucs(y, s):
    out = {}
    for name, pos, keep in [("0v1", y == 1, y <= 1), ("1v2", y == 2, y >= 1), ("0v12", y >= 1, np.ones_like(y, bool))]:
        yy, ss = pos[keep], s[keep]
        out[name] = roc_auc_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan
    return out


oofs, rowsB = [], []
for mv in range(27):
    cols = IMU_F if (mv <= 14 or mv == 26) else GLV_F
    d = F[F.move == mv]; y = d.score.values.astype(int)
    P = oof_ordinal(prep(d, cols), y); E = P @ np.array([0, 1, 2])
    a_cv = aucs(y, E)
    x = d.primary.values; ok = ~np.isnan(x); a_un = aucs(y[ok], x[ok])
    rowsB.append(dict(item=mv, n0=(y == 0).sum(), n1=(y == 1).sum(), n2=(y == 2).sum(),
                      **{f"cv_{k}": v for k, v in a_cv.items()}, **{f"uni_{k}": v for k, v in a_un.items()}))
    oofs.append(pd.DataFrame(dict(assess=d.assess.values, move=mv, score=y, prim=x, E=E, p0=P[:, 0], p1=P[:, 1], p2=P[:, 2])))
B = pd.DataFrame(rowsB); B.to_csv(res / "v5_B_item_auc.csv", index=False)
D = pd.concat(oofs, ignore_index=True); D.to_csv(res / "v5_B_oof.csv", index=False)
say("\n=== B v3 ===")
for pref in ["cv", "uni"]:
    say(f"  {pref}: median AUC 0v1 {B[pref+'_0v1'].median():.3f}, 1v2 {B[pref+'_1v2'].median():.3f}, 0v12 {B[pref+'_0v12'].median():.3f};"
        f" 1v2 < 0v1 in {(B[pref+'_1v2'] < B[pref+'_0v1']).sum()}/27")

# cluster bootstrap over assessments for median(AUC0v1 - AUC1v2)
assess_ids = np.array(sorted(D.assess.unique()))
groups = {a: g for a, g in D.groupby("assess")}
def med_diff(Dd, col):
    diffs = []
    for mv, g in Dd.groupby("move"):
        y, s = g.score.values, g[col].values
        ok = ~np.isnan(s); y, s = y[ok], s[ok]
        a = aucs(y, s); diffs.append(a["0v1"] - a["1v2"])
    return np.nanmedian(diffs)
for col, lbl in [("E", "CV ordinal"), ("prim", "single feature")]:
    obs = med_diff(D, col)
    for m, tag in [(len(assess_ids), "m=n"), (M_EFF, f"m={M_EFF}")]:
        bs = []
        for _ in range(500):
            ids = rng.choice(assess_ids, m, replace=True)
            bs.append(med_diff(pd.concat([groups[i] for i in ids]), col))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        say(f"  {lbl}: median(AUC0v1 - AUC1v2) = {obs:+.3f}, cluster bootstrap {tag} 95% CI [{lo:+.3f}, {hi:+.3f}]")

# calibration by class
Pm = D[["p0", "p1", "p2"]].values; y = D.score.values
for c in range(3):
    pr = Pm[:, c]; obs_c = (y == c).astype(float)
    bins = np.clip((pr * 10).astype(int), 0, 9)
    ece = sum(abs(pr[bins == b].mean() - obs_c[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    say(f"  calibration class {c}: mean predicted {pr.mean():.3f} vs observed {obs_c.mean():.3f}; ECE {ece:.3f}")

# discordance D1 with calibration null and cluster bootstrap
pobs = Pm[np.arange(len(y)), y]; D["D1"] = pobs < 0.10
def excess(yv, flag):
    return flag[yv == 1].mean() - flag[yv != 1].mean()
obs_ex = excess(y, D.D1.values)
rates = D.groupby("score").D1.mean()
null = []
cum = Pm.cumsum(1)
for _ in range(1000):
    u = rng.random(len(y))[:, None]; ys = (u > cum).sum(1)
    null.append(excess(ys, Pm[np.arange(len(ys)), ys] < 0.10))
null = np.array(null)
bs = []
for _ in range(1000):
    ids = rng.choice(assess_ids, M_EFF, replace=True); g = pd.concat([groups[i] for i in ids])
    gi = D.loc[g.index] if False else None
    sub = D[D.assess.isin(ids)]  # placeholder replaced below
bs = []
Dg = {a: D.loc[idx] for a, idx in D.groupby("assess").groups.items()}
for _ in range(1000):
    ids = rng.choice(assess_ids, M_EFF, replace=True); g = pd.concat([Dg[i] for i in ids])
    bs.append(excess(g.score.values, g.D1.values))
lo, hi = np.percentile(bs, [2.5, 97.5])
say(f"  D1 rates by score: " + ", ".join(f"{k}:{v:.1%}" for k, v in rates.items()))
say(f"  D1 excess at score 1 = {obs_ex:+.3f}; worst-case cluster bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
say(f"  calibration null (labels drawn from the model's own probabilities): mean excess {null.mean():+.3f}, "
    f"95% range [{np.percentile(null,2.5):+.3f}, {np.percentile(null,97.5):+.3f}]; P(null >= observed) = {(null >= obs_ex).mean():.3f}")
ent = -(Pm * np.log(np.clip(Pm, 1e-12, 1))).sum(1) / np.log(3); D["entropy"] = ent
say("  entropy by score: " + ", ".join(f"{k}:{v:.2f}" for k, v in D.groupby('score').entropy.mean().items()))

# ============ H3 v3: separate target and non-target ranges ============
SH, EL, FO, WR = [0, 1, 2, 3, 6, 10], [4, 7], [5, 8, 11, 12, 13, 14], [15, 16, 17, 18, 19]
def tgt_non(r, mv):
    if mv in SH: return r.S2_rom_max, r.elbow_rom
    if mv in EL: return r.elbow_rom, r.S2_rom_max
    if mv in FO: return r.S1_roll_rom, r.S2_rom_max
    if mv in (9, 26): return r.S1_rom_max, r.S2_rom_max
    if mv in WR: return r.g_pitch_rom, r.fall_rom
    return np.nan, np.nan
parts = []
for mv, g in F[F.score.isin([1, 2]) & F.move.isin(SH + EL + FO + WR + [9, 26])].groupby("move"):
    g = g.copy(); imu = mv <= 14 or mv == 26
    tn = g.apply(lambda r: tgt_non(r, mv), axis=1, result_type="expand")
    g["tgt"], g["non"] = np.log1p(tn[0].clip(lower=0)), np.log1p(tn[1].clip(lower=0))
    g["peaks"] = (g.S1_npeaks if imu else g.g_npeaks) / g.dur_s.clip(lower=0.5)
    g["logdur"] = np.log(g.dur_s.clip(lower=0.5))
    for c in ["tgt", "non", "peaks", "logdur"]:
        s = g[c]; g[c + "_z"] = (s - s.mean()) / s.std() if s.std() > 0 else 0.0
    parts.append(g)
Q = pd.concat(parts); Q["y2"] = (Q.score == 2).astype(int); Q.to_csv(res / "v5_H3.csv", index=False)
infl = np.sqrt(N_ALL / N_EFF)
terms = ["tgt_z", "non_z", "peaks_z", "logdur_z"]
Xv = Q[terms].dropna().values
say("\n=== H3 v3 (exploratory): score 2 vs 1, item FE, cluster SE x sqrt(222/115.7) ===")
say("  VIF: " + ", ".join(f"{t}:{variance_inflation_factor(Xv, i):.2f}" for i, t in enumerate(terms)))
hrows = []
for lbl, ids in [("pooled", SH + EL + FO + WR + [9, 26]), ("wrist", WR), ("forearm", FO), ("elbow", EL),
                 ("009/026", [9, 26]), ("shoulder", SH)]:
    s = Q[Q.move.isin(ids)].dropna(subset=terms)
    f = smf.logit("y2 ~ " + " + ".join(terms) + " + C(move)", data=s).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": s.assess.values})
    out = []
    for t in terms:
        b, se = f.params[t], f.bse[t] * infl; p = 2 * norm.sf(abs(b / se))
        out.append(f"{t[:-2]} OR {np.exp(b):.2f} [{np.exp(b-1.96*se):.2f},{np.exp(b+1.96*se):.2f}] p={p:.2g}")
        hrows.append(dict(family=lbl, term=t[:-2], OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=p, n=len(s)))
    say(f"  {lbl} (n={len(s)}): " + " | ".join(out))
pd.DataFrame(hrows).to_csv(res / "v5_H3_or.csv", index=False)
(res / "v5_A_summary.txt").write_text("\n".join(lines), encoding="utf-8")


