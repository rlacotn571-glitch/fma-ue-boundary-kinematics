"""v6 analyses after editorial critique (2026-11-24).

1. Separate binary models per boundary (0v1 fitted on items scored 0/1; 1v2 on items scored 1/2), so the 1v2 slope is not
   borrowed from the 0v1 boundary (proportional-odds constraint). Same fixed features, L2 logistic C=1, 10 x 5-fold CV.
2. Refit-in-bootstrap check: 100 assessment-level resamples (m = 112) with refitting (1 x 5-fold CV inside each) to see
   whether fixed-OOF intervals understate uncertainty.
3. "Pooled accuracy trap": item-level accuracy of (a) the proportional-odds model (argmax), (b) a rule that only detects
   movement (0 if no movement, else 2), (c) a rule that detects movement and guesses 1/2 at random.
4. Heterogeneity of the 1v2 AUC across items (Cochran-type Q with worst-case bootstrap SEs).
5. Wrist posture effect in degrees (median paired difference and Hodges-Lehmann estimate, M1).
6. Exploratory determinants with the target channel aligned to each item's primary feature.
Usage: python -I 22_v6.py <results dir>
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
res = Path(sys.argv[1]); rng = np.random.default_rng(20261124)
N_ALL, N_EFF = 222, 222 ** 2 / (109 * 4 + 4); M = 112
out = []
def say(s=""): print(s); out.append(s)

feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
F = feat[~flat & ~feat.short.astype(bool)].copy()
F.loc[F.move == 12, "primary"] = F.loc[F.move == 12, "S2_rom_max"]; F.loc[F.move == 12, "primary_name"] = "S2_rom_max"
F.loc[F.move == 13, "primary"] = F.loc[F.move == 13, "S2_pitch_rom"]; F.loc[F.move == 13, "primary_name"] = "S2_pitch_rom"
IMU_F = ["S1_pitch_rom", "S1_yaw_rom", "S1_roll_rom", "S2_pitch_rom", "S2_yaw_rom", "S2_roll_rom", "elbow_rom", "S1_w_peak", "S2_w_peak", "S1_sparc", "dur_s"]
GLV_F = ["g_col0_rom", "g_col1_rom", "g_w_peak", "g_sparc", "f1_rom", "f12_rom", "fall_rom", "fall_max", "dur_s"]
def prep(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if c.endswith(("_rom", "_peak", "_max")) or c == "dur_s": X[c] = np.log1p(X[c].clip(lower=0))
    return X.values
mdl = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def oof_binary(X, y, reps=10, seed0=0):
    s = np.zeros(len(y)); c = np.zeros(len(y))
    k = int(min(5, np.bincount(y).min()))
    for r in range(reps):
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed0 + r).split(X, y):
            s[te] += mdl().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; c[te] += 1
    return s / c

# ---------- 1. separate binary models ----------
rows, oofs = [], []
for mv in range(27):
    cols = IMU_F if (mv <= 14 or mv == 26) else GLV_F
    d = F[F.move == mv]; y = d.score.values.astype(int); X = prep(d, cols)
    r = dict(item=mv)
    for name, (a, b) in {"0v1": (0, 1), "1v2": (1, 2)}.items():
        keep = (y == a) | (y == b); yy = (y[keep] == b).astype(int)
        s = oof_binary(X[keep], yy); r[f"bin_{name}"] = roc_auc_score(yy, s)
        oofs.append(pd.DataFrame(dict(assess=d.assess.values[keep], move=mv, boundary=name, y=yy, s=s)))
    rows.append(r)
BN = pd.DataFrame(rows); O = pd.concat(oofs, ignore_index=True)
BN.to_csv(res / "v6_binary_item_auc.csv", index=False); O.to_csv(res / "v6_binary_oof.csv", index=False)
gap = BN.bin_0v1 - BN.bin_1v2
say(f"[1] separate binary models: median AUC 0v1 {BN.bin_0v1.median():.3f}, 1v2 {BN.bin_1v2.median():.3f}; gap median {gap.median():.3f}; "
    f"gap>0 in {(gap>0).sum()}/27, >0.05 in {(gap>0.05).sum()}/27")
ids = np.array(sorted(O.assess.unique())); G = {a: g for a, g in O.groupby("assess")}
def med_gap(Od):
    gs = []
    for mv, g in Od.groupby("move"):
        a = {}
        for bd, h in g.groupby("boundary"):
            a[bd] = roc_auc_score(h.y, h.s) if 0 < h.y.sum() < len(h) else np.nan
        gs.append(a.get("0v1", np.nan) - a.get("1v2", np.nan))
    return np.nanmedian(gs)
for m in [len(ids), M]:
    bs = [med_gap(pd.concat([G[i] for i in rng.choice(ids, m)])) for _ in range(1000)]
    say(f"    fixed-OOF assessment bootstrap m={m}: gap 95% CI [{np.nanpercentile(bs,2.5):.3f}, {np.nanpercentile(bs,97.5):.3f}]")
fi = np.array(sorted(F.assess.unique()))

# ---------- 2. refit inside bootstrap ----------
ref = []
for bsi in range(100):
    pick = rng.choice(fi, M, replace=True)
    Fb = pd.concat([F[F.assess == a].assign(bid=j) for j, a in enumerate(pick)])
    gs = []
    for mv in range(27):
        cols = IMU_F if (mv <= 14 or mv == 26) else GLV_F
        d = Fb[Fb.move == mv]; y = d.score.values.astype(int); X = prep(d, cols); a = {}
        for name, (lo, hi) in {"0v1": (0, 1), "1v2": (1, 2)}.items():
            keep = (y == lo) | (y == hi); yy = (y[keep] == hi).astype(int)
            if yy.sum() < 3 or (len(yy) - yy.sum()) < 3: a[name] = np.nan; continue
            # group duplicates of the same assessment into the same fold
            grp = d.bid.values[keep]; ug = np.unique(grp); fold = {g: i % 5 for i, g in enumerate(rng.permutation(ug))}
            fidx = np.array([fold[g] for g in grp]); s = np.full(len(yy), np.nan)
            for k in range(5):
                tr, te = fidx != k, fidx == k
                if te.sum() == 0 or len(np.unique(yy[tr])) < 2: continue
                s[te] = mdl().fit(X[keep][tr], yy[tr]).predict_proba(X[keep][te])[:, 1]
            ok = ~np.isnan(s); a[name] = roc_auc_score(yy[ok], s[ok]) if 0 < yy[ok].sum() < ok.sum() else np.nan
        gs.append(a["0v1"] - a["1v2"])
    ref.append(np.nanmedian(gs))
say(f"[2] refit-in-bootstrap (100 resamples, m=112, duplicates kept in one fold): gap median {np.nanmedian(ref):.3f}, "
    f"95% CI [{np.nanpercentile(ref,2.5):.3f}, {np.nanpercentile(ref,97.5):.3f}]")

# ---------- 3. accuracy trap ----------
P = pd.read_csv(res / "v5_B_oof.csv"); y = P.score.values
pred = P[["p0", "p1", "p2"]].values.argmax(1)
p0, p1, p2 = [(y == k).mean() for k in range(3)]
acc_model = (pred == y).mean()
rec = {k: (pred[y == k] == k).mean() for k in range(3)}
say(f"[3] score distribution {p0:.3f}/{p1:.3f}/{p2:.3f}. Item-level accuracy: PO model {acc_model:.3f} (recall 0/1/2 = "
    f"{rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}); movement-only rule (0 vs 2) {p0+p2:.3f}; movement + random 1/2 {p0+0.5*(p1+p2):.3f}")

# ---------- 4. heterogeneity of 1v2 AUC ----------
CI = pd.read_csv(res / "v5_B_item_auc_ci.csv")
se = (CI.cv_1v2_hi - CI.cv_1v2_lo) / (2 * 1.96); w = 1 / se ** 2; mu = np.sum(w * CI.cv_1v2) / w.sum()
Q = np.sum(w * (CI.cv_1v2 - mu) ** 2); dfree = len(CI) - 1
I2 = max(0, (Q - dfree) / Q)
say(f"[4] heterogeneity of 1v2 AUC across 27 items (worst-case SEs): Q = {Q:.1f}, df = {dfree}, p = {chi2.sf(Q, dfree):.2g}, I2 = {I2:.0%}")

# ---------- 5. wrist effect in degrees ----------
L = pd.read_csv(res / "v5_A_metrics.csv"); W = L[L.metric == "M1"].pivot(index="assess", columns="move", values="value")
for a, b in [(15, 17), (16, 18), (11, 14)]:
    ok = W[[a, b]].dropna(); ok = ok[(ok[a] > 0) | (ok[b] > 0)]; dlt = (ok[b] - ok[a]).values
    wa = (dlt[:, None] + dlt[None, :]) / 2; hl = np.median(wa[np.triu_indices(len(dlt))])
    say(f"[5] {a:03d}->{b:03d}: median easier {ok[a].median():.1f} deg, harder {ok[b].median():.1f} deg; median paired change {np.median(dlt):+.1f} deg; Hodges-Lehmann {hl:+.1f} deg")

# ---------- 6. exploratory with aligned targets ----------
SH_UP = [0, 1, 2, 10, 12, 13]; SH_ROT = [3, 6]; EL = [4, 7]; FO = [5, 8, 11, 14]; WR = [15, 16, 17, 18, 19]
def tn(r, mv):
    if mv == 13 or mv == 10: return r.S2_pitch_rom, r.elbow_rom
    if mv in SH_UP: return r.S2_rom_max, r.elbow_rom
    if mv in SH_ROT: return r.S1_rom_max, r.elbow_rom
    if mv in EL: return r.elbow_rom, r.S2_rom_max
    if mv in FO: return r.S1_roll_rom, r.S2_rom_max
    if mv in (9, 26): return r.S1_rom_max, r.S2_rom_max
    if mv in WR: return r.g_pitch_rom, r.fall_rom
parts = []
for mv, g in F[F.score.isin([1, 2]) & F.move.isin(SH_UP + SH_ROT + EL + FO + WR + [9, 26])].groupby("move"):
    g = g.copy(); imu = mv <= 14 or mv == 26; t = g.apply(lambda r: tn(r, mv), axis=1, result_type="expand")
    g["tgt"], g["non"] = np.log1p(t[0].clip(lower=0)), np.log1p(t[1].clip(lower=0))
    g["peaks"] = (g.S1_npeaks if imu else g.g_npeaks) / g.dur_s.clip(lower=0.5); g["logdur"] = np.log(g.dur_s.clip(lower=0.5))
    for c in ["tgt", "non", "peaks", "logdur"]:
        s_ = g[c]; g[c + "_z"] = (s_ - s_.mean()) / s_.std() if s_.std() > 0 else 0.0
    parts.append(g)
Q3 = pd.concat(parts); Q3["y2"] = (Q3.score == 2).astype(int); infl = np.sqrt(N_ALL / N_EFF); terms = ["tgt_z", "non_z", "peaks_z", "logdur_z"]; hr = []
for lbl, idl in [("pooled", SH_UP + SH_ROT + EL + FO + WR + [9, 26]), ("wrist", WR), ("forearm", FO), ("elbow", EL), ("009/026", [9, 26]), ("shoulder", SH_UP + SH_ROT)]:
    s = Q3[Q3.move.isin(idl)].dropna(subset=terms)
    f = smf.logit("y2 ~ " + " + ".join(terms) + " + C(move)", data=s).fit(disp=0, cov_type="cluster", cov_kwds={"groups": s.assess.values})
    o = []
    for tt in terms:
        b, se_ = f.params[tt], f.bse[tt] * infl; p = 2 * norm.sf(abs(b / se_))
        hr.append(dict(family=lbl, term=tt[:-2], OR=np.exp(b), lo=np.exp(b - 1.96 * se_), hi=np.exp(b + 1.96 * se_), p=p, n=len(s)))
        o.append(f"{tt[:-2]} {np.exp(b):.2f} [{np.exp(b-1.96*se_):.2f},{np.exp(b+1.96*se_):.2f}] p={p:.2g}")
    say(f"[6] {lbl} (n={len(s)}): " + " | ".join(o))
pd.DataFrame(hr).to_csv(res / "v6_H3_or.csv", index=False)
(res / "v6_summary.txt").write_text("\n".join(out), encoding="utf-8")
