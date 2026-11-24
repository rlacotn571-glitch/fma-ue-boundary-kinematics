"""v4 (after second review). Analysis A is unchanged from v3 (14_v3.py). Changes in B and H3:
  N1  items 012 (shoulder abduction 0-90, elbow 0, forearm pronated) and 013 (shoulder flexion 90-180,
      elbow 0, forearm mid-position) are shoulder tasks in the canonical FMA-UE. Their pre-specified primary
      feature becomes upper-arm range (012: S2_rom_max, 013: S2_pitch_rom), and in H3 they move to the
      shoulder family (target = upper arm, non-target = elbow proxy).
  N2  ordinal model = penalised proportional-odds (cumulative logit) fitted by maximum likelihood with an
      L2 penalty equivalent to sklearn C = 1; thresholds are ordered by construction, so no truncation.
      Tail calibration is reported (cases with predicted class probability < 0.10). Counts are reported.
  Minor: within-item VIF; count of items with AUC gap > 0.05.

Usage: python -I 16_v4.py <results dir>
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
res = Path(sys.argv[1])
N_ALL = 222
N_EFF = N_ALL ** 2 / (109 * 4 + 4)  # worst case, see 14_v3.py (= 112.0)
M_EFF = int(round(N_EFF))
rng = np.random.default_rng(20261123)
lines = []
def say(s=""):
    print(s); lines.append(s)

feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
F = feat[~flat & ~feat.short.astype(bool)].copy()
F.loc[F.move == 12, "primary"] = F.loc[F.move == 12, "S2_rom_max"]; F.loc[F.move == 12, "primary_name"] = "S2_rom_max"
F.loc[F.move == 13, "primary"] = F.loc[F.move == 13, "S2_pitch_rom"]; F.loc[F.move == 13, "primary_name"] = "S2_pitch_rom"
IMU_F = ["S1_pitch_rom", "S1_yaw_rom", "S1_roll_rom", "S2_pitch_rom", "S2_yaw_rom", "S2_roll_rom",
         "elbow_rom", "S1_w_peak", "S2_w_peak", "S1_sparc", "dur_s"]
GLV_F = ["g_col0_rom", "g_col1_rom", "g_w_peak", "g_sparc", "f1_rom", "f12_rom", "fall_rom", "fall_max", "dur_s"]


def prep(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if c.endswith(("_rom", "_peak", "_max")) or c == "dur_s":
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


CONV = []
def fit_po(X, y, lam=0.5):
    """Penalised proportional-odds model: P(y>=k) = expit(X b - t_k), k=1,2; t1 < t2."""
    p = X.shape[1]
    def unpack(w):
        return w[:p], w[p], w[p] + np.exp(w[p + 1])
    def nll(w):
        b, t1, t2 = unpack(w); eta = X @ b
        g1, g2 = expit(eta - t1), expit(eta - t2)
        pr = np.column_stack([1 - g1, g1 - g2, g2])
        ll = np.log(np.clip(pr[np.arange(len(y)), y], 1e-12, 1)).sum()
        return -ll + lam * b @ b
    w0 = np.zeros(p + 2); w0[p] = -0.5; w0[p + 1] = 0.0
    opt = minimize(nll, w0, method="L-BFGS-B", options={"maxiter": 5000})
    CONV.append(bool(opt.success)); w = opt.x
    return unpack(w)


def predict_po(params, X):
    b, t1, t2 = params; eta = X @ b
    g1, g2 = expit(eta - t1), expit(eta - t2)
    return np.column_stack([1 - g1, g1 - g2, g2])


def oof(Xdf, y, reps=10):
    P = np.zeros((len(y), 3)); c = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xdf, y):
            Xtr, Xte = Xdf.iloc[tr].copy(), Xdf.iloc[te].copy()
            med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            prm = fit_po(((Xtr - mu) / sd).values, y[tr])
            P[te] += predict_po(prm, ((Xte - mu) / sd).values); c[te] += 1
    return P / c[:, None]


def aucs(y, s):
    o = {}
    for nm, pos, keep in [("0v1", y == 1, y <= 1), ("1v2", y == 2, y >= 1)]:
        yy, ss = pos[keep], s[keep]
        o[nm] = roc_auc_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan
    return o


rows, oofs = [], []
for mv in range(27):
    cols = IMU_F if (mv <= 14 or mv == 26) else GLV_F
    d = F[F.move == mv]; y = d.score.values.astype(int)
    P = oof(prep(d, cols), y); E = P @ np.array([0, 1, 2])
    a, x = aucs(y, E), d.primary.values; ok = ~np.isnan(x); u = aucs(y[ok], x[ok])
    rows.append(dict(item=mv, primary=d.primary_name.iloc[0], cv_0v1=a["0v1"], cv_1v2=a["1v2"], uni_0v1=u["0v1"], uni_1v2=u["1v2"]))
    oofs.append(pd.DataFrame(dict(assess=d.assess.values, move=mv, score=y, prim=x, E=E, p0=P[:, 0], p1=P[:, 1], p2=P[:, 2])))
B = pd.DataFrame(rows); B.to_csv(res / "v5_B_item_auc.csv", index=False)
D = pd.concat(oofs, ignore_index=True); D.to_csv(res / "v5_B_oof.csv", index=False)
say(f"=== B v4 (penalised proportional-odds) ===")
for pre in ["cv", "uni"]:
    g = B[pre + "_0v1"] - B[pre + "_1v2"]
    say(f"  {pre}: median 0v1 {B[pre+'_0v1'].median():.3f}, 1v2 {B[pre+'_1v2'].median():.3f}; gap>0 in {(g>0).sum()}/27, gap>0.05 in {(g>0.05).sum()}/27")
say("  items 012/013 now: " + "; ".join(f"{int(r['item']):03d} cv1v2 {r['cv_1v2']:.3f} uni0v1 {r['uni_0v1']:.3f} uni1v2 {r['uni_1v2']:.3f}" for _, r in B[B['item'].isin([12, 13])].iterrows()))
say("  lowest cv 1v2: " + ", ".join(f"{int(r['item']):03d} {r['cv_1v2']:.2f}" for _, r in B.nsmallest(5, 'cv_1v2').iterrows()))
say("  highest cv 1v2: " + ", ".join(f"{int(r['item']):03d} {r['cv_1v2']:.2f}" for _, r in B.nlargest(3, 'cv_1v2').iterrows()))

ids = np.array(sorted(D.assess.unique())); G = {a: D.loc[i] for a, i in D.groupby("assess").groups.items()}
def med_gap(Dd, col):
    gs = []
    for mv, g in Dd.groupby("move"):
        s = g[col].values; ok = ~np.isnan(s); a = aucs(g.score.values[ok], s[ok]); gs.append(a["0v1"] - a["1v2"])
    return np.nanmedian(gs)
for col, lbl in [("E", "CV ordinal"), ("prim", "single feature")]:
    obs = med_gap(D, col)
    for m in [len(ids), M_EFF]:
        bs = [med_gap(pd.concat([G[i] for i in rng.choice(ids, m)]), col) for _ in range(500)]
        say(f"  {lbl}: median gap {obs:+.3f}, cluster bootstrap m={m} 95% CI [{np.nanpercentile(bs,2.5):+.3f}, {np.nanpercentile(bs,97.5):+.3f}]")

Pm = D[["p0", "p1", "p2"]].values; y = D.score.values
for c in range(3):
    pr = Pm[:, c]; ob = (y == c).astype(float); b = np.clip((pr * 10).astype(int), 0, 9)
    ece = sum(abs(pr[b == k].mean() - ob[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any())
    t = pr < 0.10
    say(f"  class {c}: mean pred {pr.mean():.3f} obs {ob.mean():.3f} ECE {ece:.3f}; tail p<0.10: n={t.sum()}, "
        f"mean pred {pr[t].mean():.3f}, observed {ob[t].mean():.3f} ({int(ob[t].sum())} cases)")
pobs = Pm[np.arange(len(y)), y]; D["D1"] = pobs < 0.10
say("  D1 counts: " + ", ".join(f"score {k}: {int(g.D1.sum())}/{len(g)} ({g.D1.mean():.1%})" for k, g in D.groupby("score")))
say("  min p1 among score-1 discordant: %.4f" % D[(D.score == 1) & D.D1].p1.min())
ex = lambda yy, f: f[yy == 1].mean() - f[yy != 1].mean()
obs = ex(y, D.D1.values); cum = Pm.cumsum(1)
null = np.array([ex(ys, Pm[np.arange(len(ys)), ys] < 0.10) for ys in ((rng.random(len(y))[:, None] > cum).sum(1) for _ in range(1000))])
G = {a: D.loc[i] for a, i in D.groupby("assess").groups.items()}
bs = [ex(g.score.values, g.D1.values) for g in (pd.concat([G[i] for i in rng.choice(ids, M_EFF)]) for _ in range(1000))]
say(f"  excess at score 1 {obs:+.3f}, worst-case cluster bootstrap CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]; "
    f"calibration null mean {null.mean():+.3f} [{np.percentile(null,2.5):+.3f}, {np.percentile(null,97.5):+.3f}], P(null>=obs)={(null>=obs).mean():.3f}")
by_item = D[(D.score == 1) & D.D1].groupby("move").size().sort_values(ascending=False)
say("  score-1 discordant by item: " + ", ".join(f"{int(k):03d}:{v}" for k, v in by_item.items()))
ent = -(Pm * np.log(np.clip(Pm, 1e-12, 1))).sum(1) / np.log(3); D["entropy"] = ent
say("  entropy by score: " + ", ".join(f"{k}:{v:.2f}" for k, v in D.groupby('score').entropy.mean().items()))

# ---------- H3 v4 ----------
SH, EL, FO, WR = [0, 1, 2, 3, 6, 10, 12, 13], [4, 7], [5, 8, 11, 14], [15, 16, 17, 18, 19]
def tn(r, mv):
    if mv == 13: return r.S2_pitch_rom, r.elbow_rom
    if mv in SH: return r.S2_rom_max, r.elbow_rom
    if mv in EL: return r.elbow_rom, r.S2_rom_max
    if mv in FO: return r.S1_roll_rom, r.S2_rom_max
    if mv in (9, 26): return r.S1_rom_max, r.S2_rom_max
    if mv in WR: return r.g_pitch_rom, r.fall_rom
parts, vifs = [], []
for mv, g in F[F.score.isin([1, 2]) & F.move.isin(SH + EL + FO + WR + [9, 26])].groupby("move"):
    g = g.copy(); imu = mv <= 14 or mv == 26
    t = g.apply(lambda r: tn(r, mv), axis=1, result_type="expand")
    g["tgt"], g["non"] = np.log1p(t[0].clip(lower=0)), np.log1p(t[1].clip(lower=0))
    g["peaks"] = (g.S1_npeaks if imu else g.g_npeaks) / g.dur_s.clip(lower=0.5); g["logdur"] = np.log(g.dur_s.clip(lower=0.5))
    for c in ["tgt", "non", "peaks", "logdur"]:
        s = g[c]; g[c + "_z"] = (s - s.mean()) / s.std() if s.std() > 0 else 0.0
    Xv = g[["tgt_z", "non_z", "peaks_z", "logdur_z"]].dropna().values
    vifs.append(max(variance_inflation_factor(Xv, i) for i in range(4))); parts.append(g)
Q = pd.concat(parts); Q["y2"] = (Q.score == 2).astype(int); Q.to_csv(res / "v5_H3.csv", index=False)
say(f"\n=== H3 v4 === within-item max VIF: median {np.median(vifs):.2f}, max {np.max(vifs):.2f}")
infl = np.sqrt(N_ALL / N_EFF); terms = ["tgt_z", "non_z", "peaks_z", "logdur_z"]; hr = []
for lbl, idl in [("pooled", SH + EL + FO + WR + [9, 26]), ("wrist", WR), ("forearm", FO), ("elbow", EL), ("009/026", [9, 26]), ("shoulder", SH)]:
    s = Q[Q.move.isin(idl)].dropna(subset=terms)
    f = smf.logit("y2 ~ " + " + ".join(terms) + " + C(move)", data=s).fit(disp=0, cov_type="cluster", cov_kwds={"groups": s.assess.values})
    o = []
    for tt in terms:
        b, se = f.params[tt], f.bse[tt] * infl; p = 2 * norm.sf(abs(b / se))
        hr.append(dict(family=lbl, term=tt[:-2], OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=p, n=len(s)))
        o.append(f"{tt[:-2]} {np.exp(b):.2f} [{np.exp(b-1.96*se):.2f},{np.exp(b+1.96*se):.2f}] p={p:.2g}")
    say(f"  {lbl} (n={len(s)}): " + " | ".join(o))
pd.DataFrame(hr).to_csv(res / "v5_H3_or.csv", index=False)
say(f"fit_po convergence: {sum(CONV)}/{len(CONV)} fits converged")
(res / "v5_summary.txt").write_text("\n".join(lines), encoding="utf-8")

