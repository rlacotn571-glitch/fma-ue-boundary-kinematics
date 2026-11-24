"""B / H2 revision (post hoc, documented): the |obs - E| >= 1 rule cannot flag score 1
(E would have to be exactly 0 or 2), so it is biased by construction. Revised, category-symmetric
definitions using out-of-fold probabilities from 07_B_boundary.py:
  D1: P(observed score | kinematics) < 0.10
  D2: argmax class != observed AND P(argmax) >= 0.70 (confident disagreement)
Also reports posterior uncertainty (entropy) by observed score, and a label-permutation null for
the score-1 excess (labels shuffled within item, probabilities fixed).

Usage: python -I 08_B_discordance_prob.py <B_oof_predictions.csv> <out dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, binomtest

D = pd.read_csv(sys.argv[1])
out = Path(sys.argv[2])
P = D[["p0", "p1", "p2"]].values
y = D.score.values
pobs = P[np.arange(len(y)), y]
amax = P.argmax(1)
ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1) / np.log(3)
D["p_obs"], D["argmax"], D["entropy"] = pobs, amax, ent
D["D1"] = pobs < 0.10
D["D2"] = (amax != y) & (P.max(1) >= 0.70)

print("Posterior by observed score (CV, folders 001-120):")
print(D.groupby("score").agg(n=("score", "size"), mean_p_obs=("p_obs", "mean"),
                             mean_entropy=("entropy", "mean")).round(3).to_string())

for col in ["D1", "D2"]:
    tab = pd.crosstab(D.score, D[col])
    chi = chi2_contingency(tab)
    up = int((D[col] & (D.argmax < y)).sum())   # observed above kinematic best guess
    dn = int((D[col] & (D.argmax > y)).sum())
    print(f"\n{col}: {int(D[col].sum())} of {len(D)} ({D[col].mean():.1%}); by score "
          f"{ {int(k): f'{v:.1%}' for k, v in D.groupby('score')[col].mean().items()} } chi2 p={chi.pvalue:.2g}")
    print(f"   observed above kinematic guess {up}, below {dn}, binomial p={binomtest(up, up + dn).pvalue:.2g}")
    # within score 2: upgraded from 1? (argmax=1) ; within score 1: kinematics say 0 or 2
    print("   cross-tab observed x kinematic argmax among flagged:")
    print(pd.crosstab(D[D[col]].score, D[D[col]].argmax).to_string())

# permutation null for score-1 excess under D1: shuffle labels within item, keep probabilities
rng = np.random.default_rng(1)
obs_excess = D[D.score == 1].D1.mean() - D[D.score != 1].D1.mean()
null = []
for _ in range(2000):
    yy = D.groupby("move").score.transform(lambda s: rng.permutation(s.values)).values
    pp = P[np.arange(len(yy)), yy] < 0.10
    null.append(pp[yy == 1].mean() - pp[yy != 1].mean())
null = np.array(null)
print(f"\nD1 score-1 excess (rate at 1 minus rate at 0/2): observed {obs_excess:+.3f}; "
      f"permutation null mean {null.mean():+.3f}; p(null >= obs) = {(null >= obs_excess).mean():.3f}")
print("  (null keeps class priors, so it shows how much of the score-1 excess is prior-driven)")

D.to_csv(out / "B_oof_with_discordance.csv", index=False, encoding="utf-8-sig")
top = D[D.D1].groupby("move").size().sort_values(ascending=False).head(8)
print("\nD1 flags by item (top 8):", top.to_dict())
