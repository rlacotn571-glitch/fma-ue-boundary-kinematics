"""Coupling / smoothness effect on score 2 vs 1 by item family (follow-up to 09_B_quality.py).
Usage: python -I 10_B_quality_by_family.py <B_quality_features.csv>
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

d = pd.read_csv(sys.argv[1])
fams = [("wrist 015-019", [15, 16, 17, 18, 19]), ("shoulder", [0, 1, 2, 3, 6, 10]),
        ("elbow", [4, 7]), ("forearm", [5, 8, 11, 12, 13, 14]), ("009/026", [9, 26])]
for lbl, ids in fams:
    s = d[d.move.isin(ids)].dropna(subset=["magnitude_z", "coupling_z", "peaks_ps_z"])
    f = smf.logit("y2 ~ magnitude_z + coupling_z + peaks_ps_z + C(move)", data=s).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": s.assess.values})
    ci = np.exp(f.conf_int())
    print(f"{lbl:14s} n={len(s):4d}  coupling OR {np.exp(f.params['coupling_z']):.2f} "
          f"[{ci.loc['coupling_z', 0]:.2f}, {ci.loc['coupling_z', 1]:.2f}] p={f.pvalues['coupling_z']:.2g}"
          f" | peaks/s OR {np.exp(f.params['peaks_ps_z']):.2f} p={f.pvalues['peaks_ps_z']:.2g}"
          f" | magnitude OR {np.exp(f.params['magnitude_z']):.2f}")
