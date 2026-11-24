"""Manuscript figures 1-4 (English labels, 300 dpi PNG + vector PDF).

Usage: python -I 11_figures.py <results dir> <figures out dir>
Inputs (all produced by scripts 02-10): features_v1.csv, A_paired_metrics_long.csv,
A_paired_robust.csv, B_item_auc.csv, B_quality_features.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.stats import rankdata
import statsmodels.formula.api as smf

res, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42})
BLUE, GREY, DARK = "#2166ac", "#9e9e9e", "#333333"
PAIRS = [("011", "012"), ("016", "018"), ("011", "013"), ("015", "017"),
         ("014", "013"), ("011", "014"), ("008", "011"), ("005", "011")]
SHORT = {"005": "Supination\n(flexor synergy)", "008": "Pronation\n(extensor synergy)",
         "011": "Pro/sup\nsh 0°, elbow 90°", "012": "Pronation\nsh abd 90°, elbow 0°",
         "013": "Pro/sup\nsh flex 90-180°", "014": "Pro/sup\nsh flex 30-90°",
         "015": "Wrist dorsiflexion\nelbow 90°", "017": "Wrist dorsiflexion\nelbow 0°",
         "016": "Wrist flex/ext\nelbow 90°", "018": "Wrist flex/ext\nelbow 0°"}
ONE = {"005": "Supination (flexor syn.)", "008": "Pronation (extensor syn.)",
       "011": "Pro/sup sh 0°/elb 90°", "012": "Pronation sh abd 90°/elb 0°",
       "013": "Pro/sup sh flex 90-180°", "014": "Pro/sup sh flex 30-90°",
       "015": "Dorsiflexion elb 90°", "017": "Dorsiflexion elb 0°",
       "016": "Wrist flex/ext elb 90°", "018": "Wrist flex/ext elb 0°"}


def save(fig, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------- Figure 1: sample flow + sensor layout ----------------
feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
short = feat.short.astype(bool)
m = feat.assess <= 120
n_all, n_main = len(feat), int(m.sum())
n_flat, n_short = int((m & flat).sum()), int((m & short & ~flat).sum())
n_an = int((m & ~flat & ~short).sum())

fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_axes([0, 0, 0.58, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=DARK, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=DARK))


box(0.4, 8.2, 5.6, 1.4, f"REHAB public dataset\n222 assessments x 27 FMA-UE items\n= {n_all:,} item signals", fs=7)
arrow(3.2, 8.2, 3.2, 7.3)
box(0.4, 5.9, 5.6, 1.4, f"Folders 001-120 (assumed baseline, 1 per patient)\n{n_main:,} item-assessments")
box(6.6, 8.2, 3.2, 1.4, "Folders 121-222\n(102 assessments)\nsensitivity / external", fc="#f2f2f2", fs=7)
arrow(6.0, 8.9, 6.6, 8.9)
arrow(3.2, 5.9, 3.2, 3.1)
box(6.6, 4.0, 3.2, 1.6, f"Excluded\nconstant signal: {n_flat} (all score 0)\nshorter than 0.5 s: {n_short}",
    fc="#f2f2f2", fs=7)
arrow(3.2, 4.8, 6.6, 4.8)
box(0.4, 1.7, 5.6, 1.4, f"Analysis set: {n_an:,} item-assessments", bold=True)
box(0.4, 0.1, 2.7, 1.2, "A: 8 within-patient\nposture pairs", fs=7)
box(3.3, 0.1, 2.7, 1.2, "B: 27 item-wise\nCV models", fs=7)
arrow(1.75, 1.7, 1.75, 1.3); arrow(4.65, 1.7, 4.65, 1.3)
ax.text(-0.3, 9.85, "a", fontsize=11, fontweight="bold", va="top")

bx = fig.add_axes([0.6, 0.05, 0.4, 0.9]); bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off")
bx.text(0.1, 9.85, "b", fontsize=11, fontweight="bold", va="top")
sh, el, wr, hd = (2.0, 8.0), (2.4, 5.0), (5.4, 4.2), (6.6, 4.0)
for p, q in [(sh, el), (el, wr)]:
    bx.plot([p[0], q[0]], [p[1], q[1]], color=GREY, lw=7, solid_capstyle="round")
bx.add_patch(Circle(sh, 0.45, color=GREY))
bx.add_patch(FancyBboxPatch((wr[0] - 0.1, wr[1] - 0.55), 1.6, 1.1, boxstyle="round,pad=0.1",
                            fc="#d9d9d9", ec="none"))
for (x, y), lbl, txt in [((2.2, 6.5), "S2", "Upper-arm IMU (S2)\nitems 000-014, 026"),
                         ((3.9, 4.6), "S1", "Forearm IMU (S1)\nitems 000-014, 026"),
                         ((6.2, 4.2), "S5", "Sensor glove (S5)\n5 flex sensors + hand IMU\nitems 015-025")]:
    bx.add_patch(FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5, boxstyle="round,pad=0.05", fc=BLUE, ec="none"))
    bx.text(x, y, lbl, color="white", ha="center", va="center", fontsize=6.5, fontweight="bold")
ann = [((2.2, 6.5), (4.6, 8.2), "Upper-arm IMU (S2)\nitems 000-014, 026"),
       ((3.9, 4.6), (4.4, 2.4), "Forearm IMU (S1)\nitems 000-014, 026"),
       ((6.2, 4.2), (6.8, 6.6), "Sensor glove (S5)\n5 flex sensors + hand IMU\nitems 015-025")]
for (x, y), (tx, ty), txt in ann:
    bx.annotate(txt, xy=(x + 0.35, y), xytext=(tx, ty), fontsize=6.8, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=DARK))
bx.text(0.3, 0.6, "~50 Hz; one segmented repetition per item;\nlabel = ceiling of the mean of two therapists",
        fontsize=6.5, color=DARK)
save(fig, "Figure1_flow_sensors")

# ---------------- Figure 2: posture pairs ----------------
long = pd.read_csv(res / "A_paired_metrics_long.csv")
W = long[(long.metric == "M1") & (long.assess <= 120)].pivot(index="assess", columns="move", values="value")
W.columns = [f"{c:03d}" for c in W.columns]
rob = pd.read_csv(res / "A_paired_robust.csv")
main = rob[(rob.metric == "M1") & (rob["sample"] == "001-120")].set_index("pair")


def rrb(a, b):
    d = b - a
    d = d[d != 0]
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


rng = np.random.default_rng(20261123)
ci = {}
for a, b in PAIRS:
    ok = W[[a, b]].dropna()
    ok = ok[(ok[a] > 0) | (ok[b] > 0)].values
    boots = [rrb(*ok[rng.integers(0, len(ok), len(ok))].T) for _ in range(2000)]
    ci[(a, b)] = (rrb(ok[:, 0], ok[:, 1]), *np.percentile(boots, [2.5, 97.5]))
pd.DataFrame([dict(pair=f"{a}->{b}", r=v[0], lo=v[1], hi=v[2]) for (a, b), v in ci.items()]).to_csv(
    res / "A_rrb_bootstrap_ci.csv", index=False)

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25], hspace=0.75, wspace=0.45,
                      left=0.08, right=0.97)
for k, (a, b) in enumerate(PAIRS):
    axp = fig.add_subplot(gs[k // 4, k % 4])
    ok = W[[a, b]].dropna()
    ok = ok[(ok[a] > 0) | (ok[b] > 0)]
    for va, vb in ok.values:
        axp.plot([0, 1], [va, vb], color=GREY, lw=0.4, alpha=0.35)
    m_a, m_d = ok[a].median(), (ok[b] - ok[a]).median()
    axp.plot([0, 1], [m_a, m_a + m_d], color=BLUE, lw=2, marker="o", ms=4)
    axp.text(0.5, 0.97, f"{(ok[b] < ok[a]).mean():.0%} decreased", transform=axp.transAxes,
             ha="center", va="top", fontsize=6, color=DARK)
    axp.set_xticks([0, 1]); axp.set_xticklabels([SHORT[a], SHORT[b]], fontsize=5.6)
    axp.set_xlim(-0.25, 1.25); axp.set_ylim(0, 185)
    p = main.loc[f"{a}->{b}", "p_holm"]
    axp.set_title(f"r = {ci[(a, b)][0]:+.2f}, Holm p = {p:.2g}", fontsize=6.6)
    if k % 4 == 0:
        axp.set_ylabel("Range of motion (deg, sensor axis)", fontsize=6.6)
    axp.tick_params(axis="y", labelsize=6)
    if k == 0:
        axp.text(-0.55, 1.18, "a", transform=axp.transAxes, fontsize=11, fontweight="bold")
fig.text(0.08, 0.355, "Grey: individual patients. Blue: median of the easier posture -> plus the median "
         "within-patient change.", fontsize=6, color=DARK)
axf = fig.add_axes([0.36, 0.05, 0.56, 0.27])
ys = np.arange(len(PAIRS))[::-1]
for y, (a, b) in zip(ys, PAIRS):
    r, lo, hi = ci[(a, b)]
    sig = main.loc[f"{a}->{b}", "p_holm"] < 0.05
    c = BLUE if sig else GREY
    axf.plot([lo, hi], [y, y], color=c, lw=1.2)
    axf.plot(r, y, "o", color=c, ms=5)
    n_sig = int((rob[rob.pair == f"{a}->{b}"].p_holm < 0.05).sum())
    axf.text(0.62, y, f"{n_sig}/8", va="center", fontsize=6.6, color=DARK)
axf.axvline(0, color=DARK, lw=0.6)
axf.set_yticks(ys); axf.set_yticklabels([f"{ONE[a]}  ->  {ONE[b]}" for a, b in PAIRS], fontsize=6.4)
axf.set_xlim(-0.8, 0.7)
axf.set_xlabel("Matched rank-biserial r (bootstrap 95% CI); negative = smaller motion in the harder posture",
               fontsize=6.8)
axf.text(0.62, len(PAIRS) - 0.3, "sig. in\n8 analyses", fontsize=6, va="bottom")
axf.text(-0.5, 1.05, "b", transform=axf.transAxes, fontsize=11, fontweight="bold")
save(fig, "Figure2_posture_pairs")

# ---------------- Figure 3: boundary AUCs ----------------
I = pd.read_csv(res / "B_item_auc.csv", dtype={"item": str}).sort_values("cv_auc_1v2", ascending=False)
fig, ax = plt.subplots(figsize=(4.8, 5.6))
y = np.arange(len(I))
ax.hlines(y, I.cv_auc_1v2, I.cv_auc_0v12, color="#cccccc", lw=1.4, zorder=1)
ax.scatter(I.cv_auc_0v12, y, s=16, color=GREY, zorder=2, label="Score 0 vs 1-2")
ax.scatter(I.cv_auc_1v2, y, s=18, color=BLUE, zorder=3, label="Score 1 vs 2")
ax.scatter(I.ext_auc_1v2, y, s=14, facecolors="none", edgecolors=BLUE, lw=0.7, zorder=3,
           label="Score 1 vs 2, external (121-222)")
ax.axvline(I.cv_auc_1v2.median(), color=BLUE, ls=":", lw=0.8)
ax.axvline(I.cv_auc_0v12.median(), color=GREY, ls=":", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([f"{i} {n}" for i, n in zip(I.item, I.name)], fontsize=6.4)
ax.set_xlim(0.5, 1.01); ax.set_ylim(-1, len(I))
ax.set_xlabel("Cross-validated AUC (0.5 = chance)")
ax.legend(loc="lower left", fontsize=6.2, frameon=False, bbox_to_anchor=(0, 1.0), ncol=1)
ax.text(0.70, len(I) + 1.2, f"medians: 0 vs 1-2 = {I.cv_auc_0v12.median():.3f}; "
        f"1 vs 2 = {I.cv_auc_1v2.median():.3f}\n1 vs 2 lower in "
        f"{(I.cv_auc_1v2 < I.cv_auc_0v12).sum()}/{len(I)} items (Wilcoxon p = 5.6e-6)",
        fontsize=6.2, va="bottom", color=DARK, transform=ax.transData, clip_on=False)
save(fig, "Figure3_boundary_auc")

# ---------------- Figure 4: quality ORs ----------------
Q = pd.read_csv(res / "B_quality_features.csv")
fams = [("All IMU + wrist items", list(range(20)) + [26]),
        ("Wrist (015-019)", [15, 16, 17, 18, 19]), ("Forearm rotation", [5, 8, 11, 12, 13, 14]),
        ("Elbow (004, 007)", [4, 7]), ("Hand-to-spine, finger-nose", [9, 26]),
        ("Shoulder", [0, 1, 2, 3, 6, 10])]
rows = []
for lbl, ids in fams:
    s = Q[Q.move.isin(ids)].dropna(subset=["magnitude_z", "coupling_z", "peaks_ps_z"])
    f = smf.logit("y2 ~ magnitude_z + coupling_z + peaks_ps_z + C(move)", data=s).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": s.assess.values})
    for t, name in [("coupling_z", "Inter-joint coupling"), ("peaks_ps_z", "Speed peaks per second")]:
        lo, hi = np.exp(f.conf_int().loc[t])
        rows.append(dict(family=lbl, term=name, OR=np.exp(f.params[t]), lo=lo, hi=hi, p=f.pvalues[t], n=len(s)))
R = pd.DataFrame(rows)
R.to_csv(res / "B_quality_family_or.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True,
                        gridspec_kw=dict(wspace=0.55, left=0.25, right=0.88))
for axq, term, tag in zip(axs, ["Inter-joint coupling", "Speed peaks per second"], ["a", "b"]):
    s = R[R.term == term].reset_index(drop=True)
    ys = np.arange(len(s))[::-1]
    for yv, (_, r) in zip(ys, s.iterrows()):
        c = BLUE if r.p < 0.05 else GREY
        axq.plot([r.lo, r.hi], [yv, yv], color=c, lw=1.2)
        axq.plot(r.OR, yv, "s" if yv == ys[0] else "o", color=c, ms=5)
        axq.text(1.03, yv, f"{r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})", va="center", fontsize=6.2,
                 transform=axq.get_yaxis_transform())
    axq.axvline(1, color=DARK, lw=0.6)
    axq.set_xscale("log"); axq.set_xlim(0.15, 3)
    axq.set_xticks([0.2, 0.5, 1, 2]); axq.set_xticklabels(["0.2", "0.5", "1", "2"])
    axq.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axq.set_yticks(ys); axq.set_yticklabels([f"{f} (n={n})" for f, n in zip(s.family, s.n)], fontsize=6.6)
    axq.set_title(f"{term}\n(OR per SD for score 2 vs 1, adjusted for magnitude)", fontsize=7)
    axq.text(-0.08 if tag == "b" else -0.75, 1.08, tag, transform=axq.transAxes, fontsize=11, fontweight="bold")
axs[1].tick_params(labelleft=False)
fig.text(0.5, -0.04, "Odds ratio (log scale; < 1 = lower odds of score 2). Item fixed effects, "
         "cluster-robust 95% CI by assessment. Exploratory analysis.", ha="center", fontsize=6.4)
save(fig, "Figure4_quality_or")
