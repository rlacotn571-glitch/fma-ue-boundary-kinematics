"""Manuscript figures 1-4, design v2 (all 222 assessments, no folder-order assumption).

Usage: python -I 13_figures_v2.py <results dir> <figures out dir>
Inputs: features_v1.csv, A_paired_metrics_long.csv, v2_A_pairs.csv, v2_B_item_auc.csv, v2_H3_features.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.stats import norm
import statsmodels.formula.api as smf

res, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42})
BLUE, GREY, DARK = "#2166ac", "#9e9e9e", "#333333"
N_ALL, N_EFF = 222, 120
PAIRS = ["011->012", "016->018", "011->013", "015->017", "014->013", "011->014", "008->011", "005->011"]
SHORT = {"005": "Supination\n(flexor synergy)", "008": "Pronation\n(extensor synergy)",
         "011": "Pro/sup\nsh 0°, elbow 90°", "012": "Pronation\nsh abd 90°, elbow 0°",
         "013": "Pro/sup\nsh flex 90-180°", "014": "Pro/sup\nsh flex 30-90°",
         "015": "Dorsiflexion\nelbow 90°", "017": "Dorsiflexion\nelbow 0°",
         "016": "Wrist flex/ext\nelbow 90°", "018": "Wrist flex/ext\nelbow 0°"}
ONE = {"005": "Supination (flexor syn.)", "008": "Pronation (extensor syn.)",
       "011": "Pro/sup sh 0°/elb 90°", "012": "Pronation sh abd 90°/elb 0°",
       "013": "Pro/sup sh flex 90-180°", "014": "Pro/sup sh flex 30-90°",
       "015": "Dorsiflexion elb 90°", "017": "Dorsiflexion elb 0°",
       "016": "Wrist flex/ext elb 90°", "018": "Wrist flex/ext elb 0°"}


def save(fig, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight"); fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("saved", name)


# ---------- Figure 1 ----------
feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
short = feat.short.astype(bool) & ~flat
n_all, n_flat, n_short = len(feat), int(flat.sum()), int(short.sum())
n_an = n_all - n_flat - n_short
fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_axes([0, 0, 0.58, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def box(x, y, w, h, text, fc="white", fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=DARK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=DARK))


box(0.4, 8.1, 5.6, 1.6, "REHAB public dataset: 120 patients\n222 assessments (120 admission + 102 discharge;\n"
    "order and patient links not released)")
arrow(3.2, 8.1, 3.2, 7.2)
box(0.4, 5.8, 5.6, 1.4, f"27 FMA-UE items per assessment\n{n_all:,} item-assessments")
arrow(3.2, 5.8, 3.2, 3.1)
box(6.6, 4.0, 3.2, 1.6, f"Excluded\nconstant signal: {n_flat} (all score 0)\nshorter than 0.5 s: {n_short}", fc="#f2f2f2", fs=6.8)
arrow(3.2, 4.8, 6.6, 4.8)
box(0.4, 1.7, 5.6, 1.4, f"Analysis set: {n_an:,} item-assessments\nunit = assessment; worst-case n_eff = 120", bold=True)
box(0.4, 0.1, 2.7, 1.2, "A: 8 within-assessment\nposture pairs", fs=6.8)
box(3.3, 0.1, 2.7, 1.2, "B: 27 item-wise\nboundary analyses", fs=6.8)
arrow(1.75, 1.7, 1.75, 1.3); arrow(4.65, 1.7, 4.65, 1.3)
box(6.6, 7.9, 3.2, 1.8, "Inference without IDs:\nm-out-of-n bootstrap (m = 120)\nworst-case p (n_eff = 120)\n"
    "leakage-free univariate check", fc="#f2f2f2", fs=6.5)
ax.text(-0.3, 9.95, "a", fontsize=11, fontweight="bold", va="top")
bx = fig.add_axes([0.6, 0.05, 0.4, 0.9]); bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off")
bx.text(0.1, 9.85, "b", fontsize=11, fontweight="bold", va="top")
sh, el, wr = (2.0, 8.0), (2.4, 5.0), (5.4, 4.2)
for p, q in [(sh, el), (el, wr)]:
    bx.plot([p[0], q[0]], [p[1], q[1]], color=GREY, lw=7, solid_capstyle="round")
bx.add_patch(Circle(sh, 0.45, color=GREY))
bx.add_patch(FancyBboxPatch((wr[0] - 0.1, wr[1] - 0.55), 1.6, 1.1, boxstyle="round,pad=0.1", fc="#d9d9d9", ec="none"))
for (x, y), lbl, (tx, ty), txt in [((2.2, 6.5), "S2", (4.6, 8.2), "Upper-arm IMU (S2)\nitems 000-014, 026"),
                                   ((3.9, 4.6), "S1", (4.4, 2.4), "Forearm IMU (S1)\nitems 000-014, 026"),
                                   ((6.2, 4.2), "S5", (6.8, 6.6), "Sensor glove (S5)\n5 flex sensors + hand IMU\nitems 015-025")]:
    bx.add_patch(FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5, boxstyle="round,pad=0.05", fc=BLUE, ec="none"))
    bx.text(x, y, lbl, color="white", ha="center", va="center", fontsize=6.5, fontweight="bold")
    bx.annotate(txt, xy=(x + 0.35, y), xytext=(tx, ty), fontsize=6.8, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=DARK))
bx.text(0.3, 0.6, "~50 Hz; one segmented repetition per item;\nlabel = ceiling of the mean of two therapists",
        fontsize=6.5, color=DARK)
save(fig, "Figure1_flow_sensors")

# ---------- Figure 2 ----------
long = pd.read_csv(res / "A_paired_metrics_long.csv")
W = long[long.metric == "M1"].pivot(index="assess", columns="move", values="value")
W.columns = [f"{c:03d}" for c in W.columns]
A = pd.read_csv(res / "v2_A_pairs.csv")
main = A[(A.metric == "M1") & (A["sample"] == "all 222")].set_index("pair")
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25], hspace=0.75, wspace=0.45, left=0.08, right=0.97)
for k, pr in enumerate(PAIRS):
    a, b = pr.split("->")
    axp = fig.add_subplot(gs[k // 4, k % 4])
    ok = W[[a, b]].dropna(); ok = ok[(ok[a] > 0) | (ok[b] > 0)]
    for va, vb in ok.values:
        axp.plot([0, 1], [va, vb], color=GREY, lw=0.35, alpha=0.3)
    m_a, m_d = ok[a].median(), (ok[b] - ok[a]).median()
    axp.plot([0, 1], [m_a, m_a + m_d], color=BLUE, lw=2, marker="o", ms=4)
    axp.text(0.5, 0.97, f"{(ok[b] < ok[a]).mean():.0%} decreased (n={len(ok)})", transform=axp.transAxes,
             ha="center", va="top", fontsize=5.8, color=DARK)
    axp.set_xticks([0, 1]); axp.set_xticklabels([SHORT[a], SHORT[b]], fontsize=5.6)
    axp.set_xlim(-0.25, 1.25); axp.set_ylim(0, 185)
    axp.set_title(f"r = {main.loc[pr, 'r']:+.2f}, Holm p = {main.loc[pr, 'p_holm']:.2g}", fontsize=6.6)
    if k % 4 == 0:
        axp.set_ylabel("Range of motion (deg, sensor axis)", fontsize=6.6)
    axp.tick_params(axis="y", labelsize=6)
    if k == 0:
        axp.text(-0.55, 1.18, "a", transform=axp.transAxes, fontsize=11, fontweight="bold")
fig.text(0.08, 0.355, "Grey: individual assessments. Blue: median of the easier posture plus the median "
         "within-assessment change.", fontsize=6, color=DARK)
axf = fig.add_axes([0.36, 0.05, 0.56, 0.27])
ys = np.arange(len(PAIRS))[::-1]
for y, pr in zip(ys, PAIRS):
    a, b = pr.split("->"); r = main.loc[pr]
    c = BLUE if r.p_holm_worst < 0.05 else (DARK if r.p_holm < 0.05 else GREY)
    axf.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=1.2); axf.plot(r.r, y, "o", color=c, ms=5)
    s = A[A.pair == pr]
    axf.text(0.62, y, f"{int((s.p_holm < 0.05).sum())}/12", va="center", fontsize=6.6, color=DARK)
axf.axvline(0, color=DARK, lw=0.6)
axf.set_yticks(ys); axf.set_yticklabels([f"{ONE[p.split('->')[0]]}  ->  {ONE[p.split('->')[1]]}" for p in PAIRS], fontsize=6.4)
axf.set_xlim(-0.8, 0.7)
axf.set_xlabel("Matched rank-biserial r, 95% CI from m-out-of-n bootstrap (m = 120); negative = less motion "
               "in the harder posture", fontsize=6.4)
axf.text(0.62, len(PAIRS) - 0.3, "significant in\n12 analyses", fontsize=6, va="bottom")
axf.text(-0.5, 1.05, "b", transform=axf.transAxes, fontsize=11, fontweight="bold")
axf.text(-0.8, -2.2, "Blue: Holm p < 0.05 under worst-case dependence (n_eff = 120). Black: Holm p < 0.05 "
         "with n = 222 only. Grey: not significant.", fontsize=5.8, color=DARK)
save(fig, "Figure2_posture_pairs")

# ---------- Figure 3 ----------
names = ["sh elevation", "sh retraction", "sh abduction", "sh ext rot", "elbow flex", "supination", "sh add/IR",
         "elbow ext", "pronation", "hand to lumbar", "sh flex 90", "pro/sup e90", "pronation abd90",
         "pro/sup flex90-180", "pro/sup flex30-90", "wrist dorsi e90", "wrist flex/ext e90", "wrist dorsi e0",
         "wrist flex/ext e0", "wrist circum", "mass flexion", "mass extension", "thumb add", "pinch",
         "cylinder", "sphere", "finger-nose"]
I = pd.read_csv(res / "v2_B_item_auc.csv").sort_values("cv_1v2", ascending=False)
fig, ax = plt.subplots(figsize=(4.8, 5.6))
y = np.arange(len(I))
ax.hlines(y, I.cv_1v2, I.cv_0v12, color="#cccccc", lw=1.4, zorder=1)
ax.scatter(I.cv_0v12, y, s=16, color=GREY, zorder=2, label="Score 0 vs 1-2 (CV model)")
ax.scatter(I.cv_1v2, y, s=18, color=BLUE, zorder=3, label="Score 1 vs 2 (CV model)")
ax.scatter(I.uni_1v2, y, s=14, facecolors="none", edgecolors=BLUE, lw=0.7, zorder=3,
           label="Score 1 vs 2 (single pre-specified feature, no fitting)")
ax.axvline(I.cv_1v2.median(), color=BLUE, ls=":", lw=0.8); ax.axvline(I.cv_0v12.median(), color=GREY, ls=":", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([f"{int(i):03d} {names[int(i)]}" for i in I["item"]], fontsize=6.4)
ax.set_xlim(0.4, 1.01); ax.set_ylim(-1, len(I))
ax.set_xlabel("AUC (0.5 = chance); all 222 assessments")
ax.legend(loc="lower left", fontsize=6.0, frameon=False, bbox_to_anchor=(0, 1.0))
ax.text(0.40, -4.4, f"CV medians: 0 vs 1-2 = {I.cv_0v12.median():.3f}; 1 vs 2 = {I.cv_1v2.median():.3f} "
        f"(lower in {(I.cv_1v2 < I.cv_0v12).sum()}/27 items)", fontsize=6.0, color=DARK, clip_on=False)
save(fig, "Figure3_boundary_auc")

# ---------- Figure 4 ----------
Q = pd.read_csv(res / "v2_H3_features.csv")
infl = np.sqrt(N_ALL / N_EFF)
fams = [("All coupling-defined items", list(range(20)) + [26]), ("Wrist (015-019)", [15, 16, 17, 18, 19]),
        ("Forearm rotation", [5, 8, 11, 12, 13, 14]), ("Elbow (004, 007)", [4, 7]),
        ("Hand-to-spine, finger-nose", [9, 26]), ("Shoulder", [0, 1, 2, 3, 6, 10])]
rows = []
for lbl, ids in fams:
    s = Q[Q.move.isin(ids)].dropna(subset=["magnitude_z", "coupling_z", "peaks_ps_z"])
    f = smf.logit("y2 ~ magnitude_z + coupling_z + peaks_ps_z + C(move)", data=s).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": s.assess.values})
    for t, nm in [("coupling_z", "Inter-joint coupling"), ("peaks_ps_z", "Speed peaks per second")]:
        b, se = f.params[t], f.bse[t] * infl
        rows.append(dict(family=lbl, term=nm, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se),
                         p=2 * norm.sf(abs(b / se)), n=len(s)))
R = pd.DataFrame(rows); R.to_csv(res / "v2_H3_family_or.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw=dict(wspace=0.55, left=0.25, right=0.88))
for axq, term, tag in zip(axs, ["Inter-joint coupling", "Speed peaks per second"], ["a", "b"]):
    s = R[R.term == term].reset_index(drop=True); ys = np.arange(len(s))[::-1]
    for yv, (_, r) in zip(ys, s.iterrows()):
        c = BLUE if r.p < 0.05 else GREY
        axq.plot([r.lo, r.hi], [yv, yv], color=c, lw=1.2); axq.plot(r.OR, yv, "s" if yv == ys[0] else "o", color=c, ms=5)
        axq.text(1.03, yv, f"{r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})", va="center", fontsize=6.2, transform=axq.get_yaxis_transform())
    axq.axvline(1, color=DARK, lw=0.6); axq.set_xscale("log"); axq.set_xlim(0.15, 3)
    axq.set_xticks([0.2, 0.5, 1, 2]); axq.set_xticklabels(["0.2", "0.5", "1", "2"])
    axq.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axq.set_yticks(ys); axq.set_yticklabels([f"{f} (n={n})" for f, n in zip(s.family, s.n)], fontsize=6.6)
    axq.set_title(f"{term}\n(OR per SD, score 2 vs 1, adjusted for magnitude)", fontsize=7)
    axq.text(-0.08 if tag == "b" else -0.75, 1.08, tag, transform=axq.transAxes, fontsize=11, fontweight="bold")
axs[1].tick_params(labelleft=False)
fig.text(0.5, -0.05, "Odds ratio (log scale; < 1 = lower odds of score 2). Item fixed effects; 95% CI with "
         "cluster-robust SE inflated for worst-case patient dependence (x sqrt(222/120)). Exploratory.",
         ha="center", fontsize=6.2)
save(fig, "Figure4_quality_or")
