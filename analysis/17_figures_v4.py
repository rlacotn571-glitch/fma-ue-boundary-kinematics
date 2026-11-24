"""Manuscript figures, v4 (after second review): Figure 2 and 4 use v4 results; Figures 1 and 3 unchanged from v3.

Usage: python -I 15_figures_v3.py <results dir> <figures out dir>
Inputs: features_v1.csv, v5_A_metrics.csv, v5_A_pairs.csv, v5_B_item_auc.csv, v6_H3_or.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

res, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42})
BLUE, GREY, DARK = "#2166ac", "#9e9e9e", "#333333"
NAMES = ["sh elevation", "sh retraction", "sh abduction", "sh ext rot", "elbow flex", "supination", "sh add/IR",
         "elbow ext", "pronation", "hand to lumbar", "sh flex 0-90", "pro/sup elbow 90", "sh abduction 0-90 (isol.)",
         "sh flex 90-180 (isol.)", "pro/sup elbow 0 (isol.)", "wrist stability elb 90", "wrist flex/ext elb 90",
         "wrist stability elb 0", "wrist flex/ext elb 0", "wrist circumduction", "mass flexion", "mass extension",
         "grasp a (thumb add)", "grasp b (pinch)", "grasp c (cylinder)", "grasp d (sphere)", "finger-to-nose"]


def save(fig, name, max_w=6.35):
    # JNER: final width <= 170 mm. Shrink the canvas (fonts unchanged) until the tight bounding box fits.
    for _ in range(12):
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if w <= max_w:
            break
        W, Hh = fig.get_size_inches(); fig.set_size_inches(W - (w - max_w) * 1.05, Hh)
    fig.savefig(out / f"{name}.png", bbox_inches="tight"); fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("saved", name)


# ---------- Figure 1 ----------
feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
short = feat.short.astype(bool) & ~flat
n_all, n_flat, n_short = len(feat), int(flat.sum()), int(short.sum())
fig = plt.figure(figsize=(6.4, 3.5))
ax = fig.add_axes([0, 0, 0.58, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
def box(x, y, w, h, t, fc="white", fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=DARK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=DARK))
box(0.4, 8.1, 5.6, 1.6, "REHAB: 120 patients, 2 centres (RCT)\n229 sessions - 7 erroneous = 222 assessments\n(visit and patient links not released)")
arrow(3.2, 8.1, 3.2, 7.2)
box(0.4, 5.8, 5.6, 1.4, f"27 FMA-UE items per assessment\n{n_all:,} item-assessments")
arrow(3.2, 5.8, 3.2, 3.1)
box(6.6, 4.0, 3.2, 1.6, f"Excluded\nconstant signal: {n_flat} (all score 0)\nshorter than 0.5 s: {n_short}", fc="#f2f2f2", fs=5.9)
arrow(3.2, 4.8, 6.6, 4.8)
box(0.4, 1.7, 5.6, 1.4, f"Analysed: {n_all - n_flat - n_short:,} item-assessments\nunit = assessment; worst-case n_eff = 112", bold=True)
box(0.4, 0.1, 2.7, 1.2, "Primary: two score\nboundaries, 27 items", fs=5.9)
box(3.3, 0.1, 2.7, 1.2, "Supporting: posture\npairs; determinants", fs=5.9)
arrow(1.75, 1.7, 1.75, 1.3); arrow(4.65, 1.7, 4.65, 1.3)
box(6.6, 7.9, 3.2, 1.8, "Inference without IDs:\nn = 222 and worst-case\nn_eff = 112; assessment-level\nbootstrap; fitting-free\nsingle-feature check", fc="#f2f2f2", fs=5.6)
ax.text(-0.3, 9.95, "a", fontsize=11, fontweight="bold", va="top")
bx = fig.add_axes([0.6, 0.05, 0.4, 0.9]); bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off")
bx.text(0.1, 9.85, "b", fontsize=11, fontweight="bold", va="top")
sh, el, wr = (2.0, 8.0), (2.4, 5.0), (5.4, 4.2)
for p, q in [(sh, el), (el, wr)]:
    bx.plot([p[0], q[0]], [p[1], q[1]], color=GREY, lw=7, solid_capstyle="round")
bx.add_patch(Circle(sh, 0.45, color=GREY))
bx.add_patch(FancyBboxPatch((wr[0] - 0.1, wr[1] - 0.55), 1.6, 1.1, boxstyle="round,pad=0.1", fc="#d9d9d9", ec="none"))
for (x, y), lbl, (tx, ty), txt in [((2.2, 6.5), "S2", (3.8, 8.8), "Upper-arm IMU (S2)\nitems 000-014, 026"),
                                   ((3.9, 4.6), "S1", (4.4, 2.4), "Forearm IMU (S1)\nitems 000-014, 026"),
                                   ((6.2, 4.2), "S5", (5.2, 7.2), "Sensor glove (S5)\nfinger flex sensors + hand IMU\nitems 015-025")]:
    bx.add_patch(FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5, boxstyle="round,pad=0.05", fc=BLUE, ec="none"))
    bx.text(x, y, lbl, color="white", ha="center", va="center", fontsize=6.5, fontweight="bold")
    bx.annotate(txt, xy=(x + 0.35, y), xytext=(tx, ty), fontsize=6.0, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=DARK))
bx.text(0.3, 0.6, "~50 Hz; one segmented repetition per item;\nlabel = ceiling of the mean of two therapists", fontsize=5.9, color=DARK)
save(fig, "Figure1_flow_sensors", max_w=99)

# ---------- Figure 2: score boundaries (primary) ----------
I = pd.read_csv(res / "v5_B_item_auc.csv").sort_values("cv_1v2", ascending=False)
fig, ax = plt.subplots(figsize=(4.9, 5.6)); y = np.arange(len(I))
ax.hlines(y, I.cv_1v2, I.cv_0v1, color="#cccccc", lw=1.4, zorder=1)
ax.scatter(I.cv_0v1, y, s=16, color=GREY, zorder=2, label="Score 0 vs 1 (CV proportional-odds model)")
ax.scatter(I.cv_1v2, y, s=18, color=BLUE, zorder=3, label="Score 1 vs 2 (CV proportional-odds model; bar, worst-case 95% CI)")
CIt = pd.read_csv(res / "v5_B_item_auc_ci.csv").set_index("item").loc[I["item"].values]
ax.hlines(y, CIt.cv_1v2_lo.values, CIt.cv_1v2_hi.values, color="#2166ac", lw=0.7, alpha=0.6, zorder=2)
ax.scatter(I.uni_1v2, y, s=14, facecolors="none", edgecolors=BLUE, lw=0.7, zorder=3, label="Score 1 vs 2 (single pre-specified feature, no fitting)")
ax.axvline(I.cv_1v2.median(), color=BLUE, ls=":", lw=0.8); ax.axvline(I.cv_0v1.median(), color=GREY, ls=":", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([f"{int(i):03d} {NAMES[int(i)]}" for i in I["item"]], fontsize=6.2)
ax.set_xlim(0.3, 1.01); ax.set_ylim(-1, len(I)); ax.set_xlabel("AUC (0.5 = chance)")
ax.legend(loc="lower left", fontsize=6.0, frameon=False, bbox_to_anchor=(0, 1.0))
ax.text(0.30, -4.4, f"Medians: 0 vs 1 = {I.cv_0v1.median():.3f}, 1 vs 2 = {I.cv_1v2.median():.3f}; 1 vs 2 lower in "
        f"{(I.cv_1v2 < I.cv_0v1).sum()}/27 items", fontsize=6.0, color=DARK, clip_on=False)
save(fig, "Figure2_boundary_auc")

# ---------- Figure 3: posture pairs (secondary) ----------
L = pd.read_csv(res / "v5_A_metrics.csv"); A = pd.read_csv(res / "v5_A_pairs.csv")
W = L[L.metric == "M1"].pivot(index="assess", columns="move", values="value")
pairs = [(11, 14, "Pro/sup\nelb 90°, sh 0°", "Pro/sup\nelb 0°, sh 30-90°"),
         (15, 17, "Wrist stability\nelbow 90°", "Wrist stability\nelbow 0°"),
         (16, 18, "Wrist flex/ext\nelbow 90°", "Wrist flex/ext\nelbow 0°")]
fig = plt.figure(figsize=(6.7, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.7, wspace=0.4, left=0.08, right=0.97)
for k, (a, b, la, lb) in enumerate(pairs):
    axp = fig.add_subplot(gs[0, k]); ok = W[[a, b]].dropna(); ok = ok[(ok[a] > 0) | (ok[b] > 0)]
    for va, vb in ok.values:
        axp.plot([0, 1], [va, vb], color=GREY, lw=0.35, alpha=0.3)
    m_a, m_d = ok[a].median(), (ok[b] - ok[a]).median()
    axp.plot([0, 1], [m_a, m_a + m_d], color=BLUE, lw=2, marker="o", ms=4)
    r = A[(A.metric == "M1") & (A["sample"] == "all") & (A.pair == f"{a:03d}->{b:03d}")].iloc[0]
    fp = lambda p: f"{p:.3f}" if p < 0.01 else f"{p:.2f}" if p >= 0.05 else f"{p:.3f}"
    axp.set_title(f"r = {r.r:+.2f}, Holm p = {fp(r.p_holm)}\n(worst case {fp(r.p_holm_worst)}); {r.frac_drop:.0%} decreased", fontsize=6.6)
    axp.set_xticks([0, 1]); axp.set_xticklabels([la, lb], fontsize=5.6); axp.set_xlim(-0.25, 1.25); axp.set_ylim(0, 185)
    if k == 0:
        axp.set_ylabel("Range (deg, sensor channel)", fontsize=6.6)
        axp.text(-0.4, 1.25, "a", transform=axp.transAxes, fontsize=11, fontweight="bold")
axf = fig.add_subplot(gs[1, :])
mlab = {"M1": "full range", "M2": "robust range", "M3": "peak speed", "M5": "rotation path"}
ylabels, yv = [], 0
for a, b, la, lb in pairs[::-1]:
    for mtr in ["M5", "M3", "M2", "M1"]:
        r = A[(A.metric == mtr) & (A["sample"] == "all") & (A.pair == f"{a:03d}->{b:03d}")].iloc[0]
        c = BLUE if r.p_holm_worst < 0.05 else (DARK if r.p_holm < 0.05 else GREY)
        axf.plot([r.ci_lo, r.ci_hi], [yv, yv], color=c, lw=1.1); axf.plot(r.r, yv, "o", color=c, ms=4)
        ylabels.append(f"{a:03d}->{b:03d}  {mlab[mtr]}"); yv += 1
    yv += 0.6
ys = [i + 0.6 * (i // 4) for i in range(len(ylabels))]
axf.set_yticks(ys); axf.set_yticklabels(ylabels, fontsize=6)
axf.axvline(0, color=DARK, lw=0.6); axf.set_xlim(-0.65, 0.45)
axf.set_xlabel("Matched rank-biserial r (95% CI, bootstrap of 112 of 222 assessments); negative = less motion with the elbow extended", fontsize=6.4)
axf.text(-0.12, 1.05, "b", transform=axf.transAxes, fontsize=11, fontweight="bold")
axf.text(0.47, ys[-1], "Blue: Holm p<0.05 at worst case\nBlack: at n = 222 only\nGrey: n.s.", fontsize=5.8, va="top")
save(fig, "Figure3_posture_pairs")

# ---------- Figure 4: exploratory determinants of 2 vs 1 ----------
H = pd.read_csv(res / "v6_H3_or.csv")
fams = ["pooled", "wrist", "forearm", "elbow", "009/026", "shoulder"]
flab = {"pooled": "All items (pooled)", "wrist": "Wrist (015-019)", "forearm": "Forearm (005,008,011,014)",
        "elbow": "Elbow (004, 007)", "009/026": "Hand-to-spine, finger-nose", "shoulder": "Shoulder (000-003,006,010,012,013)"}
tlab = {"tgt": "Target-joint range", "non": "Non-target-joint range", "peaks": "Speed peaks per second"}
fig, axs = plt.subplots(1, 3, figsize=(6.7, 2.8), sharey=True, gridspec_kw=dict(wspace=0.15, left=0.22, right=0.98))
for axq, t, tag in zip(axs, ["tgt", "non", "peaks"], "abc"):
    s = H[H.term == t].set_index("family").loc[fams].reset_index(); ys = np.arange(len(s))[::-1]
    for yv, (_, r) in zip(ys, s.iterrows()):
        c = BLUE if r.p < 0.05 else GREY
        axq.plot([r.lo, r.hi], [yv, yv], color=c, lw=1.1); axq.plot(r.OR, yv, "s" if r.family == "pooled" else "o", color=c, ms=4.5)
    axq.axvline(1, color=DARK, lw=0.6); axq.set_xscale("log"); axq.set_xlim(0.2, 10)
    axq.set_xticks([0.5, 1, 2, 4]); axq.set_xticklabels(["0.5", "1", "2", "4"])
    axq.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axq.set_title(tlab[t], fontsize=7); axq.text(-0.05, 1.08, tag, transform=axq.transAxes, fontsize=11, fontweight="bold")
axs[0].set_yticks(np.arange(len(fams))[::-1]); axs[0].set_yticklabels([f"{flab[f]}" for f in fams], fontsize=6.4)
fig.text(0.6, -0.06, "OR per within-item SD for score 2 vs 1 (log scale), mutually adjusted, item fixed effects.\n95% CI with "
         "cluster-robust SE x sqrt(222/112). Exploratory.", ha="center", fontsize=6.2)
save(fig, "Figure4_determinants")











