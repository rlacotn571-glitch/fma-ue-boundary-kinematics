"""Manuscript figures, v5 (journal-style redesign; same results as v4/v6).

Usage: python -I 23_figures_v5.py <results dir> <figures out dir>
Inputs: features_v1.csv, v5_A_metrics.csv, v5_A_pairs.csv, v5_B_item_auc.csv, v5_B_item_auc_ci.csv,
        v6_H3_or.csv, v5_summary.txt, v6_summary.txt
Style: Arial, Okabe-Ito colour-blind-safe palette, width <= 170 mm (JNER), 300 dpi PNG + vector PDF.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Ellipse
from matplotlib.lines import Line2D

res, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 0.9, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222", "text.color": "#111111"})
B01, B12 = "#0072B2", "#D55E00"         # boundary 0 vs 1, boundary 1 vs 2 (Okabe-Ito)
SIG, NS, INK, ROW = "#0072B2", "#A6A6A6", "#222222", "#F4F4F4"
NAMES = ["Shoulder elevation", "Shoulder retraction", "Shoulder abduction", "Shoulder external rotation",
         "Elbow flexion", "Forearm supination", "Shoulder adduction/IR", "Elbow extension", "Forearm pronation",
         "Hand to lumbar spine", "Shoulder flexion 0-90°", "Pronation-supination, elbow 90°",
         "Shoulder abduction 0-90° (isolated)", "Shoulder flexion 90-180° (isolated)",
         "Pronation-supination, elbow 0° (isolated)", "Wrist stability, elbow 90°", "Wrist flexion-extension, elbow 90°",
         "Wrist stability, elbow 0°", "Wrist flexion-extension, elbow 0°", "Wrist circumduction", "Mass flexion",
         "Mass extension", "Grasp a (thumb adduction)", "Grasp b (thumb pinch)", "Grasp c (cylinder)",
         "Grasp d (sphere)", "Finger-to-nose"]


def tag(ax, letter, x=-0.02, y=1.02, fig_coords=False):
    kw = dict(fontsize=10, fontweight="bold", va="bottom", ha="right")
    (ax.text(x, y, letter, transform=ax.transAxes, **kw) if not fig_coords else ax.text(x, y, letter, **kw))


def save(fig, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.canvas.draw(); w = fig.get_tightbbox(fig.canvas.get_renderer()).width / MM
    plt.close(fig); print(f"saved {name} (width {w:.0f} mm)")


def grab(txt, pat):
    m = re.search(pat, txt); assert m, pat
    return [float(g) for g in m.groups()]


s5 = (res / "v5_summary.txt").read_text(encoding="utf-8")
s6 = (res / "v6_summary.txt").read_text(encoding="utf-8")

# =====================================================================  Figure 1
feat = pd.read_csv(res / "features_v1.csv")
flat = feat[["S1_flat", "S2_flat", "g_flat"]].fillna(False).astype(bool).any(axis=1)
short = feat.short.astype(bool) & ~flat
n_all, n_flat, n_short = len(feat), int(flat.sum()), int(short.sum())
n_an = n_all - n_flat - n_short

fig = plt.figure(figsize=(166 * MM, 82 * MM))
ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, body="", fc="white", ec=INK, tc=INK, bold_title=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=0.7))
    if body:
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top", fontsize=6.8, fontweight="bold" if bold_title else "normal", color=tc)
        ax.text(x + w / 2, y + h - 8.6, body, ha="center", va="top", fontsize=6.0, color=tc, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=6.6, color=tc, linespacing=1.35)


def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.7, color=INK, shrinkA=0, shrinkB=0))


cx, bw = 33.5, 56
box(cx - bw / 2, 78, bw, 21, "REHAB dataset (two-centre RCT)",
    "120 patients with subacute stroke\n229 sessions − 7 erroneous = 222 assessments\nno patient, visit, site or group links released")
arr(cx, 78, cx, 71)
box(cx - bw / 2, 58, bw, 13, "27 FMA-UE items per assessment", f"{n_all:,} item-assessments (scores 0, 1, 2)")
arr(cx, 58, cx, 44)
ax.plot([cx, 67], [51, 51], color=INK, lw=0.7); arr(67, 51, 68, 51)
box(68, 42, 31, 18, "Excluded", f"constant signal: {n_flat}\n(movement not performed)\nshorter than 0.5 s: {n_short}",
    fc="#F2F2F2", ec="#8C8C8C")
box(cx - bw / 2, 31, bw, 13, f"Analysed: {n_an:,} item-assessments",
    "inference at n = 222 and worst-case n$_{eff}$ = 112", fc="#E8F1F8", ec=B01)
arr(cx - 14.5, 31, cx - 14.5, 25); arr(cx + 14.5, 31, cx + 14.5, 25)
box(cx - bw / 2, 1, 27, 24, "Primary analysis",
    "AUC, 0 vs 1 and 1 vs 2,\nfor each of 27 items:\nordinal, binary and\nno-fitting analyses", fc="#FBEADF", ec=B12)
box(cx + 1, 1, 27, 24, "Supporting analyses",
    "same movement with\nthe elbow flexed vs\nextended; kinematic\ndeterminants of score 2")
ax.text(0, 99.5, "a", fontsize=10, fontweight="bold", va="top")

# panel b: arm schematic
bx = fig.add_axes([0.575, 0.02, 0.425, 0.96]); bx.set_xlim(0, 100); bx.set_ylim(0, 100); bx.axis("off"); bx.set_aspect("equal")
bx.text(0, 99.5, "b", fontsize=10, fontweight="bold", va="top")
SK, SKE = "#EDE7E1", "#9A8F86"
bx.add_patch(Ellipse((14, 83), 13, 15.5, fc=SK, ec=SKE, lw=0.7))                         # head
bx.add_patch(Polygon([(2, 34), (4, 66), (12, 73), (16, 73), (28, 68), (30, 34)], closed=True, fc=SK, ec=SKE, lw=0.7))  # trunk
sh, el, wr, hd = np.array([27, 64]), np.array([33, 38]), np.array([62, 30]), np.array([74, 28])


def limb(p, q, w):
    d = q - p; n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * w / 2
    bx.add_patch(Polygon([p + n, q + n * 0.85, q - n * 0.85, p - n], closed=True, fc=SK, ec=SKE, lw=0.7, joinstyle="round"))


limb(sh, el, 9.5); limb(el, wr, 8)
bx.add_patch(Ellipse(tuple(el), 8.6, 8.6, fc=SK, ec="none")); bx.add_patch(Ellipse(tuple(sh), 10, 10, fc=SK, ec="none"))
bx.add_patch(FancyBboxPatch((63, 23.5), 17, 12, boxstyle="round,pad=0,rounding_size=3", fc="#D9D2CB", ec=SKE, lw=0.7))  # hand in glove
for k in range(4):
    bx.plot([80, 86.5], [33 - 2.6 * k, 33.6 - 2.9 * k], color=SKE, lw=2.6, solid_capstyle="round")
for k in range(4):
    bx.plot([80, 86.5], [33 - 2.6 * k, 33.6 - 2.9 * k], color="#D9D2CB", lw=1.7, solid_capstyle="round")


def sensor(xy, lab, col):
    bx.add_patch(FancyBboxPatch((xy[0] - 3.6, xy[1] - 2.4), 7.2, 4.8, boxstyle="round,pad=0,rounding_size=0.8", fc=col, ec="white", lw=0.6))
    bx.text(xy[0], xy[1] - 0.1, lab, ha="center", va="center", fontsize=6, color="white", fontweight="bold")


s2, s1, s5p = (sh + el) / 2 + [0.5, 0], (el + wr) / 2 + [0, -0.4], np.array([71.5, 29.5])
sensor(s2, "S2", B01); sensor(s1, "S1", B01); sensor(s5p, "S5", B12)


def callout(src, txt_xy, head, body, col):
    bx.plot([src[0], txt_xy[0] - 1.2], [src[1], txt_xy[1] + 1.6], color="#6B6B6B", lw=0.5)
    bx.text(txt_xy[0], txt_xy[1] + 1.2, head, fontsize=6.6, fontweight="bold", color=col, va="bottom")
    bx.text(txt_xy[0], txt_xy[1] - 0.4, body, fontsize=6.0, va="top", linespacing=1.3)


callout(s2 + [3.6, 0.5], (44, 74), "Upper-arm IMU (S2)", "9-axis, ~50 Hz\nitems 000–014, 026", B01)
callout(s1 + [0.5, -2.4], (52, 13), "Forearm IMU (S1)", "9-axis, ~50 Hz\nitems 000–014, 026", B01)
callout(s5p + [1.5, 2.4], (75, 52), "Sensor glove (S5)", "finger flex sensors\n+ hand IMU\nitems 015–025", B12)
save(fig, "Fig1")

# =====================================================================  Figure 2
I = pd.read_csv(res / "v5_B_item_auc.csv")
CI = pd.read_csv(res / "v5_B_item_auc_ci.csv").set_index("item")
I = I.join(CI[["cv_1v2_lo", "cv_1v2_hi"]], on="item").sort_values("cv_1v2", ascending=True).reset_index(drop=True)

fig = plt.figure(figsize=(160 * MM, 132 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1], wspace=0.16, hspace=0.42,
                      left=0.27, right=0.985, top=0.9, bottom=0.075)
ax = fig.add_subplot(gs[:, 0]); y = np.arange(len(I))
for k in y[::2]:
    ax.axhspan(k - 0.5, k + 0.5, color=ROW, lw=0, zorder=0)
ax.hlines(y, I.cv_1v2_lo, I.cv_1v2_hi, color="#F2B98F", lw=1.1, zorder=1)
ax.hlines(y, I.cv_1v2, I.cv_0v1, color="#9E9E9E", lw=0.6, zorder=2)
ax.scatter(I.uni_1v2, y, s=11, marker="D", facecolors="white", edgecolors=B12, lw=0.6, zorder=3)
ax.scatter(I.cv_0v1, y, s=17, color=B01, zorder=4, lw=0)
ax.scatter(I.cv_1v2, y, s=17, color=B12, zorder=4, lw=0)
m01, m12 = I.cv_0v1.median(), I.cv_1v2.median()
for m, c in [(m01, B01), (m12, B12)]:
    ax.axvline(m, color=c, ls=(0, (2, 2)), lw=0.7, zorder=0.5)
ax.text(m12, len(I) - 0.1, f"median\n{m12:.3f}", color=B12, ha="center", va="bottom", fontsize=6.1)
ax.text(m01, len(I) - 0.1, f"median\n{m01:.3f}", color=B01, ha="center", va="bottom", fontsize=6.1)
ax.set_yticks(y); ax.set_yticklabels([f"{NAMES[int(i)]}  {int(i):03d}" for i in I["item"]], fontsize=6.0)
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0.3, 1.005); ax.set_ylim(-0.6, len(I) - 0.4); ax.set_xlabel("Area under the ROC curve (0.5 = chance)")
ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
h = [Line2D([], [], marker="o", ls="", color=B01, ms=4.2, label="Score 0 vs 1"),
     Line2D([], [], marker="o", ls="-", color=B12, ms=4.2, lw=0.9, label="Score 1 vs 2 (worst-case 95% CI)"),
     Line2D([], [], marker="D", ls="", mfc="white", mec=B12, ms=3.4, mew=0.6, label="Score 1 vs 2, single feature (no fitting)")]
ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.6, 1.055), ncol=3, frameon=False, handletextpad=0.4,
          columnspacing=1.2, borderaxespad=0)
tag(ax, "a", x=-0.56, y=1.0)

# b: median gap across analyses (worst-case CIs)
G = [("Ordinal model", *grab(s5, r"CV ordinal: median gap \+([\d.]+), cluster bootstrap m=112 95% CI \[\+([\d.]+), \+([\d.]+)\]")),
     ("Binary models", grab(s6, r"gap median ([\d.]+);")[0], *grab(s6, r"m=112: gap 95% CI \[([\d.]+), ([\d.]+)\]")),
     ("Binary, refitted\nin bootstrap", *grab(s6, r"refit-in-bootstrap.*?gap median ([\d.]+), 95% CI \[([\d.]+), ([\d.]+)\]")),
     ("Single feature,\nno fitting", *grab(s5, r"single feature: median gap \+([\d.]+), cluster bootstrap m=112 95% CI \[\+([\d.]+), \+([\d.]+)\]"))]
bx2 = fig.add_subplot(gs[0, 1]); yy = np.arange(len(G))[::-1]
for yv, (lab, est, lo, hi) in zip(yy, G):
    bx2.plot([lo, hi], [yv, yv], color=B12, lw=1.0); bx2.plot(est, yv, "o", color=B12, ms=4)
    bx2.text(0.004, yv + 0.2, lab.replace("\n", " "), va="bottom", fontsize=6.2)
    bx2.text(hi + 0.005, yv, f"{est:.3f}\n({lo:.3f}–{hi:.3f})", va="center", fontsize=5.8, color="#444444", linespacing=1.15)
bx2.set_yticks([]); bx2.spines["left"].set_visible(False)
bx2.set_xlim(0, 0.30); bx2.set_ylim(-0.5, len(G) - 0.2); bx2.set_xticks([0, 0.10, 0.20, 0.30])
bx2.set_xlabel("Median AUC difference, 0 vs 1 minus 1 vs 2\n(worst-case 95% CI)")
bx2.set_title(f"Median gap across analyses\n1 vs 2 lower in {(I.cv_1v2 < I.cv_0v1).sum()}/27 items", fontsize=6.6, loc="left", pad=3)
tag(bx2, "b", x=-0.03, y=1.03)

# c: accuracy decomposition
acc = grab(s6, r"PO model ([\d.]+) \(recall 0/1/2 = ([\d.]+)/([\d.]+)/([\d.]+)\); movement-only rule \(0 vs 2\) ([\d.]+); movement \+ random 1/2 ([\d.]+)")
cx2 = fig.add_subplot(gs[1, 1])
bars = [("Ordinal model", acc[0], INK), ("Perfect movement detection, all scored 2", acc[4], "#8C8C8C"),
        ("Perfect movement detection, 1 or 2 at random", acc[5], "#BDBDBD")]
yc = np.arange(len(bars))[::-1]
for yv, (lab, v, c) in zip(yc, bars):
    cx2.barh(yv, v * 100, height=0.42, color=c, lw=0)
    cx2.text(0.5, yv + 0.25, lab, va="bottom", fontsize=6.2)
    cx2.text(v * 100 + 1, yv, f"{v * 100:.1f}%", va="center", fontsize=6.2)
cx2.set_yticks([]); cx2.spines["left"].set_visible(False)
cx2.set_xlim(0, 100); cx2.set_ylim(-0.4, len(bars) - 0.15); cx2.set_xlabel("Item-level accuracy (%)")
cx2.set_title(f"Pooled accuracy\nmodel recall: {acc[1]:.2f} (0), {acc[2]:.2f} (1), {acc[3]:.2f} (2)",
              fontsize=6.6, loc="left", pad=3)
tag(cx2, "c", x=-0.03, y=1.03)
save(fig, "Fig2")

# =====================================================================  Figure 3
L = pd.read_csv(res / "v5_A_metrics.csv"); A = pd.read_csv(res / "v5_A_pairs.csv")
W = L[L.metric == "M1"].pivot(index="assess", columns="move", values="value")
S = feat.pivot(index="assess", columns="move", values="score")
pairs = [(11, 14, "Forearm rotation", "011 → 014"), (15, 17, "Wrist stability", "015 → 017"), (16, 18, "Wrist flexion-extension", "016 → 018")]
fig = plt.figure(figsize=(170 * MM, 112 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.02], hspace=0.62, wspace=0.42, left=0.075, right=0.985, top=0.93, bottom=0.08)
rng = np.random.default_rng(1)

# a: mean scores
axa = fig.add_subplot(gs[0, 0])
for k, (a, b, lab, _) in enumerate(pairs):
    sc = S[[a, b]].dropna()
    for j, col in enumerate([a, b]):
        v = sc[col].values; m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)) * np.sqrt(222 / 112)
        xpos = k + (j - 0.5) * 0.34
        axa.bar(xpos, m, width=0.3, color=B01 if j == 0 else "#9CC3E0", lw=0)
        axa.errorbar(xpos, m, yerr=1.96 * se, color=INK, lw=0.7, capsize=1.6, capthick=0.6)
axa.set_xticks(range(3)); axa.set_xticklabels([p[2].replace(" ", "\n", 1) for p in pairs], fontsize=6.2)
axa.set_ylim(0, 2); axa.set_yticks([0, 0.5, 1, 1.5, 2]); axa.set_ylabel("Mean item score (0–2)")
axa.legend(handles=[Rectangle((0, 0), 1, 1, color=B01), Rectangle((0, 0), 1, 1, color="#9CC3E0")],
           labels=["Elbow flexed (90°)", "Elbow extended (0°)", ], loc="upper right", bbox_to_anchor=(1.0, 1.0),
           frameon=False, ncol=1, handlelength=1, handletextpad=0.4, title="error bars: worst-case 95% CI", title_fontsize=5.8)
tag(axa, "a", x=-0.2, y=1.08)

# b: within-assessment change in range
axb = fig.add_subplot(gs[0, 1:])
for k, (a, b, lab, code) in enumerate(pairs):
    ok = W[[a, b]].dropna(); ok = ok[(ok[a] > 0) | (ok[b] > 0)]; d = (ok[b] - ok[a]).values
    r = A[(A.metric == "M1") & (A["sample"] == "all") & (A.pair == f"{a:03d}->{b:03d}")].iloc[0]
    xs = k + rng.uniform(-0.17, 0.17, len(d))
    axb.scatter(xs, d, s=3, color="#9E9E9E", alpha=0.45, lw=0, zorder=1)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    axb.add_patch(Rectangle((k - 0.24, q1), 0.48, q3 - q1, fc="none", ec=INK, lw=0.8, zorder=2))
    axb.plot([k - 0.24, k + 0.24], [med, med], color=B12, lw=1.6, zorder=3)
    fp = lambda p: f"{p:.3f}" if p < 0.05 else f"{p:.2f}"
    axb.text(k, 258, f"median {(med if abs(med) >= 0.05 else 0.0):+.1f}°\nr = {r.r:+.2f}, Holm p = {fp(r.p_holm)}\nworst case p = {fp(r.p_holm_worst)}",
             ha="center", va="top", fontsize=5.9, linespacing=1.3)
axb.axhline(0, color=INK, lw=0.6, zorder=0)
axb.set_xticks(range(3)); axb.set_xticklabels([f"{p[2]}\n{p[3]}" for p in pairs], fontsize=6.2)
axb.set_xlim(-0.55, 2.55); axb.set_ylim(-195, 262); axb.set_yticks([-180, -90, 0, 90, 180])
axb.set_ylabel("Change in range with elbow\nextended (deg, sensor channel)")
tag(axb, "b", x=-0.1, y=1.02)

# c: robustness across measures
axc = fig.add_subplot(gs[1, :])
mlab = {"M1": "Full range", "M2": "5th–95th percentile range", "M3": "Peak angular speed", "M5": "Rotation path"}
mets = ["M1", "M2", "M3", "M5"]; off = {m: (1.5 - i) * 0.17 for i, m in enumerate(mets)}
mk = {"M1": "o", "M2": "s", "M3": "^", "M5": "D"}
for k, (a, b, lab, code) in enumerate(pairs):
    if k % 2 == 0:
        axc.axhspan(k - 0.45, k + 0.45, color=ROW, lw=0, zorder=0)
    for m in mets:
        r = A[(A.metric == m) & (A["sample"] == "all") & (A.pair == f"{a:03d}->{b:03d}")].iloc[0]
        c = SIG if r.p_holm_worst < 0.05 else (INK if r.p_holm < 0.05 else NS)
        yv = 2 - k + off[m]
        axc.plot([r.ci_lo, r.ci_hi], [yv, yv], color=c, lw=0.9)
        axc.plot(r.r, yv, mk[m], color=c, ms=3.6, mfc=c if c != NS else "white", mew=0.7)
axc.axvline(0, color=INK, lw=0.6)
axc.set_yticks([2, 1, 0]); axc.set_yticklabels([f"{p[2]}\n{p[3]}" for p in pairs], fontsize=6.3); axc.tick_params(axis="y", length=0)
axc.set_ylim(-0.55, 2.55); axc.set_xlim(-0.6, 0.4)
axc.set_xlabel("Matched rank-biserial r, elbow extended vs flexed (95% CI; bootstrap of 112 of 222 assessments)\n"
               "negative = less motion with the elbow extended")
hm = [Line2D([], [], marker=mk[m], ls="", color=INK, mfc="white", ms=3.6, mew=0.7, label=mlab[m]) for m in mets]
hc = [Line2D([], [], marker="o", ls="-", color=SIG, ms=3.6, lw=0.9, label="Holm p < 0.05, worst case"),
      Line2D([], [], marker="o", ls="-", color=INK, ms=3.6, lw=0.9, label="Holm p < 0.05 at n = 222 only"),
      Line2D([], [], marker="o", ls="-", color=NS, mfc="white", ms=3.6, lw=0.9, label="not significant")]
leg1 = axc.legend(handles=hm, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, title="Measure (top to bottom)",
                  title_fontsize=6.3, alignment="left", borderaxespad=0.2)
axc.add_artist(leg1)
axc.legend(handles=hc, loc="lower left", bbox_to_anchor=(1.0, 0.0), frameon=False, borderaxespad=0.2)
axc.set_position([0.16, gs[1, :].get_position(fig).y0, 0.6, gs[1, :].get_position(fig).height])
tag(axc, "c", x=-0.16, y=1.02)
save(fig, "Fig3")

# =====================================================================  Figure 4
H = pd.read_csv(res / "v6_H3_or.csv")
fams = ["pooled", "shoulder", "elbow", "forearm", "wrist", "009/026"]
flab = {"pooled": "All items, pooled", "wrist": "Wrist, 5 items", "forearm": "Forearm, 4 items",
        "elbow": "Elbow, 2 items", "009/026": "Multi-joint, 2 items", "shoulder": "Shoulder, 8 items"}
tlab = {"tgt": "Target-joint range", "non": "Non-target-joint range", "peaks": "Speed peaks per second"}
nmap = H[H.term == "tgt"].set_index("family")["n"]
fig, axs = plt.subplots(1, 3, figsize=(170 * MM, 62 * MM), sharey=True,
                        gridspec_kw=dict(wspace=0.12, left=0.2, right=0.99, top=0.86, bottom=0.2))
for axq, t, letter in zip(axs, ["tgt", "non", "peaks"], "abc"):
    s = H[H.term == t].set_index("family").loc[fams].reset_index(); ys = np.arange(len(s))[::-1]
    axq.axhspan(ys[0] - 0.5, ys[0] + 0.5, color="#E8F1F8", lw=0, zorder=0)
    for yv, (_, r) in zip(ys, s.iterrows()):
        c = SIG if r.p < 0.05 else NS
        axq.plot([r.lo, r.hi], [yv, yv], color=c, lw=1.0, zorder=2)
        axq.plot(r.OR, yv, "D" if r.family == "pooled" else "o", color=c, ms=4.2 if r.family == "pooled" else 3.8,
                 mfc=c if c == SIG else "white", mew=0.8, zorder=3)
    axq.axvline(1, color=INK, lw=0.6, zorder=1); axq.set_xscale("log"); axq.set_xlim(0.25, 10)
    axq.set_xticks([0.25, 0.5, 1, 2, 4, 8]); axq.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    axq.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axq.set_title(tlab[t], fontsize=7, fontweight="bold", pad=4)
    axq.tick_params(axis="y", length=0)
    tag(axq, letter, x=-0.03, y=1.03)
axs[0].set_yticks(np.arange(len(fams))[::-1])
axs[0].set_yticklabels([f"{flab[f]}  (n = {int(nmap[f]):,})" for f in fams], fontsize=6.3)
for k in range(3):
    axs[k].spines["left"].set_visible(k == 0)
fig.text(0.595, 0.03, "Odds ratio for a full (2) rather than partial (1) score per within-item SD (log scale; worst-case 95% CI)",
         ha="center", fontsize=7)
save(fig, "Fig4")
