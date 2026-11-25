"""Additional file 1 (DOCX) for the JNER submission + copy of figures.

Usage: python -I 19_additional_file.py <results dir> <submission dir> <figures dir>
"""
import shutil
import sys
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt

res, sub, figs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
B = pd.read_csv(res / "v5_B_item_auc.csv")
CI = pd.read_csv(res / "v5_B_item_auc_ci.csv").set_index("item")
BIN = pd.read_csv(res / "v6_binary_item_auc.csv").set_index("item")
D = pd.read_csv(res / "v5_B_oof.csv")
cnt = D.groupby(["move", "score"]).size().unstack(fill_value=0)

CANON = ["Shoulder elevation (flexor synergy)", "Shoulder retraction (flexor synergy)", "Shoulder abduction >= 90 deg (flexor synergy)",
         "Shoulder external rotation (flexor synergy)", "Elbow flexion (flexor synergy)", "Forearm supination (flexor synergy)",
         "Shoulder adduction/internal rotation (extensor synergy)", "Elbow extension (extensor synergy)", "Forearm pronation (extensor synergy)",
         "Hand to lumbar spine (mixed)", "Shoulder flexion 0-90 deg, elbow 0 deg (mixed)", "Pronation/supination, elbow 90 deg, shoulder 0 deg (mixed)",
         "Shoulder abduction 0-90 deg, elbow 0 deg, forearm pronated (isolated)", "Shoulder flexion 90-180 deg, elbow 0 deg, forearm mid-position (isolated)",
         "Pronation/supination, elbow 0 deg, shoulder flexed 30-90 deg (isolated)", "Wrist stability, elbow 90 deg", "Wrist flexion/extension, elbow 90 deg",
         "Wrist stability, elbow 0 deg", "Wrist flexion/extension, elbow 0 deg", "Wrist circumduction", "Finger mass flexion", "Finger mass extension",
         "Grasp a (hook/thumb adduction)", "Grasp b (thumb pinch)", "Grasp c (cylinder)", "Grasp d (sphere)", "Coordination: finger-to-nose"]
LABEL = ["Shoulder elevation", "Shoulder retraction", "Shoulder abduction (at least 90 deg)", "Shoulder external rotation", "Elbow flexion",
         "Forearm supination", "Shoulder adduction and internal rotation", "Elbow extension", "Forearm pronation", "Hand touching the lumbar spine",
         "Shoulder flexion to 90 deg (elbow 0 deg)", "Forearm pronation or supination (shoulder 0 deg, elbow 90 deg)",
         "Forearm pronation (shoulder abduction 90 deg, elbow 0 deg)", "Forearm pronation and supination (shoulder flexion 90-180 deg, elbow 0 deg)",
         "Forearm pronation or supination (shoulder flexion 30-90 deg, elbow 0 deg)", "Wrist dorsiflexion (elbow 90 deg, shoulder 0 deg)",
         "Wrist flexion and extension (elbow 90 deg, shoulder 0 deg)", "Wrist dorsiflexion (elbow 0 deg, shoulder 30 deg)",
         "Wrist flexion and extension (elbow 0 deg, shoulder 30 deg)", "Wrist circumduction", "Finger flexion (all fingers together)",
         "Finger extension (all fingers together)", "Thumb adduction with all joints at 0 deg", "Thumb pinch (holding a pencil)",
         "Grasp a cylindrical object", "Grasp a spherical object", "Finger-to-nose test (five times)"]
PRIM = {"S2_rom_max": "upper-arm range (max axis)", "S2_pitch_rom": "upper-arm pitch range", "elbow_rom": "elbow proxy range",
        "S1_roll_rom": "forearm roll range", "S1_rom_max": "forearm range (max axis)", "g_pitch_rom": "glove wrist pitch range",
        "fall_rom": "all-finger flexion range", "f1_rom": "thumb flexion range", "f12_rom": "thumb+index flexion range", "S1_sparc": "forearm SPARC"}

doc = Document()
from docx.enum.section import WD_ORIENT
_sec = doc.sections[0]; _sec.orientation = WD_ORIENT.LANDSCAPE; _sec.page_width, _sec.page_height = _sec.page_height, _sec.page_width
doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(10)
doc.add_heading("Additional file 1. Supplementary methods and results", 0)
doc.add_paragraph("Wearable kinematics detect whether, more than how fully, Fugl-Meyer upper-extremity movements are performed: an item-level analysis of 222 assessments in subacute stroke.")


def table(header, rows, size=8):
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.style = "Table Grid"
    for j, h in enumerate(header):
        c = t.cell(0, j); c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i + 1, j); c.text = ""; c.paragraphs[0].add_run(str(v)).font.size = Pt(size)


doc.add_heading("S1. Item mapping, sensors and per-item results", 1)
doc.add_paragraph("Canonical task follows Fugl-Meyer et al. (1975); dataset label is the short name in the REHAB data descriptor (Table 2). "
                  "Sensors: S1 forearm IMU, S2 upper-arm IMU (items 000-014, 026); glove (items 015-025). Counts are item-assessments after "
                  "quality control. AUCs: cross-validated penalised proportional-odds model (PO; worst-case 95% CI from an assessment-level bootstrap of 112 of 222 assessments, 1,000 resamples), separate binary logistic models per boundary (Bin), and the pre-specified single feature without fitting (SF).")
rows = []
for _, r in B.iterrows():
    m = int(r["item"]); c = cnt.loc[m]
    rows.append([f"{m:03d}", CANON[m], LABEL[m], "S1+S2" if (m <= 14 or m == 26) else "glove", PRIM.get(r["primary"], r["primary"]),
                 int(c.get(0, 0)), int(c.get(1, 0)), int(c.get(2, 0)),
                 f"{r['cv_0v1']:.3f} ({CI.loc[m,'cv_0v1_lo']:.2f}-{CI.loc[m,'cv_0v1_hi']:.2f})", f"{r['cv_1v2']:.3f} ({CI.loc[m,'cv_1v2_lo']:.2f}-{CI.loc[m,'cv_1v2_hi']:.2f})", f"{BIN.loc[m,'bin_0v1']:.3f}", f"{BIN.loc[m,'bin_1v2']:.3f}", f"{r['uni_0v1']:.3f}", f"{r['uni_1v2']:.3f}"])
table(["Item", "Canonical FMA-UE task", "Dataset label", "Sensors", "Primary feature", "n0", "n1", "n2", "PO 0v1 (95% CI)", "PO 1v2 (95% CI)", "Bin 0v1", "Bin 1v2", "SF 0v1", "SF 1v2"], rows, 7)

doc.add_heading("S2. Record of changes to the analysis plan", 1)
changes = [
    ("1", "Folders 001-120 assumed to be admission assessments (one per patient).", "Score difference between folder halves (1.9/54) contradicted the reported FMA gains.", "All 222 assessments; worst-case dependence."),
    ("2", "Item difficulty from attainment of anatomical target angles.", "Sensor Euler ranges are not joint angles (e.g., elbow flexion median 32 deg); rank correlation with Rasch difficulty 0.08.", "Within-assessment comparisons only."),
    ("3", "Eight posture pairs, including 011->012/013 and synergy->mixed pairs (main result: 011->012 r = -0.39).", "Items 012/013 are shoulder tasks with the forearm held still; synergy items differ in goal.", "Three same-task pairs (011->014, 015->017, 016->018)."),
    ("4", "Discordance rule |observed - expected| >= 1.", "Cannot flag score 1 by construction.", "Probability-based rule with calibration null."),
    ("5", "Coupling index = log(non-target/target range); OR 0.72 suggested compensation lowers scores.", "Ratio algebraically tied to magnitude.", "Target and non-target ranges entered separately (final OR 1.18 for non-target)."),
    ("6", "Items 012/013 kept forearm-roll primary feature and forearm family in exploratory model.", "Inconsistent with canonical tasks.", "Upper-arm features; shoulder family."),
    ("7", "Two-logit ordinal approximation; discordance excess judged beyond calibration null.", "Truncation could set middle-class probabilities to zero.", "Penalised proportional-odds model; tail calibration; discordance descriptive."),
    ("8", "Statements that compensation did not explain the boundary.", "Absence of evidence; limited channels; protocol suppressed compensation.", "Limited to measured channels."),
    ("9", "Worst-case effective sample 115.7 (102 x 2 + 18 x 1).", "Visits of the 7 erroneous sessions not reported.", "112 (109 x 2 + 4 x 1)."),
    ("10", "Boundary gap estimated only with the proportional-odds model and fixed out-of-fold predictions.", "Shared slope could build in the gap; fixed predictions omit refit variability.", "Added separate binary models per boundary, refit-in-bootstrap check, per-item CIs, heterogeneity test and pooled-accuracy comparison; gap unchanged (0.132; refit 0.128)."),
    ("11", "Exploratory target channel for items 003, 006 and 010 differed from the primary feature.", "Inconsistent target definition across analyses.", "Target channel = primary feature for every item (pooled target OR 2.55, non-target OR 1.18)."),
]
table(["#", "Earlier version", "Reason for change", "Final version"], changes, 8)
doc.add_paragraph("Changes 3 and 5-11 followed internal adversarial reviews conducted with an AI assistant. Superseded scripts and outputs are archived with the code.")
doc.add_heading("S3. Calibration in the low-probability tail and discordant scores", 1)
doc.add_paragraph("Source: results/v5_summary.txt (script 16_v4.py). A score was discordant when its out-of-fold probability was below 0.10. Calibration null: labels drawn 1,000 times from the model's own probabilities.")
table(["Score", "Tail cases (pred. < 0.10)", "Mean predicted", "Observed", "Discordant / scored (rate)", "Entropy"],
      [("0", "2,737", "0.018", "0.004 (12)", "12 / 2,566 (0.5%)", "0.17"), ("1", "2,370", "0.021", "0.022 (52)", "52 / 1,619 (3.2%)", "0.61"),
       ("2", "2,965", "0.009", "0.009 (28)", "28 / 1,667 (1.7%)", "0.52")], 8)
doc.add_paragraph("Excess discordance at score 1 (rate at 1 minus rate at 0 and 2): 0.023 (worst-case 95% CI 0.010 to 0.036); calibration null mean 0.012 (95% range 0.003 to 0.022), p = 0.011. Because tail calibration for score 1 was accurate, the excess arises mainly from fewer discordant scores of 0 than predicted. Discordant scores of 1 were spread over 20 items, most often thumb adduction (7), wrist circumduction (5) and cylinder grasp (5).")

doc.add_heading("S4. Literature search for the novelty statement", 1)
doc.add_paragraph("Searches were run on 23-24 November 2026 in PubMed and with a general web search engine.")
q = [("PubMed", "wearable sensors Fugl-Meyer automated assessment stroke machine learning"),
     ("PubMed", "Fugl-Meyer automatic scoring sensor stroke upper limb"),
     ("PubMed", "Fugl-Meyer wearable sensors stroke upper extremity (2023-)"),
     ("PubMed", "Fugl-Meyer automated assessment kinematic stroke machine learning (2023-)"),
     ("PubMed", "Fugl-Meyer upper extremity measurement properties kinematics validity (2023-)"),
     ("PubMed", "Fugl-Meyer Rasch item hierarchy kinematics; Fugl-Meyer label noise machine learning"),
     ("Web", "Fugl-Meyer item hierarchy kinematic validation wearable sensors Rasch"),
     ("Web", "Fugl-Meyer score inconsistency label noise wearable sensor scoring boundary"),
     ("Web", "kinematic cut-off distinguishing Fugl-Meyer item scores 1 and 2"),
     ("Web", "automated Fugl-Meyer assessment wearable class-wise accuracy partial vs full")]
table(["Source", "Query"], q, 8)
doc.add_paragraph("Closest prior work: Formstone et al. (2021) reported per-class F1 for scores 0, 1 and 2 and found score 1 hardest; "
                  "Zhou et al. (2025) analysed the same cohort with per-item accuracy and noted ambiguous 0/1 and 1/2 boundaries. "
                  "No study analysed the adjacent boundaries separately with calibrated ordinal models and dependence-robust inference.")

doc.add_heading("S5. Code and reproducibility", 1)
table(["Script", "Purpose"], [
    ("02_extract_features.py", "Feature extraction from raw signals (features_v1.csv)"),
    ("14_v3.py", "Supporting analysis 1 (same-task posture pairs), worst-case n_eff = 112"),
    ("16_v4.py", "Primary ordinal boundary analysis, calibration, discordance"),
    ("20_tripod_support.py", "Missing feature values; per-item AUC CIs"),
    ("21_tripod_checklist.py", "TRIPOD+AI checklist (Additional file 2)"),
    ("22_v6.py", "Separate binary models, refit-in-bootstrap, pooled accuracy, heterogeneity, wrist change in degrees, determinants (supporting analysis 2)"),
    ("17_figures_v4.py", "Figures 1-4 (v4 layout, superseded)"), ("23_figures_v5.py", "Figures 1-4 (final)"),
    ("18_build_manuscript.py", "Manuscript assembly with numbered references"),
    ("01, 03-13, 15", "Superseded exploratory versions (see S2)")], 8)
out = sub / "Additional_file_1.docx"
doc.save(out); print("saved", out)

fdir = sub / "figures"; fdir.mkdir(exist_ok=True)
for n, f in [(1, "Figure1_flow_sensors"), (2, "Figure2_boundary_auc"), (3, "Figure3_posture_pairs"), (4, "Figure4_determinants")]:
    for ext in ["pdf", "png"]:
        shutil.copy(figs / f"{f}.{ext}", fdir / f"Fig{n}.{ext}")
print("figures copied to", fdir)


