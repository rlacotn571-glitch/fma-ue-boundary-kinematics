"""Additional file 2: TRIPOD+AI checklist (Collins et al., BMJ 2024;385:e078378; checklist version 11-Jan-2024).

Item texts are short paraphrases; locations refer to manuscript sections (page/line numbers to be added by
the authors after final typesetting). Usage: python -I 21_tripod_checklist.py <submission dir>
"""
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt

sub = Path(sys.argv[1])
NOTE = ("This study develops ordinal models with internal cross-validation as analytic instruments to quantify how well "
        "kinematics separate FMA-UE score boundaries; it does not propose a model for clinical deployment and includes no external "
        "evaluation. Items marked E (evaluation only) are therefore not applicable. Locations are manuscript sections; "
        "AF1 = Additional file 1.")
R = [
 ("Title", "1", "D;E", "Identify study as developing/evaluating a multivariable prediction model, population and outcome", "Title (item-level analysis of wearable kinematics for FMA-UE item scores; 222 assessments; subacute stroke)"),
 ("Abstract", "2", "D;E", "See TRIPOD+AI for Abstracts", "Abstract (Background, Methods, Results with CIs, Conclusions; trial registration not applicable)"),
 ("Background", "3a", "D;E", "Healthcare context and rationale, including existing models", "Background, paragraphs 1-2 (existing sensor-based FMA-UE models)"),
 ("Background", "3b", "D;E", "Target population, intended purpose and users", "Background, paragraph 4; Methods, Models (analytic instruments, not deployment tools)"),
 ("Background", "3c", "D;E", "Known health inequalities between sociodemographic groups", "Methods, Preprocessing (no sociodemographic variables linked to assessments); Limitations"),
 ("Objectives", "4", "D;E", "Objectives, development and/or validation", "Background, paragraph 4 (primary, secondary, exploratory questions)"),
 ("Data", "5a", "D;E", "Data sources, rationale and representativeness", "Methods, Data source (two-centre RCT, public release); Limitations"),
 ("Data", "5b", "D;E", "Dates of data collection", "Methods, Data source (accrual dates not reported; approvals 2019-2020, registration 2022)"),
 ("Participants", "6a", "D;E", "Setting, number and location of centres", "Methods, Data source (two hospitals, Xi'an, China)"),
 ("Participants", "6b", "D;E", "Eligibility criteria", "Methods, Data source (subacute stroke 15-180 days, Brunnstrom II-VI; full criteria in the REHAB data descriptor)"),
 ("Participants", "6c", "D;E", "Treatments received and how handled", "Methods, Data source (two 14-day programmes; allocation not linked, not modelled)"),
 ("Data preparation", "7", "D;E", "Pre-processing and quality checks, incl. across groups", "Methods, Preprocessing and quality control; Fig. 1"),
 ("Outcome", "8a", "D;E", "Outcome definition, how/when assessed, rationale", "Methods, Data source (FMA-UE item score 0/1/2, ceiling of two therapists' mean, at admission and discharge)"),
 ("Outcome", "8b", "D;E", "Qualifications of outcome assessors if subjective", "Methods, Data source (two therapists with unified training; further details not reported)"),
 ("Outcome", "8c", "D;E", "Blinding of outcome assessment", "Methods, Data source (blinding not reported in source)"),
 ("Predictors", "9a", "D", "Choice and pre-selection of predictors", "Methods, Primary analysis - Features (fixed set per sensor type, chosen before boundary analysis)"),
 ("Predictors", "9b", "D;E", "Definition and measurement of predictors; blinding", "Methods, Features; Data source (features computed automatically without access to scores); AF1 S1"),
 ("Predictors", "9c", "D;E", "Qualifications of predictor assessors if subjective", "Methods, Data source (manual segmentation by trained therapists; feature computation automatic)"),
 ("Sample size", "10", "D;E", "How study size was arrived at; sufficiency", "Methods, Unit of analysis (fixed by public release; no formal calculation; worst-case CIs)"),
 ("Missing data", "11", "D;E", "Handling of missing data and omitted data", "Methods, Preprocessing (122 constant and 20 short signals excluded; 6 feature values imputed with training-fold median)"),
 ("Analytical methods", "12a", "D", "Use of data; partitioning", "Methods, Models (10 x 5-fold stratified cross-validation; ordinal and separate binary models); Fitting-free check"),
 ("Analytical methods", "12b", "D", "Handling of predictors (transformation, standardisation)", "Methods, Features and Model (log transform; standardisation within training folds)"),
 ("Analytical methods", "12c", "D", "Model type, rationale, building steps, tuning, internal validation", "Methods, Models (penalised proportional-odds and separate L2 logistic models per boundary; no tuning; cross-validation)"),
 ("Analytical methods", "12d", "D;E", "Heterogeneity across clusters", "Methods, Unit of analysis (sites not released; dependence handled by worst-case inference)"),
 ("Analytical methods", "12e", "D;E", "Performance measures and plots, with rationale", "Methods, Model and Boundary comparison (adjacent-boundary AUC, calibration incl. tail, cluster bootstrap); Fig. 2"),
 ("Analytical methods", "12f", "E", "Model updating", "Not applicable (no external evaluation)"),
 ("Analytical methods", "12g", "E", "How predictions were calculated for evaluation", "Not applicable (no external evaluation); code available"),
 ("Class imbalance", "13", "D;E", "Class imbalance methods", "Methods, Models (none used; AUC prevalence-independent; per-class calibration)"),
 ("Fairness", "14", "D;E", "Approaches to model fairness", "Methods, Preprocessing (not assessable: no linked sociodemographic data); Limitations"),
 ("Model output", "15", "D", "Model output and any thresholds", "Methods, Models (class probabilities and expected score; no thresholds)"),
 ("Training vs evaluation", "16", "D;E", "Differences between development and evaluation data", "Not applicable (internal cross-validation only)"),
 ("Ethical approval", "17", "D;E", "Ethics committee and consent", "Declarations, Ethics approval and consent to participate"),
 ("Open science", "18a", "D;E", "Funding and role of funders", "Declarations, Funding (no specific funding)"),
 ("Open science", "18b", "D;E", "Conflicts of interest", "Declarations, Competing interests (none)"),
 ("Open science", "18c", "D;E", "Protocol access or statement that none prepared", "Methods, Changes during analysis (no protocol preregistered; all changes in AF1 S2)"),
 ("Open science", "18d", "D;E", "Registration or statement not registered", "Abstract, Trial registration (study not registered; source trial ChiCTR2200061310)"),
 ("Open science", "18e", "D;E", "Data availability", "Declarations, Availability of data and materials"),
 ("Open science", "18f", "D;E", "Code availability", "Declarations, Availability of data and materials [AUTHOR: DOI]; AF1 S4"),
 ("PPI", "19", "D;E", "Patient and public involvement", "Methods, Patient and public involvement (none)"),
 ("Participants", "20a", "D;E", "Participant flow; numbers with and without outcome", "Results, Sample; Fig. 1; AF1 S1 (counts per score per item)"),
 ("Participants", "20b", "D;E", "Characteristics overall, missing data, differences across groups", "Results, Sample (age, sex, stroke type, time since stroke; score distribution)"),
 ("Participants", "20c", "E", "Comparison of evaluation with development data", "Not applicable (no external evaluation)"),
 ("Model development", "21", "D;E", "Numbers of participants and events in each analysis", "Results, Sample; Table 1 (n per pair); AF1 S1 (n0, n1, n2 per item)"),
 ("Model specification", "22", "D", "Full model details to allow predictions", "Methods, Model; code with fitted pipeline available [AUTHOR: DOI]; models not intended for deployment"),
 ("Model performance", "23a", "D;E", "Performance with CIs, incl. subgroups", "Results (median AUCs and gap with CIs; heterogeneity Q and I2; pooled accuracy and recall); AF1 S1 (per-item AUC with worst-case 95% CI); Fig. 2; subgroups not available"),
 ("Model performance", "23b", "D;E", "Heterogeneity across clusters", "Not examined (sites not released)"),
 ("Model updating", "24", "E", "Results of model updating", "Not applicable"),
 ("Interpretation", "25", "D;E", "Overall interpretation incl. fairness and previous studies", "Discussion, Why the partial-versus-full boundary is harder; Implications (comparison with previous models); fairness not assessable"),
 ("Limitations", "26", "D;E", "Limitations and effects on bias, uncertainty, generalisability", "Discussion, Strengths and limitations"),
 ("Usability", "27a", "D", "Handling poor quality/unavailable input data in implementation", "Not applicable (not intended for implementation); quality control described in Methods"),
 ("Usability", "27b", "D", "User interaction and expertise required", "Not applicable (analytic instrument)"),
 ("Usability", "27c", "D;E", "Next steps for future research", "Discussion, Implications (future datasets; boundary-specific reporting)"),
]
doc = Document(); doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(9)
doc.add_heading("Additional file 2. TRIPOD+AI checklist", 0)
doc.add_paragraph("Checklist: Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ 2024;385:e078378 (checklist version 11 January 2024). Item descriptions are abbreviated.")
doc.add_paragraph(NOTE)
t = doc.add_table(rows=1 + len(R), cols=5); t.style = "Table Grid"
for j, h in enumerate(["Section", "Item", "D/E", "Checklist item (abbreviated)", "Where reported / comment"]):
    c = t.cell(0, j); c.text = ""; run = c.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(8)
for i, row in enumerate(R):
    for j, v in enumerate(row):
        c = t.cell(i + 1, j); c.text = ""; c.paragraphs[0].add_run(v).font.size = Pt(8)
out = sub / "Additional_file_2_TRIPOD-AI.docx"; doc.save(out)
na = sum("Not applicable" in r[4] or "Not examined" in r[4] for r in R); au = sum("[AUTHOR" in r[4] for r in R)
print("saved", out, "| items:", len(R), "| not applicable/not examined:", na, "| needing author input:", au)



