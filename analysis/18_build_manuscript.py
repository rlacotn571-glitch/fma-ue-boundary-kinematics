"""Build the JNER submission manuscript (DOCX) from submission/manuscript_src.md + references.json.

Citations [@key; @key2] are numbered in order of first appearance (Vancouver) and rendered as [1, 3-5].
Formatting per JNER: double spacing, line numbers, page numbers, editable tables, no page breaks in text.
Usage: python -I 18_build_manuscript.py <submission dir>
"""
import json
import re
import sys
from pathlib import Path
from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm

sub = Path(sys.argv[1])
src = (sub / "manuscript_src.md").read_text(encoding="utf-8-sig")
refs = json.loads((sub / "references.json").read_text(encoding="utf-8"))

SUBS = [("R^2", "R²"), ("222^2", "222²"), ("0.5||b||^2", "0.5‖b‖²"), ("t_1 < t_2", "t₁ < t₂"), ("t_k", "tₖ"),
        ("sqrt(112.0/222)", "√(112.0/222)"), ("sqrt(222/112.0)", "√(222/112.0)"),
        ("sqrt(112/222)", "√(112/222)"), ("sqrt(222/112)", "√(222/112)"), (" x p0 + 1 x p1 + 2 x p2", " × p₀ + 1 × p₁ + 2 × p₂"),
        ("(0 × p₀", "(0 × p₀"), ("109 x 4", "109 × 4"), (">=", "≥"), ("n_eff", "n_eff"), (" -> ", " → ")]

# ---------- citation numbering ----------
order = []
for m in re.finditer(r"\[(@[^\]]+)\]", src):
    for k in re.findall(r"@([A-Za-z0-9]+)", m.group(1)):
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in refs]
unused = [k for k in refs if k not in order]
assert not missing, f"missing references: {missing}"
num = {k: i + 1 for i, k in enumerate(order)}


def compress(ns):
    ns = sorted(set(ns)); out = []; i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"{ns[i]}" if j == i else (f"{ns[i]}, {ns[j]}" if j == i + 1 else f"{ns[i]}-{ns[j]}"))
        i = j + 1
    return ", ".join(out)


def cite(m):
    keys = re.findall(r"@([A-Za-z0-9]+)", m.group(1))
    return "[" + compress([num[k] for k in keys]) + "]"


text = re.sub(r"\[(@[^\]]+)\]", cite, src)
for a, b in SUBS:
    text = text.replace(a, b)

# ---------- document setup ----------
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
pf = st.paragraph_format; pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE; pf.space_after = Pt(0)
for lvl, size in [(1, 14), (2, 12)]:
    h = doc.styles[f"Heading {lvl}"]; h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True
    h.font.color.rgb = None; h.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
sec = doc.sections[0]
for side in ["left_margin", "right_margin", "top_margin", "bottom_margin"]:
    setattr(sec, side, Cm(2.5))
ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
sec._sectPr.append(ln)
fp = sec.footer.paragraphs[0]; fp.alignment = 1
r = fp.add_run()
for tag, val in [("begin", None), (None, "PAGE"), ("end", None)]:
    if tag:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag); r._r.append(e)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = val; r._r.append(e)


def add_runs(p, s):
    for part in re.split(r"(\*\*[^*]+\*\*)", s):
        if part.startswith("**") and part.endswith("**"):
            p.add_run(part[2:-2]).bold = True
        elif part:
            p.add_run(part)


title = re.search(r"^% TITLE: (.+)$", text, re.M).group(1)
running = re.search(r"^% RUNNING: (.+)$", text, re.M).group(1)
p = doc.add_paragraph(); p.add_run(title).bold = True
# title-page details (names, affiliation, emails) are kept in a local, unshared file
_tp = sub / "title_page.txt"
_lines = (_tp.read_text(encoding="utf-8-sig").strip().splitlines() if _tp.exists()
          else ["[AUTHOR NAMES]", "[Affiliation]", "Corresponding author: [name, email]"])
for _ln in _lines:
    doc.add_paragraph(_ln)
doc.add_paragraph(f"Running title: {running}")

lines = [l for l in text.splitlines() if not l.startswith("% ")]
i = 0
table_rows = []
def flush_table():
    global table_rows
    if not table_rows:
        return
    rows = [r for r in table_rows if not re.match(r"^\|\s*-", r)]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    t = doc.add_table(rows=len(cells), cols=len(cells[0])); t.style = "Table Grid"
    for ri, row in enumerate(cells):
        for ci, c in enumerate(row):
            cell = t.cell(ri, ci); cell.text = ""
            par = cell.paragraphs[0]; par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            run = par.add_run(c); run.font.size = Pt(9); run.bold = (ri == 0)
    table_rows = []

for line in lines:
    if line.startswith("|"):
        table_rows.append(line); continue
    flush_table()
    if not line.strip():
        continue
    if line.startswith("# "):
        doc.add_heading(line[2:].strip(), level=1)
    elif line.startswith("## "):
        doc.add_heading(line[3:].strip(), level=2)
    else:
        add_runs(doc.add_paragraph(), line.strip())
flush_table()

doc.add_heading("References", level=1)
for k in order:
    p = doc.add_paragraph(f"{num[k]}. {refs[k]}")
    p.paragraph_format.left_indent = Cm(0.75); p.paragraph_format.first_line_indent = Cm(-0.75)

out = sub / "Manuscript_JNER.docx"
doc.save(out)
print("saved", out)
print("references cited:", len(order), "| unused in json:", unused)
print("order:", ", ".join(f"{num[k]}={k}" for k in order))
