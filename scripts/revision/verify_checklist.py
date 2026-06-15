"""Programmatic verification of the THESIS_REVISION_PLAN.md checklist."""
import re
import sys
sys.stdout.reconfigure(encoding="utf-8")
from docx_helpers import load, iter_blocks
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

doc = load()
blocks = list(iter_blocks(doc))
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))

def has_text(prefix):
    norm = " ".join(prefix.split())
    return any(isinstance(b, Paragraph)
               and " ".join(b.text.split()).startswith(norm) for b in blocks)

def par_index(prefix):
    norm = " ".join(prefix.split())
    for i, b in enumerate(blocks):
        if isinstance(b, Paragraph) and " ".join(b.text.split()).startswith(norm):
            return i
    return -1

def has_drawing(p):
    return bool(p._p.findall(".//" + qn("w:drawing")))

def is_textpar(b):
    return isinstance(b, Paragraph) and b.text.strip() and not has_drawing(b) \
        and (not b.style or "Caption" not in b.style.name)

# --- captions above tables / below figures; intro+discussion presence
bad_tbl, bad_fig, fig_missing, tbl_missing = [], [], [], []
for i, b in enumerate(blocks):
    if isinstance(b, Paragraph) and b.style and b.style.name == "Caption":
        txt = b.text.strip()
        if txt.startswith("Table"):
            nxt = next((x for x in blocks[i+1:i+3] if not
                        (isinstance(x, Paragraph) and not x.text.strip())), None)
            if not isinstance(nxt, Table):
                bad_tbl.append(txt[:50])
            if not (i >= 1 and is_textpar(blocks[i-1])):
                tbl_missing.append(("intro", txt[:50]))
            # explanation after the table
            j = i + 1
            while j < len(blocks) and not isinstance(blocks[j], Table):
                j += 1
            if j + 1 >= len(blocks) or not any(
                    is_textpar(x) for x in blocks[j+1:j+3]):
                tbl_missing.append(("explanation", txt[:50]))
        elif txt.startswith("Figure"):
            prevs = [x for x in blocks[max(0, i-3):i]]
            if not any(isinstance(x, Paragraph) and has_drawing(x) for x in prevs):
                bad_fig.append(txt[:50])
            # intro above figure par
            k = next((j for j in range(i-1, max(0, i-4)-1, -1)
                      if isinstance(blocks[j], Paragraph) and has_drawing(blocks[j])), None)
            if k is None or not any(is_textpar(x) for x in blocks[max(0, k-2):k]):
                fig_missing.append(("intro", txt[:50]))
            if not any(is_textpar(x) for x in blocks[i+1:i+3]):
                fig_missing.append(("discussion", txt[:50]))
check("All table captions above tables", not bad_tbl, str(bad_tbl))
check("All figure captions below figures", not bad_fig, str(bad_fig))
check("Figures have intro+discussion", not fig_missing, str(fig_missing))
check("Tables have intro+explanation", not tbl_missing, str(tbl_missing))

# --- no numbered/bulleted lists in Ch1/Ch2
c1, c3 = par_index("CHAPTER 1:"), par_index("CHAPTER 3:")
bullets = [b.text[:40] for b in blocks[c1:c3]
           if isinstance(b, Paragraph)
           and not (b.style and b.style.name.startswith("Heading"))
           and (b._p.findall(".//" + qn("w:numPr"))
                or (b.style and "List" in b.style.name))]
check("No bullet/numbered lists in Ch1–Ch2", not bullets, str(bullets))

# --- required sections
for sec in ["1.5 Significance of the Study",
            "1.6 Alignment with United Nations Sustainable",
            "1.7 Scope and Limitations",
            "2.13 Comparative Analysis of Existing Technologies",
            "2.14 Theoretical Framework",
            "3.1 Research Design",
            "3.7 Ethical Considerations", "3.7.1 Data Privacy",
            "3.7.2 Ethical Use of Predictions",
            "4.1 RQ1", "4.2 RQ2", "4.3 RQ3",
            "4.4 Metric Convergence", "4.5 RQ4",
            "4.6 Prototype Demonstration",
            "4.7 Benchmark Dataset Generalizability",
            "5.5 Conclusion",
            "APPENDIX G:", "APPENDIX H:"]:
    check(f"Section exists: {sec}", has_text(sec))

# research design before first DFD figure (search captions only, after Ch3)
first_fig3 = next((i for i, b in enumerate(blocks)
                   if i > c3 and isinstance(b, Paragraph)
                   and b.style and b.style.name == "Caption"
                   and b.text.strip().startswith("Figure 3.1:")), -1)
check("3.1 Research Design before first Ch3 figure",
      c3 < par_index("3.1 Research Design") < first_fig3)

# Table 1.1 exists
check("Table 1.1 exists", has_text("Table 1.1. Summary of existing"))

# 5.4 mentions Kaggle adapter
fd = next((b.text for b in blocks if isinstance(b, Paragraph)
           and b.text.startswith("Future research and development")), "")
check("5.4 mentions Kaggle adapter", "Kaggle" in fd and "adapter" in fd)

# data acquisition narrative
check("Data acquisition: pseudonymization+split",
      has_text("Data were obtained with written institutional approval"))

# appendix page breaks
missing_pb = []
for b in blocks:
    if isinstance(b, Paragraph) and b.text.strip().startswith("APPENDIX") \
            and b.style and b.style.name == "Heading 2":
        pPr = b._p.find(qn("w:pPr"))
        if pPr is None or pPr.find(qn("w:pageBreakBefore")) is None:
            missing_pb.append(b.text[:20])
check("Each appendix starts on its own page", not missing_pb, str(missing_pb))

# references sequential and complete
refs = [b.text for b in blocks if isinstance(b, Paragraph)
        and re.match(r"\[\d+\] ", b.text)]
nums = [int(re.match(r"\[(\d+)\]", t).group(1)) for t in refs]
check("References numbered 1..47 sequentially", nums == list(range(1, 48)),
      f"got {len(nums)} entries")
joined = "\n".join(refs)
for needle in ["Chawla", "Cox", "10173", "United Nations. 2015",
               "He and E. A. Garcia", "Han", "Batista"]:
    check(f"Reference present: {needle}", needle in joined)
for junk in ["MachineLearningMastery", "GeeksforGeeks", "Olawale Awe"]:
    check(f"Junk reference removed: {junk}", junk not in joined)

# citation sanity: all in-text citations within 1..47
ref_start = par_index("REFERENCES")
overflow = set()
for b in blocks[:ref_start]:
    texts = []
    if isinstance(b, Paragraph):
        texts = [b.text]
    elif isinstance(b, Table):
        texts = [c.text for r in b.rows for c in r.cells]
    for t in texts:
        for m in re.finditer(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]", t):
            for n in (int(x) for x in m.group(1).split(",")):
                if not 1 <= n <= 47:
                    overflow.add(n)
check("All in-text citations within [1..47]", not overflow, str(overflow))

fails = [r for r in results if not r[1]]
for name, ok, detail in results:
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {detail}"))
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed.")
sys.exit(1 if fails else 0)
