"""Phase 7 — References (Tasks 7.1–7.3).

Step A (7.2/7.3): replace the three informal entries and add new references:
  [41] Chawla et al. 2002 (SMOTE)           — replaces Awe (incomplete)
  [42] Han et al. 2005 (Borderline-SMOTE)   — replaces MachineLearningMastery
  [43] Batista et al. 2004 (SMOTE+Tomek)    — replaces GeeksforGeeks
  [44] He & Garcia 2009 (imbalanced data)
  [45] Cox 1972 (proportional hazards)
  [46] RA 10173 (Data Privacy Act of 2012)
  [47] United Nations 2015 (SDG framework)
All seven are already cited in the revised text (Sections 1.6, 2.11, 3.6, 3.7).

Step B (7.1): renumber all [n] citations in order of first mention (body
paragraphs and table cells, in reading order) and reorder the reference list
to match. Citation groups are re-sorted ascending after mapping.
"""
import re
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          set_par_text, iter_blocks)
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

doc = load()
assert par_exists(doc, "[41] O. Olawale Awe"), "phase 7 already applied"

# ------------------------------------------------------------------- Step A
set_par_text(find_par(doc, "[41] O. Olawale Awe"),
    "[41] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer. 2002. "
    "SMOTE: Synthetic minority over-sampling technique. Journal of Artificial "
    "Intelligence Research 16 (2002), 321-357. https://doi.org/10.1613/jair.953")
set_par_text(find_par(doc, "[42] 5 Effective Ways"),
    "[42] H. Han, W.-Y. Wang, and B.-H. Mao. 2005. Borderline-SMOTE: A new "
    "over-sampling method in imbalanced data sets learning. In Advances in "
    "Intelligent Computing (ICIC 2005), Lecture Notes in Computer Science "
    "3644, 878-887. Springer. https://doi.org/10.1007/11538059_91")
last = find_par(doc, "[43] Handling Imbalanced Data")
set_par_text(last,
    "[43] G. E. A. P. A. Batista, R. C. Prati, and M. C. Monard. 2004. A study "
    "of the behavior of several methods for balancing machine learning "
    "training data. ACM SIGKDD Explorations Newsletter 6, 1 (2004), 20-29. "
    "https://doi.org/10.1145/1007730.1007735")
NEW_ENTRIES = [
    "[44] H. He and E. A. Garcia. 2009. Learning from imbalanced data. IEEE "
    "Transactions on Knowledge and Data Engineering 21, 9 (2009), 1263-1284. "
    "https://doi.org/10.1109/TKDE.2008.239",
    "[45] D. R. Cox. 1972. Regression models and life-tables. Journal of the "
    "Royal Statistical Society. Series B (Methodological) 34, 2 (1972), "
    "187-202.",
    "[46] Republic of the Philippines. 2012. Republic Act No. 10173: Data "
    "Privacy Act of 2012. Official Gazette of the Republic of the Philippines.",
    "[47] United Nations. 2015. Transforming our world: The 2030 agenda for "
    "sustainable development. United Nations General Assembly.",
]
anchor = last
for entry in NEW_ENTRIES:
    anchor = insert_par_after(doc, anchor._p, entry, "Normal")

# ------------------------------------------------------------------- Step B
N_REFS = 47
CITE_RE = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")


def doc_paragraphs_in_order(doc):
    """Yield (paragraph, in_refs) for body paragraphs and table-cell
    paragraphs in reading order; in_refs marks the references section."""
    in_refs = False
    for block in iter_blocks(doc):
        if isinstance(block, Paragraph):
            if (block.style and block.style.name == "Heading 1"
                    and block.text.strip().startswith("REFERENCES")):
                in_refs = True
                continue
            yield block, in_refs
        elif isinstance(block, Table):
            for row in block.rows:
                seen = set()
                for cell in row.cells:
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    for p in cell.paragraphs:
                        yield p, in_refs


# 1) scan for first-mention order (excluding the reference list itself)
mapping = {}
for par, in_refs in doc_paragraphs_in_order(doc):
    if in_refs:
        continue
    for m in CITE_RE.finditer(par.text):
        for num in (int(x) for x in m.group(1).split(",")):
            if 1 <= num <= N_REFS and num not in mapping:
                mapping[num] = len(mapping) + 1

missing = [n for n in range(1, N_REFS + 1) if n not in mapping]
assert not missing, f"uncited references would be orphaned: {missing}"


def remap_group(match):
    nums = sorted(mapping[int(x)] for x in match.group(1).split(","))
    return "[" + ", ".join(str(n) for n in nums) + "]"


def rewrite_paragraph(par):
    """Apply remapping to a paragraph, handling citations that span runs."""
    text = par.text
    if not CITE_RE.search(text):
        return
    runs = par.runs
    # offsets of each run within the concatenated text
    bounds, pos = [], 0
    for r in runs:
        bounds.append((pos, pos + len(r.text)))
        pos += len(r.text)
    # process matches right-to-left so offsets stay valid
    for m in reversed(list(CITE_RE.finditer(text))):
        repl = remap_group(m)
        s, e = m.span()
        touched = [i for i, (a, b) in enumerate(bounds) if a < e and b > s]
        if not touched:
            continue
        first = touched[0]
        a, b = bounds[first]
        head = runs[first].text[: s - a]
        tail_in_first = runs[first].text[max(0, e - a):] if e <= b else ""
        runs[first].text = head + repl + tail_in_first
        for i in touched[1:]:
            a_i, b_i = bounds[i]
            keep = runs[i].text[e - a_i:] if e < b_i else ""
            runs[i].text = keep
        # recompute bounds for subsequent (earlier) matches
        bounds, pos = [], 0
        for r in runs:
            bounds.append((pos, pos + len(r.text)))
            pos += len(r.text)
        text = "".join(r.text for r in runs)


ref_entries = {}
for par, in_refs in doc_paragraphs_in_order(doc):
    if in_refs:
        m = re.match(r"\[(\d{1,2})\]\s*(.*)", par.text, re.DOTALL)
        if m:
            ref_entries[int(m.group(1))] = (par, m.group(2))
    else:
        rewrite_paragraph(par)

assert len(ref_entries) == N_REFS, f"expected {N_REFS} entries, got {len(ref_entries)}"

# reorder: slot k (the k-th entry paragraph in the list) receives the entry
# whose new number is its position
slots = [ref_entries[old][0] for old in sorted(ref_entries)]          # in-place order
new_texts = {mapping[old]: body for old, (par, body) in ref_entries.items()}
for k, par in enumerate(slots, start=1):
    set_par_text(par, f"[{k}] {new_texts[k]}")

save(doc)
print(f"Phase 7 applied. {len(mapping)} references renumbered by first mention.")
print("old->new:", {k: mapping[k] for k in sorted(mapping)})
