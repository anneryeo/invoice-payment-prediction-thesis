"""Phase 8 — final checklist fixes.

- Move Table 2.1 caption above its table; add intro sentence (the existing
  paragraph after it already serves as the explanation)
- Add intro + observation text for the Appendix D/E/F figures (formatting
  rules require every figure to have both)
- Set w:updateFields so Word refreshes the List of Figures/Tables and page
  numbers on next open
"""
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          insert_par_before, move_before, BODY)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

doc = load()
assert not par_exists(doc, "Table 2.1 summarizes the institution's"), "phase 8 already applied"

# ------------------------------------------------ Table 2.1 caption + intro
cap21 = find_par(doc, "Table 2.1.", style="Caption")
el = cap21._p.getprevious()
while el is not None and el.tag != qn("w:tbl"):
    el = el.getprevious()
assert el is not None
move_before(cap21._p, el)
insert_par_before(doc, cap21._p, (
    "Table 2.1 summarizes the institution's bad debts expense, student "
    "balances, and Days Sales Outstanding across school years 2019–2025, "
    "documenting the receivables trend that motivates this study."), BODY)

# -------------------------------------------- Appendix D/E/F figure text
def appendix_text(letter, intro, obs):
    head = find_par(doc, f"APPENDIX {letter}:", style="Heading 2")
    intro_p = insert_par_after(doc, head._p, intro, BODY)
    el = intro_p._p.getnext()
    while el is not None and not (el.tag == qn("w:p")
                                  and el.findall(".//" + qn("w:drawing"))):
        el = el.getnext()
    assert el is not None, f"no figure in Appendix {letter}"
    insert_par_after(doc, el, obs, BODY)

appendix_text("D",
    "Appendix D reproduces the first set of feature importance rankings "
    "reported by Schoonbee et al. [5], included as the principal point of "
    "comparison for the importance analysis in Section 4.5.",
    "Their top-ranked variables are aggregate, student-level payment history "
    "features, which is consistent with this study's finding that behavioral "
    "payment history dominates prediction — while lacking the category-level "
    "granularity and engineered survival features introduced here.")
appendix_text("E",
    "Appendix E reproduces the second set of feature importance rankings from "
    "Schoonbee et al. [5], complementing Appendix D.",
    "Together, Appendices D and E show that prior educational IPPP work "
    "concentrated its predictive signal in a small set of aggregate history "
    "variables, supporting this study's design decision to expand the feature "
    "space at line-item granularity (Section 3.3).")
appendix_text("F",
    "Appendix F reproduces the correlation scores between candidate features "
    "and the days-to-payment variable reported by Moore and van Vuuren [4].",
    "The strongest correlates in their framework are historical payment-delay "
    "aggregates, anticipating the dominance of dtp-derived features observed "
    "in this study's importance rankings (Figures 4.5 and 4.6).")

# ------------------------------------------------------- update fields flag
settings = doc.settings.element
if not settings.findall(qn("w:updateFields")):
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)

save(doc)
print("Phase 8 applied.")
