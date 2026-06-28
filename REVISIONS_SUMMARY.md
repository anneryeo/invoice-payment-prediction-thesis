# Revisions Summary — Beley-Reyes Thesis (IPPP)

**Manuscript:** `Beley-Reyes_Thesis2-ACM.docx` · ACM format
**Repo:** `invoice-payment-prediction-thesis` (branch `main`)
**Prepared for:** Anne Reyes
**Date:** June 23, 2026 · **Updated:** June 29, 2026 (Chapter 3 architectural revision completed; em-dash cleanup)

This document summarizes the revisions pulled from `remote/main` (primarily your thesis partner, **RJ Beley** / `RJbeley`), checks them against the original **Thesis Revision Implementation Plan** authored June 12, and notes the work that went **beyond** what the plan required.

> **Page numbers** in this document refer to the printed page numbers shown in the manuscript after we refreshed the Table of Contents and repaginated on June 27, 2026 (the document is now **121 pages**).

---

## 0. June 27, 2026 — Formatting cleanup & adviser-revision verification

This pass operated on `docs/thesis_paper/Beley-Reyes_Thesis2-ACM.docx`. (We also removed a stale duplicate copy of the manuscript that had been left at the repository root and was missing the appendix work.)

### 0.1 Actions we made

- **We removed the blue paragraph highlights.** We deleted the light-blue background shading (`A5C9EB`) from the **18 paragraphs** that had been used to flag revised passages. They were concentrated in Chapters 1–2 as noted, but also appeared further in: Chapter 1 — the research-questions list (**p. 14**) and the Significance of the Study discussion (**p. 15**); Chapter 2 — the Section 2.1 introduction (**p. 18**) and the Comparative Analysis discussion (**p. 33**); Chapter 3 — the hyperparameter-grid introduction (**p. 53**); and Chapter 4 — the Figure 4.9–4.14 prototype-screenshot captions (**pp. 78–83**).
- **We removed all em dashes.** We replaced every one of the **71 em dashes** across Chapters 1–5 with conventional punctuation, chosen by context: **colons** for the RQ-linked result headings, **commas/semicolons** for appositive and parenthetical clauses, **parentheses** for asides that contained their own commas, and **"N/A"** for the single em dash that served as a placeholder cell in the hyperparameter table (**p. 53**). Representative cleaned passages include p. 26 (Chapter 2) and p. 82 (Chapter 4).
- **We refreshed the document fields.** We updated the Table of Contents, List of Figures, and List of Tables and repaginated. The Table of Contents now includes **Appendix K** and reflects current page numbers.
- **We resolved the Appendix G placeholder.** The duplicate authorization letter entry was corrected. Appendix G now contains the actual letter content; the earlier placeholder ("Letter on file, co-author to provide") has been removed.

### 0.2 Adviser revision conformity (the 13 official revisions)

We checked each adviser-requested revision against the current manuscript:

| # | Revision | Status | Where it appears | Actions done |
|---|----------|--------|------------------|--------------|
| 1 | Discuss all tables and figures (intro paragraph above, in-depth explanation below) | ✅ Implemented | Captions sit above tables and below figures, each with surrounding discussion; Appendices A–F carry intro + discussion text. Throughout Ch. 3–4. | We added an introductory sentence above each table/figure and an in-depth explanatory paragraph below it. Appendices A–F were each given surrounding discussion text following the same pattern. |
| 2 | Add the letter of consent | ✅ Implemented | Covered by **Appendix G** — the authorization/communication letter — **p. 96**; and **Appendix K** — the censored NDA doubling as the consent letter — **p. 112**. | We embedded the actual authorization letter in Appendix G. We also used the NDA as the consent letter (Appendix K) since it already establishes data usage agreement and identity protection, with the school name and signee censored. |
| 3 | Include ethical considerations in Chapter 3 | ✅ Implemented | **§3.7 Ethical Considerations** (3.7.1 Data Privacy, 3.7.2 Ethical Use of Predictions) — **p. 59**. | We added §3.7 with two subsections covering data privacy obligations under RA 10173 and the ethical use of payment-delay predictions in student financial aid decisions. |
| 4 | Include SDG in the paper | ✅ Implemented | **§1.6 Alignment with United Nations Sustainable Development Goals** — **p. 16** (SDG 10, 16, 17). | We added §1.6 explicitly mapping the study to SDG 10 (Reduced Inequalities), SDG 16 (Peace, Justice and Strong Institutions), and SDG 17 (Partnerships for the Goals). |
| 5 | Make comparative analyses on other methods vs ours | ✅ Implemented | **§2.13 Comparative Analysis of Existing Technologies** — **p. 33**; **Table 2.2** — **p. 34**. | We added §2.13 with Table 2.2 directly comparing existing payment prediction approaches against our system across key dimensions (algorithm, dataset scale, feature type, and interpretability). |
| 6 | Summarize Section 2.1 in a paragraph | ✅ Implemented | **§2.1 Introduction** — **p. 18**. | We wrote a concise summary paragraph at the opening of §2.1 to orient the reader before the detailed subsections, consolidating the section's scope into a single cohesive overview. |
| 7 | Redefine the architecture (DFD update; in-depth modelling + data-prep / applied ethics) | ✅ Implemented | **§3.1–§3.6** throughout Chapter 3 (pp. 38–59). Level-1 DFD replaced with the updated 12-process pipeline; five Level-2 sub-DFDs added (Processes 1.0, 5.0, 7.5, 8.5, 8.0) with accompanying narrative paragraphs. N1 and N2 narrative overflow sections fixed. Data-prep detail expanded in §3.2; applied-ethics modelling reflected in §3.7. | Replaced the Level-1 DFD diagram with the revised 12-process pipeline. Added five Level-2 sub-DFD diagrams with post-figure narrative paragraphs each. Applied N1/N2 formatting and narrative fixes. Expanded §3.2 data-acquisition/preprocessing narrative and aligned §3.7 Ethical Considerations with the updated pipeline. |
| 8 | Table title on top of tables | ✅ Implemented | Table captions are above each table (e.g., **Table 1.1, p. 11**; **Table 2.2, p. 34**). | We ensured all table captions are placed above their respective tables throughout the manuscript, consistent with ACM formatting requirements. |
| 9 | Add conclusion to the manuscript | ✅ Implemented | **§5.5 Conclusion** (distinct from §5.1 Summary) — **p. 88**. | We added §5.5 as a standalone section separate from the Summary, synthesizing the study's contributions and closing the manuscript with forward-looking remarks. |
| 10 | Add titles to results connecting to each research question | ✅ Implemented | **§4.1 RQ1 (p. 62)**, **§4.2 RQ2 (p. 65)**, **§4.3 RQ3 (p. 66)**, **§4.5 RQ4 (p. 75)** — formatted as "RQ_n: …" after the em-dash cleanup. | We retitled Chapter 4 subsections to explicitly reference each research question (e.g., "§4.1 RQ1: …"), making the mapping between results and research questions immediately visible. |
| 11 | Add the demo to the results (4.6 prototype demonstration) | ✅ Implemented | **§4.6 Prototype Demonstration** with Figures **4.9–4.14**, **pp. 78–83**. | We added §4.6 with six annotated screenshots (Figures 4.9–4.14) covering the end-to-end prototype workflow, from the dashboard KPI view through invoice drilldown and batch prediction. |
| 12 | Add communication letter to appendices | ✅ Implemented | **Appendix G** (authorization/communication letter) — **p. 96**; and **Appendix K** (censored NDA as consent + communication letter) — **p. 112**. | We embedded the actual authorization letter in Appendix G and attached the censored NDA in Appendix K. The NDA covers consent and communication in one document since it establishes the data-sharing agreement with identity protections in place. |
| 13 | Include a section for the significance of the study | ✅ Implemented | **§1.5 Significance of the Study** — **p. 15**. | We added §1.5 as a new section in Chapter 1, articulating the study's value to school finance offices, DSOs, and the broader invoice payment prediction research domain. |

### 0.3 Open items to resolve before final submission

All 13 adviser revisions are now implemented. The remaining pre-submission checks are:

- **UN SDG (2015) citation.** Confirm a formal "Transforming our world / Sustainable Development" entry appears in the References list, or add it.
- **Page count.** The plan targeted ~100 pages; the manuscript is currently **121 pages** — verify this is acceptable to the adviser.
- **Refresh document fields.** After the Chapter 3 DFD and em-dash edits, update the Table of Contents, List of Figures, and List of Tables in Word and repaginate.

---

## 1. What was revised

The pulled commits split into two waves with two authors.

**Wave 1 — Manuscript revisions (Anne Reyes, June 12).** The chapter-by-chapter rewrite that implemented the revision plan: expanded Chapters 1–2, added Significance, SDG, and Comparative Analysis sections, rebuilt Chapter 3 methodology and ethics, retitled Chapter 4 around the research questions, added the Conclusion, and reworked references and appendices.

**Wave 2 — App + final-submission revisions (RJ Beley, June 16–18).** The bulk of the pulled commits. These hardened the Dash application and produced the final manuscript layout pass:

| Area | What changed |
|------|-------------|
| **Manuscript (final pass)** | `docs(thesis): revise layout, figures, and appendices for final submission`; added running-app screenshots to §4.6 |
| **Dashboard** | Replaced hardcoded KPIs with live `ResultsAnalyzer` data; fixed bar-chart clipping; on-disk caching |
| **Invoice Drilldown** | Full rebuild — bracket-aware KPIs, cash-flow chart, toggleable filters, chart interactivity, two-layer on-disk cache |
| **Inference pipeline** | Decomposed `InferencePipeline`; added public `predict_proba`; threaded `cox_scaler` through; CPU remap for XGBoost at load; batch inference with a `/batch` command |
| **Training** | Aligned production config with the notebook sweep; per-run training logs; picklable Pool args; `cox_tuning_report.xlsx`; relocated training cache |
| **Bug fixes** | Broke the dashboard infinite-loading loop; fixed redirect targets, amount/summary duplication, ordinal label-encoder mismatch, hyperparameter recovery from old-format sessions |
| **Hardening / cleanup** | Disabled Dash debug mode; removed unused modules and the revision scripts/`plan.md`; moved thesis-paper files into their own folder; dependency fixes (`pyarrow`, `python-calamine`) |

---

## 2. Conformity to the revision plan

The current manuscript was verified section-by-section against the plan's completion checklist. **Conformity is high — nearly every required item is present.**

### Fully conformant

| Plan requirement | Status in manuscript |
|-----------------|---------------------|
| §1.5 Significance of the Study | Present |
| §1.6 SDG Alignment | Present |
| §1.7 Scope (renumbered) | Present |
| Table 1.1 — existing approaches comparison | Present, with intro + caption above |
| §2.13 Comparative Analysis of Existing Technologies | Present (Table 2.2) |
| §2.14 Theoretical Framework (renumbered) | Present |
| §3.1 Research Design (before first DFD) | Present |
| §3.7 Ethical Considerations (3.7.1 Privacy, 3.7.2 Ethical Use) | Both subsections present |
| Ch.4 retitled around RQ1–RQ4 (§4.1–4.5) | Present |
| §4.6 Prototype Demonstration (MVP framing) | Present; "MVP"/"Minimum Viable" used |
| §4.7 Benchmark Dataset Generalizability | Present |
| §5.5 Conclusion (distinct from Summary) | Present |
| Appendices A–H, each its own heading | Present |
| Appendices A/B/C DSO intro + discussion text | Present |
| Appendix G (Authorization Letter) + Appendix H (Variable Dictionary) | Both added |
| Table captions **above** tables, with intro sentence | Conformant across Ch.3–4 |
| Figure captions **below**, with intro paragraphs above | 26 figures, conformant |
| References [41]/[42]/[43] informal sources removed | Olawale, GeeksforGeeks, MachineLearningMastery — none remain |
| Added refs: RA 10173, Cox (1972), Chawla/SMOTE (2002) | All present in reference list |
| References renumbered | List now runs to [47] (43 + new additions) |

### Items to double-check before final submission

- **UN SDG (2015) citation.** The §1.6 SDG section is present, but a formal "Transforming our world / Sustainable Development" entry was not detected in the References list. Confirm the SDG framework is cited, or add it.
- **Page count.** The plan targeted ~100 pages; the manuscript is currently **121 pages**.

---

## 3. Beyond the plan

The revision plan was **manuscript-focused** and asked only for two app screens to be wired up (KPI Dashboard + Invoice Drilldown callbacks). The pulled commits went well past that scope into **production hardening of the application** — work the plan did not require:

- **Performance engineering** — two-layer on-disk caching to eliminate per-visit recompute on the dashboard and drilldown; relocated/structured training cache. Not requested in the plan.
- **Inference architecture** — decomposed `InferencePipeline`, public `predict_proba`, `cox_scaler` threading, XGBoost CPU remap at load, and a **batch inference pipeline with a `/batch` command**. The plan only asked for per-invoice prediction in the drilldown.
- **Chart interactivity & cash-flow visualization** in the drilldown beyond the basic paginated table the plan specified.
- **Training reproducibility** — aligning the production training config to the notebook sweep, per-run logs, and an exported `cox_tuning_report.xlsx`.
- **Production readiness** — debug mode disabled, dependency fixes, and a substantial repo cleanup (unused modules removed, thesis files reorganized, revision scaffolding deleted).

These changes strengthen the prototype's claim to being a working MVP in §4.6, but note they also **deleted the revision plan and its phase scripts** (`THESIS_REVISION_PLAN.md`, `scripts/revision/*`) as "no longer needed" — this summary was reconstructed from git history.

---

## 4. Bottom line

The manuscript conforms to the June 12 revision plan on essentially every structural and formatting requirement. The remaining open items are minor and verification-only (SDG citation, §4.6 screenshot captions, final page count). The partner's app work substantially exceeded the plan's scope, moving the prototype from "wire up two screens" to a cached, batch-capable, production-hardened MVP.

*Reconstructed from git history on branch `main`; the original revision plan was recovered from commit `ad37755~1`.*
