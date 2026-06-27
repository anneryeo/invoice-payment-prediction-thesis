# Revisions Summary — Beley-Reyes Thesis (IPPP)

**Manuscript:** `Beley-Reyes_Thesis2-ACM.docx` · ACM format
**Repo:** `invoice-payment-prediction-thesis` (branch `main`)
**Prepared for:** Anne Reyes
**Date:** June 23, 2026 · **Updated:** June 27, 2026 (formatting cleanup + adviser-revision verification; Appendix G resolved)

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

### 0.2 Adviser revision conformity (the 12 official revisions)

We checked each adviser-requested revision against the current manuscript:

| #  | Revision                                                                                | Status                            | Where it appears                                                                                                                                                                                   |
| -- | --------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Discuss all tables and figures (intro paragraph above, in-depth explanation below)      | ✅ Implemented                    | Captions sit above tables and below figures, each with surrounding discussion; Appendices A–F carry intro + discussion text. Throughout Ch. 3–4.                                                 |
| 2  | Add the letter of consent                                                               | ✅ Implemented                    | Covered by**Appendix K** — the censored NDA, which doubles as the consent letter — **p. 112**.                                                                                       |
| 3  | Include ethical considerations in Chapter 3                                             | ✅ Implemented                    | **§3.7 Ethical Considerations** (3.7.1 Data Privacy, 3.7.2 Ethical Use of Predictions) — **p. 59**.                                                                                  |
| 4  | Include SDG in the paper                                                                | ✅ Implemented                    | **§1.6 Alignment with United Nations Sustainable Development Goals** — **p. 16** (SDG 10, 16, 17).                                                                                   |
| 5  | Summarize Section 2.1 in a paragraph                                                    | ✅ Implemented                    | **§2.1 Introduction** — **p. 18**.                                                                                                                                                   |
| 6  | Redefine the architecture (DFD update; in-depth modelling + data-prep / applied ethics) | ⚠️**Pending (co-author)** | **§3.1 Research Design (p. 38)** / **§3.2 Data Acquisition and Preprocessing (p. 39)** exist, but the DFD diagram has **not yet been updated** — to be completed by RJ Beley. |
| 7  | Table title on top of tables                                                            | ✅ Implemented                    | Table captions are above each table (e.g.,**Table 1.1, p. 10**; **Table 2.2, p. 33**).                                                                                                 |
| 8  | Add conclusion to the manuscript                                                        | ✅ Implemented                    | **§5.5 Conclusion** (distinct from §5.1 Summary) — **p. 88**.                                                                                                                       |
| 9  | Add titles to results connecting to each research question                              | ✅ Implemented                    | **§4.1 RQ1 (p. 62)**, **§4.2 RQ2 (p. 65)**, **§4.3 RQ3 (p. 66)**, **§4.5 RQ4 (p. 75)** — now formatted as "RQ_n: …" after the em-dash cleanup.                       |
| 10 | Add the demo to the results (4.6 prototype demonstration)                               | ✅ Implemented                    | **§4.6 Prototype Demonstration** with Figures **4.9–4.14**, **pp. 78–83**.                                                                                                    |
| 11 | Add communication letter to appendices                                                  | ✅ Implemented                    | **Appendix G** (authorization/communication letter, now with actual content) — **p. 112**.                                                                                           |
| 12 | Include a section for the significance of the study                                     | ✅ Implemented                    | **§1.5 Significance of the Study** — **p. 15**.                                                                                                                                      |

### 0.3 Open items to resolve before final submission

- **Revision #6 (architecture / DFD)** remains with the co-author, pending the updated DFD and the expanded modelling/data-prep narrative.

---

## 1. What was revised

The pulled commits split into two waves with two authors.

**Wave 1 — Manuscript revisions (Anne Reyes, June 12).** The chapter-by-chapter rewrite that implemented the revision plan: expanded Chapters 1–2, added Significance, SDG, and Comparative Analysis sections, rebuilt Chapter 3 methodology and ethics, retitled Chapter 4 around the research questions, added the Conclusion, and reworked references and appendices.

**Wave 2 — App + final-submission revisions (RJ Beley, June 16–18).** The bulk of the pulled commits. These hardened the Dash application and produced the final manuscript layout pass:

| Area                              | What changed                                                                                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Manuscript (final pass)** | `docs(thesis): revise layout, figures, and appendices for final submission`; added running-app screenshots to §4.6                                                                      |
| **Dashboard**               | Replaced hardcoded KPIs with live`ResultsAnalyzer` data; fixed bar-chart clipping; on-disk caching                                                                                       |
| **Invoice Drilldown**       | Full rebuild — bracket-aware KPIs, cash-flow chart, toggleable filters, chart interactivity, two-layer on-disk cache                                                                      |
| **Inference pipeline**      | Decomposed`InferencePipeline`; added public `predict_proba`; threaded `cox_scaler` through; CPU remap for XGBoost at load; batch inference with a `/batch` command                 |
| **Training**                | Aligned production config with the notebook sweep; per-run training logs; picklable Pool args;`cox_tuning_report.xlsx`; relocated training cache                                         |
| **Bug fixes**               | Broke the dashboard infinite-loading loop; fixed redirect targets, amount/summary duplication, ordinal label-encoder mismatch, hyperparameter recovery from old-format sessions            |
| **Hardening / cleanup**     | Disabled Dash debug mode; removed unused modules and the revision scripts/`plan.md`; moved thesis-paper files into their own folder; dependency fixes (`pyarrow`, `python-calamine`) |

---

## 2. Conformity to the revision plan

The current manuscript was verified section-by-section against the plan's completion checklist. **Conformity is high — nearly every required item is present.**

### Fully conformant

| Plan requirement                                                     | Status in manuscript                                          |
| -------------------------------------------------------------------- | ------------------------------------------------------------- |
| §1.5 Significance of the Study                                      | Present                                                       |
| §1.6 SDG Alignment                                                  | Present                                                       |
| §1.7 Scope (renumbered)                                             | Present                                                       |
| Table 1.1 — existing approaches comparison                          | Present, with intro + caption above                           |
| §2.13 Comparative Analysis of Existing Technologies                 | Present (Table 2.2)                                           |
| §2.14 Theoretical Framework (renumbered)                            | Present                                                       |
| §3.1 Research Design (before first DFD)                             | Present                                                       |
| §3.7 Ethical Considerations (3.7.1 Privacy, 3.7.2 Ethical Use)      | Both subsections present                                      |
| Ch.4 retitled around RQ1–RQ4 (§4.1–4.5)                           | Present                                                       |
| §4.6 Prototype Demonstration (MVP framing)                          | Present; "MVP"/"Minimum Viable" used                          |
| §4.7 Benchmark Dataset Generalizability                             | Present                                                       |
| §5.5 Conclusion (distinct from Summary)                             | Present                                                       |
| Appendices A–H, each its own heading                                | Present                                                       |
| Appendices A/B/C DSO intro + discussion text                         | Present                                                       |
| Appendix G (Authorization Letter) + Appendix H (Variable Dictionary) | Both added                                                    |
| Table captions**above** tables, with intro sentence            | Conformant across Ch.3–4                                     |
| Figure captions**below**, with intro paragraphs above          | 26 figures, conformant                                        |
| References [41]/[42]/[43] informal sources removed                   | Olawale, GeeksforGeeks, MachineLearningMastery — none remain |
| Added refs: RA 10173, Cox (1972), Chawla/SMOTE (2002)                | All present in reference list                                 |
| References renumbered                                                | List now runs to [47] (43 + new additions)                    |

### Items to double-check before final submission

- **UN SDG (2015) citation.** The §1.6 SDG section is present, but a formal "Transforming our world / Sustainable Development" entry was not detected in the References list. Confirm the SDG framework is cited, or add it.
- **§4.6 screenshots.** A commit added running-app screenshots, and 26 images are embedded, but no `Figure 4.9+` captions were found for the five app screens. Confirm the screenshots are present and properly captioned/numbered.
- **Page count.** The plan targeted ~100 pages. Body text is ~14,100 words; in ACM double-column this is substantial but worth a final page-count check against the target.

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

The manuscript conforms to the June 12 revision plan on essentially every structural and formatting requirement. The remaining open items are minor and verification-only (SDG citation, §4.6 screenshot captions, Appendix G letter, final page count). The partner's app work substantially exceeded the plan's scope, moving the prototype from "wire up two screens" to a cached, batch-capable, production-hardened MVP.

*Reconstructed from git history on branch `main`; the original revision plan was recovered from commit `ad37755~1`.*
