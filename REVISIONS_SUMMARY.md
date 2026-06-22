# Revisions Summary — Beley-Reyes Thesis (IPPP)

**Manuscript:** `Beley-Reyes_Thesis2-ACM.docx` · ACM format
**Repo:** `invoice-payment-prediction-thesis` (branch `main`)
**Prepared for:** Anne Reyes
**Date:** June 23, 2026

This document summarizes the revisions pulled from `remote/main` (primarily your thesis partner, **RJ Beley** / `RJbeley`), checks them against the original **Thesis Revision Implementation Plan** authored June 12, and notes the work that went **beyond** what the plan required.

---

## 1. What was revised

The pulled commits split into two waves with two authors.

**Wave 1 — Manuscript revisions (Anne Reyes, June 12).** The chapter-by-chapter rewrite that implemented the revision plan: expanded Chapters 1–2, added Significance, SDG, and Comparative Analysis sections, rebuilt Chapter 3 methodology and ethics, retitled Chapter 4 around the research questions, added the Conclusion, and reworked references and appendices.

**Wave 2 — App + final-submission revisions (RJ Beley, June 16–18).** The bulk of the pulled commits. These hardened the Dash application and produced the final manuscript layout pass:

| Area | What changed |
|---|---|
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
|---|---|
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
- **§4.6 screenshots.** A commit added running-app screenshots, and 26 images are embedded, but no `Figure 4.9+` captions were found for the five app screens. Confirm the screenshots are present and properly captioned/numbered.
- **Appendix G letter.** Verify whether the real authorization letter is in place or still the "Letter on file — co-author to provide" placeholder.
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
