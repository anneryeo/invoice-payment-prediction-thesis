# Thesis Revision Implementation Plan
**Document:** `Beley-Reyes_Thesis2-ACM.docx`
**Target:** ACM-format thesis manuscript, ~100 pages
**Current state:** ~60 pages, missing several required sections, formatting issues

This document is a complete, self-contained handoff plan for an agent. Read it in full before starting. Work section by section in the order listed. Every task references specific chapter names, section numbers, and file paths from the repository at `D:\Developer\Projects\THESIS-Utilizing-ML-to-Solve-the-IPPP`.

---

## Repository Context

The thesis is a BS Data Science undergraduate thesis by Beley and Reyes (Mapúa University). It solves the Invoice Payment Prediction Problem (IPPP) using ML at a private Philippine school. The manuscript is already in ACM format. The app is a Dash (Python/Plotly) web app. Read `README.md` for the full technical context before writing anything.

**Key files:**
- `Beley-Reyes_Thesis2-ACM.docx` — the manuscript to revise
- `README.md` — full technical context (must read)
- `FEATURE_REFERENCE.md` — feature descriptions
- `src/app/` — Dash web application source
- `results/2026_04_18_02/results.db` — SQLite results database (1092 experiments)
- `data/eda_results/` — EDA figures used in the paper

---

## Formatting Rules (Apply Everywhere)

These apply to every section touched in this plan:

1. **ACM format** — double-column, ACM-standard headings, Times New Roman body
2. **Table captions/titles → ABOVE the table** (the current manuscript places them below — this is wrong)
3. **Figure captions/titles → BELOW the figure** (standard ACM)
4. **Every figure** must have: (a) an introductory sentence or short paragraph immediately above it that sets up what the reader is about to see, and (b) a substantive analytical paragraph immediately below the caption that discusses key patterns, trends, or takeaways
5. **Every table** must have: (a) an introductory sentence immediately above it, (b) the caption above the table, and (c) a brief explanatory paragraph below discussing what the table shows
6. **No bullet lists in prose sections** — convert all bullets in Chapter 1 Introduction, Chapter 2, and Chapter 3 narrative text to flowing prose paragraphs
7. **References numbered in order of first mention** — renumber all `[n]` citations so reference 1 is the first cited in the paper, reference 2 is the second, etc. Update the References section accordingly
8. **No orphan sections** — every section must connect logically to the one before and after it

---

## Section-by-Section Tasks

---

### CHAPTER 1: INTRODUCTION

**Current state:** Short (~3 pages), lacks technology overview before objectives, no significance of study, no clear statement of existing models.

#### Task 1.1 — Expand the opening introduction paragraphs

The first two paragraphs introduce ML broadly and the IPPP. Expand them to also cover:
- What existing ML-based approaches are currently used to predict invoice payment (name specific models: XGBoost, LightGBM, Logistic Regression, Random Forest, Neural Networks)
- Their reported accuracy/precision/recall/F1 scores from the literature (pull from Chapter 2 — Schoonbee et al. reported accuracy around 80–85%; Moore and van Vuuren used survival analysis + ML)
- A brief comparative framing: "Despite these advances, existing approaches share three limitations: [list inline, not as bullets]..."
- End with: "This study addresses these gaps by introducing a multi-stage pipeline across 1,092 experimental configurations." (transition into context of study)

This expansion should add roughly half a page of dense, cited prose.

#### Task 1.2 — Add "Significance of the Study" section (new Section 1.5, renumber Scope to 1.6)

Insert a new section **1.5 Significance of the Study** between Research Objectives (1.4) and Scope and Limitations (current 1.5, renumber to 1.6).

Write ~300–400 words covering:
- **To the partner institution:** How prediction outputs can reduce bad debt exposure (cite Table 2.1 DSO data) and shift from reactive to proactive cash flow management
- **To Philippine private schools broadly:** The model is a transferable framework applicable to any school with installment billing under RA 11984 constraints
- **To the academic community:** This is the first study to benchmark ordinal classifiers and two-stage architectures specifically for educational IPPP; fills gaps identified in Schoonbee et al. [8] and Moore & van Vuuren [4]
- **To data science practice:** Demonstrates that survival-analysis feature engineering has conditional utility (helps probabilistic models, hurts tree ensembles) — a finding with broader implications for feature engineering strategy

#### Task 1.3 — Remove bullet lists in 1.4 Research Objectives

Objectives (1)–(4) are currently written as a parenthetical list embedded in a paragraph. That's acceptable. Ensure no standalone bullet/numbered lists appear in Chapter 1 prose. Convert the Scope and Limitations bullet list `(1)...(2)...(3)` into prose sentences separated by commas or semicolons.

#### Task 1.4 — Add comparative table of existing approaches to Chapter 1

After the expanded opening paragraphs (Task 1.1), insert a small table (Table 1.1) summarizing existing invoice payment prediction models. Columns: **Study | Dataset Type | Models Used | Best Accuracy / F1 | Key Limitation**. Rows should cover at minimum: Moore & van Vuuren (2020), Schoonbee et al. (2021), Mugorobin et al. (2020), Martikainen (2023), Cheong (2022). Add a sentence above and explanatory paragraph below per the formatting rules.

---

### CHAPTER 2: REVIEW OF RELATED LITERATURE & STUDIES

**Current state:** Reasonably developed but uses bullet lists in the introduction, sections are brief, lacks a comparative analysis table, and the narrative does not consistently flow from general → international → local → current study.

#### Task 2.1 — Convert 2.1 Introduction bullets to prose

The 2.1 Introduction section uses a bulleted list of 8 items. Rewrite this as two flowing paragraphs that describe the structure of the chapter in prose, removing all bullets. Retain all the content; just restructure the grammar.

#### Task 2.2 — Restructure narrative flow throughout Chapter 2

Apply the following ordering principle to each subsection where relevant: **General/global context → International studies → Asian/Philippine context → Current study positioning.** Specifically:

- Section 2.2 already flows well; add a transition sentence at the end pointing toward Section 2.3
- Section 2.3 — add a transition at the end connecting tuition collection strategies to the Philippine context (Section 2.4)
- Section 2.4 — after presenting Table 2.1, end with a paragraph that explicitly states how this institutional data motivates the prediction problem
- Section 2.5 — add a closing sentence that bridges ML for credit risk into ML specifically for invoice prediction (Section 2.6)
- Section 2.6 — add a paragraph naming the specific gap: none of the cited studies benchmarked ordinal or two-stage architectures at scale

#### Task 2.3 — Add comparative analysis table of existing IPPP technologies

Insert a new section **2.13 Comparative Analysis of Existing Technologies** before the current 2.13 Theoretical Framework (renumber Theoretical Framework to 2.14).

This section must include:
- A prose introduction (~150 words) explaining why comparing existing models is important
- **Table 2.X: Comparative Analysis of Existing Invoice Payment Prediction Studies** — Columns: **Study | Year | Domain | Method | Features Used | Performance Metrics | Limitations**
  - Include at minimum 8 rows: Schoonbee et al., Moore & van Vuuren, Mugorobin et al., Martikainen, Cheong, Appel et al., Thuy et al. (student credit scoring Vietnam), and Abbas & Hussein (loan default)
- A prose synthesis (~200 words) below the table that identifies the three or four key patterns across the compared studies and positions the current study
- Caption above the table, discussion paragraph after

#### Task 2.4 — Expand thin subsections

The following sections are currently too brief (1–3 sentences) and need expansion to at least a full paragraph each:

- **2.9.1 Current Institutional Practices** — expand to describe specific mechanisms (promissory notes, late fees, installment restructuring, text reminder campaigns) with at least 2 citations
- **2.9.2 Structural Constraints** — expand to explain how RA 11984 procedurally limits enforcement: students must submit DSWD certifications, schools must provide installment restructuring, exam denial is fully prohibited. Cite [7], [13], [23]
- **2.9.3 ML as Proactive Alternative** — expand to 2 paragraphs explaining the specific mechanism by which ML replaces reactive strategies: early-warning outputs allow targeted pre-due-date interventions rather than post-due-date enforcement
- **2.10 Ordinal Classification** — expand by explaining the Frank and Hall [9] decomposition mechanism in more detail, then describe how two-stage architectures differ conceptually (hierarchical vs. decomposed)
- **2.11 Survival Analysis** — expand by explaining what censoring means in the invoice context (some invoices unpaid as of cutoff), how Cox PH generates a hazard rate per invoice, and why this is conceptually richer than a simple binary late/not-late variable

#### Task 2.5 — Align RL with Chapter 1 objectives

At the end of Section 2.12 (Synthesis of Gaps), add a paragraph that explicitly maps each identified gap in the literature to the four research objectives stated in Section 1.4. Format as flowing prose: "Gap (1) — the lack of granular student-product variables — is directly addressed by Objective (4), which evaluates the contribution of line-item features..."

#### Task 2.6 — Add local studies (Philippine context)

Check that the following Philippine-context studies are properly discussed in the RL, not just cited:
- Carvajal et al. (2025) — financial literacy and debt management [21]
- Mencias-Tabernilla (2023) — teacher debt profiles [22]
- RA 11984 implications [7, 13, 23]

If they are only mentioned in passing, expand each to 2–3 sentences in Section 2.4 or 2.8 explaining the local mechanism and why it is relevant to educational IPPP specifically.

---

### CHAPTER 3: METHODOLOGY

**Current state:** Mostly complete but missing: research design section, ethical considerations, data privacy discussion, clearer data acquisition narrative, and figure/table formatting is wrong (captions in wrong position, no intro/explanation paragraphs).

#### Task 3.1 — Add Research Design section (new Section 3.0, before 3.1)

Insert a new section **3.1 Research Design** (push existing 3.1–3.5 to 3.2–3.6).

Write ~200–300 words explaining:
- The study uses a **quantitative, experimental research design**
- The research paradigm: benchmarking study comparing 15 model architectures under controlled conditions
- The justification for this design: experimental comparison is the standard approach for ML system evaluation (cite Lessmann et al. [24])
- Data provenance: institutional records from a private school (pseudonymized), not a survey or controlled experiment on human subjects
- The DFD diagrams (Figures 3.1–3.5) illustrate the system architecture, not the research design per se — clarify this distinction

This section should appear BEFORE the first DFD figure.

#### Task 3.2 — Fix figure/table captions and add intro+explanation paragraphs

Go through every figure and table in Chapter 3 and apply the formatting rules:

**Tables 3.1, 3.2, 3.3 (entity data dictionaries):**
- Move captions to above the table
- Add 1-sentence intro above: "Table 3.X presents the data dictionary for the [name] table, detailing each attribute's format and role in the preprocessing pipeline."
- Add explanatory paragraph after: focus on which fields are most important and why (e.g., `due_date` and `amount_paid` are the source of the target variable DTP; `category_id` is the link to granular line-item modeling)

**Tables 3.4, 3.5, 3.6 (hyperparameters):**
- Move captions above
- Add intro: "The following table presents the hyperparameter grid searched for [model type] configurations."
- Add post-table explanation: why these ranges were chosen (computational budget, prior literature, GPU constraints for XGBoost)

**Table 3.7 (feature importance methods):**
- Move caption above
- Add intro + explanation

**Figure 3.1 (ERD):**
- Add intro paragraph before: "Figure 3.1 presents the entity-relationship diagram (ERD) of the institutional dataset. Three core entities — Transactions, Categories, and Enrollees — are linked through foreign keys to form the basis for feature engineering. The Transactions table captures every discrete receivable item at category-level granularity, which is the primary innovation of this study's data model compared to aggregate student-level approaches in prior work."
- Add discussion paragraph after caption: explain the key relationships visible in the ERD

**Figure 3.2 (Level-1 DFD Pre-Processing):**
- Intro before: set up what the DFD shows
- Discussion after: walk through the main processing steps visible in the diagram

**Figure 3.3 (Modelling DFD):**
- Intro + discussion as above

**Figure 3.4 (Distribution of payment statuses):**
- Intro before: "Figure 3.4 illustrates the distribution of the four payment brackets across the 6,527 training records used in this study."
- Discussion after caption: comment on the extreme class imbalance (~76% Class 0), why this necessitates SMOTE/resampling, and how it shapes the choice of macro-F1 as the primary metric

**Figure 3.5 (Analysis DFD):**
- Intro + discussion

#### Task 3.3 — Expand data acquisition narrative in Section 3.2 (was 3.1)

The current Section 3.1 says data were "extracted from institutional journal entries" without explaining how. Expand to cover:
- **Data source:** Three Excel files manually exported from the school's internal ERP system (revenues, enrollees, chart of accounts) — describe what each file contains
- **Access and consent:** Data were obtained with written institutional approval. All student identifiers were pseudonymized using a deterministic hash (`src/utils/pseudonymizer.py`) prior to any analysis. The original data files are excluded from the repository per `.gitignore`
- **Date range:** Records span academic years 2019–2025 (through March 31, 2026). The temporal train/test split date is March 7, 2025
- **Record count:** 11,440 raw invoice records → 6,527 labeled records after filtering for records with sufficient payment history to compute DTP features
- **Label construction:** Explain how DTP was computed (payment date minus due date) and how the four-bracket ordinal target was assigned

This expansion should add roughly 200–250 words to the section.

#### Task 3.4 — Add Ethical Considerations section (new Section 3.7, after Experimental Design)

Insert a new section **3.7 Ethical Considerations** at the end of Chapter 3.

Write ~400–500 words covering two subsections:

**3.7.1 Data Privacy**
- The dataset contains sensitive student financial records. Privacy was addressed through pseudonymization of all student identifiers before any analysis (describe the method briefly — hash-based, irreversible, see `src/utils/pseudonymizer.py`)
- Raw data files are never committed to version control (enforced via `.gitignore`)
- The study complies with the Data Privacy Act of 2012 (Republic Act 10173), which governs the processing of personal information in the Philippines
- The predictive model outputs are intended for institutional use only, not individual student profiling for public purposes
- Cite RA 10173 as a reference

**3.7.2 Ethical Use of Predictions**
- There is an inherent risk that ML predictions could be used to discriminate against students with poor payment histories. The researchers acknowledge this risk
- Recommendations to mitigate: (a) predictions should inform early outreach and support, not punitive action; (b) intervention workflows should comply with RA 11984, which prohibits academic penalties for outstanding balances; (c) the model should be audited periodically for demographic bias
- The study's framing is explicitly supportive: the goal is to enable schools to offer targeted payment assistance earlier, not to flag students for enforcement

Note: This section should be completely separate from the Ethics discussion above. Do NOT collapse data privacy into one section.

#### Task 3.5 — Add communication/consent letter to appendices

Add a reference in Section 3.7.1 to **Appendix G: Institutional Communication and Authorization Letter**, which gives permission for data use. This letter is different from any NDA — it is the formal school authorization to use data for research. If the actual letter is not available in the repository, write a placeholder that describes what the letter should contain (school letterhead, authorized signatory, scope of data access, data use restrictions, date).

---

### CHAPTER 4: RESULTS AND DISCUSSION

**Current state:** Tables and figures have no introductory sentences or post-figure analytical paragraphs. Sections are not titled to connect to research questions. No demo section.

#### Task 4.1 — Retitle sections to connect to research questions

Rename the chapter subsections so they explicitly reference the research questions from Section 1.3:

- **4.1 Model Family Comparison** → **4.1 RQ1 — Performance of Ensemble Architectures vs. Base Classifiers**
- **4.2 Impact of Class Balancing** → **4.3 RQ3 — Effect of Class-Balancing Strategies on Model Performance**  
- **4.3 Convergence of Metrics** → keep as **4.4 Metric Convergence and Model Selection Criteria**

Add a new section **4.2 RQ2 — Impact of Survival-Analysis-Derived Features** (carve out the existing discussion of survival feature impact from within the current 4.2 section, where it appears as the last two paragraphs, and promote it to its own section).

Add a new section **4.5 RQ4 — Granular Feature Contribution** to present the feature importance analysis (Figures 4.5 and 4.6) as its own discussion section, rather than buried within the balancing section.

At the top of Chapter 4, add an introductory paragraph (~100 words) that lists the four research questions and maps each to the section that addresses it.

#### Task 4.2 — Add intro + discussion paragraphs for every figure and table in Chapter 4

**Table 4.1 (Peak performance per model family):**
- Caption should already be above — verify
- Intro sentence before the table: "Table 4.1 presents the peak macro-F1 and ROC-AUC scores for each model family under the enhanced feature regime, identifying the best individual model and resampling strategy per family."
- Discussion paragraph after: summarize which family won, by how much, and what this means for RQ1

**Figure 4.1 (Ordinal & Two-Stage vs Base Classifiers):**
- Intro before: set up that this figure visualizes the full F1 lift across the three families
- Discussion after caption: quantify the lift, note which configurations actually underperformed

**Figures 4.2, 4.3 (F1/AUC by model & balance strategy, enhanced):**
- Intro before each: "Figure 4.X presents [metric] distributions across all model types and balancing strategies under the enhanced feature regime."
- Discussion after each: identify which strategy wins most consistently, note the variance

**Figure 4.4 (F1 baseline features):**
- Intro + discussion: explain the baseline vs enhanced comparison, quantify the survival feature contribution

**Figures 4.5, 4.6 (Feature Importance):**
- Intro before 4.5: "Figures 4.5 and 4.6 present feature importance scores aggregated across model types, revealing which variables most consistently drive payment bracket predictions."
- Discussion after 4.6: highlight the top features (`prev_bracket`, `dtp_wavg`, `opening_balance_flag`), connect to the granular feature engineering rationale from Chapter 3

**Figure 4.7 (ROC Curves):**
- Intro + discussion: explain what ROC shows, which model achieves the best AUC and why

**Figure 4.8 (Confusion Matrices):**
- Intro + discussion: explain class-level recall patterns, note the challenge of Class 2/3 recall under all strategies

#### Task 4.3 — Add Section 4.6: Demo / Prototype System

Add a new section **4.6 Prototype Demonstration** at the end of Chapter 4.

Write ~300–400 words covering:
- Description of the Dash web application (`run_app.py`) as the prototype delivery vehicle
- Walk through the five screens: (1) Initial Setup Wizard (multi-step data upload + model training trigger), (2) KPI Dashboard, (3) Comparative Model Dashboard, (4) Invoice Drilldown, (5) Model Analysis
- Include screenshots if available (check `data/` folder for any screenshots; if none, describe the UI)
- Describe the prototype as a **Minimum Viable Product (MVP)** — it is a functional, working system that demonstrates the core workflow end-to-end, with some non-core features deferred to future development
- List which features are implemented in the MVP and which are deferred (see Task 4.3 notes below)
- State that the Kaggle generalizability adapter and additional data source integrations are planned for future development (connect to Section 5.4)

**Incomplete app components (document these accurately in the thesis):**
- `src/app/screens/dashboard.py` — KPI values are hardcoded (120, 80, 200, 92%); callbacks to load real data from the results database are not implemented
- `src/app/screens/invoice_drilldown.py` — Layout skeleton only (21 lines); no callbacks to load invoice records or invoke the trained model for per-invoice prediction
- The app does not currently support uploading data from external sources (e.g., Kaggle datasets) — only the proprietary school dataset schema is supported in the `CreditSalesProcessor` pipeline
- The "Export CSV" button in the Invoice Drilldown has no callback

#### Task 4.4 — Add Section 4.7: Kaggle Benchmark Compatibility

Add a brief section **4.7 Benchmark Dataset Generalizability** after 4.6.

Write ~200 words explaining:
- The current pipeline is designed for a specific institutional schema (revenues + enrollees + chart of accounts Excel files)
- To validate generalizability, the pipeline should be tested against publicly available invoice/accounts-receivable datasets on Kaggle (suggest searching for: "B2B invoice payment prediction", "accounts receivable dataset", "customer payment behavior")
- The feature engineering pipeline (`CreditSalesProcessor`) would need a compatibility adapter to map Kaggle dataset columns to the expected schema
- This is listed as a planned future work item — flag it explicitly

---

### CHAPTER 5: CONCLUSION AND RECOMMENDATION

**Current state:** Has "Summary of Findings" but NO actual conclusion. Sections are brief.

#### Task 5.1 — Add a proper Conclusion section (new Section 5.5)

The current Chapter 5 has Summary of Findings, Implications for Practice, Recommendation for Continuous Calibration, and Future Directions. Add a new section **5.5 Conclusion** at the end.

Write ~400–500 words. A conclusion is NOT a summary — it should:
- State what was proven or demonstrated (not just found): "This study demonstrated that hierarchical ensemble decomposition is a viable architectural paradigm for the IPPP, outperforming both ordinal classifiers and single-stage baselines across the majority of resampling and feature regimes tested."
- Connect back to the significance of the study (Section 1.5): how does achieving F1 = 0.60 for payment bracket prediction actually help the partner school? What does a 0.60 F1 mean in operational terms (e.g., if 100 invoices are due next month, the model correctly identifies ~60 of the late ones in their correct bracket)?
- Address the broader implications: beyond this single school, this paper contributes a replicable benchmarking methodology for any institution facing the IPPP
- Close with a forward-looking statement about how predictive receivables management aligns with responsible institutional governance and educational equity

#### Task 5.2 — Expand Future Directions (Section 5.4)

Current future directions are 3 items in a sentence. Expand to a full paragraph (~200 words) covering:
- SHAP explainability integration for administrative trust
- Time-varying Cox covariates as partial payments arrive
- Multi-institution federated learning
- **Completing the Dash prototype** — full callbacks for dashboard KPIs, invoice drilldown with live model inference, and Kaggle dataset adapter
- Testing on Kaggle public AR/invoice datasets for generalizability
- Retraining cadence: annual Cox PH recalibration

---

### APPENDICES

**Current state:** All appendices are embedded consecutively in one section with no page breaks. Some appendices need to be added.

#### Task 6.1 — Put each appendix on its own page

Every appendix (A through F, and new G and H below) must start on a new page. Insert a page break before each appendix heading.

#### Task 6.2 — Add Appendix G: Institutional Authorization Letter

Add a new Appendix G: **Institutional Communication and Authorization Letter**

Content: The formal letter from the partner school authorizing data use for research. If the actual letter document is not in the repository, create a placeholder page that reads: "Letter on file. This appendix contains the written authorization from [School Name] permitting the research team to access and process student financial records for the purposes of this study, subject to the conditions described in Section 3.7.1."

#### Task 6.3 — Add Appendix H: Dataset Description and Variable Dictionary

Add a new Appendix H: **Dataset Description and Variable Dictionary**

Content:
- Brief description of the pseudonymized dataset (number of records, date range, source tables)
- Full table of all 40+ features used in the ML pipeline (pull from FEATURE_REFERENCE.md in the repository)
- Columns: Feature Name | Category | Type | Description | Source
- A note clarifying that the raw data files are not publicly available due to privacy constraints but can be requested through institutional channels

#### Task 6.4 — Fix DSO appendix captions

Appendices A, B, C (DSO per plan type for 2023, 2024, 2025) currently have no explanatory text. Add:
- One paragraph description before each figure: "Appendix [X] presents the Days Sales Outstanding (DSO) per payment plan type for calendar year [year]. DSO is computed as the average number of days between invoice due date and full settlement, segmented by the student's chosen installment plan (A through E, or none)."
- Post-figure observation: note which plan type consistently has the highest DSO and what that implies

---

### REFERENCES

#### Task 7.1 — Renumber references in order of first mention

The current reference list is numbered [1]–[43] but the numbering does not consistently follow first-mention order. Go through the entire manuscript in reading order, track which reference is cited first, and renumber accordingly. Every `[n]` citation in the text must be updated to match. The References section must be reordered to match.

#### Task 7.2 — Fix incomplete/informal references

The following references at the end of the list are incomplete or informal and must be fixed with proper ACM-format citations:
- [41] "O. Olawale Awe. Computational Strategies for Handling Imbalanced Data..." — missing journal/conference, year, volume, pages, DOI
- [42] "5 Effective Ways to Handle Imbalanced Data in Machine Learning. MachineLearningMastery.com." — if this is a blog post, cite it as a web resource with author, title, URL, and access date. If there is a better academic source for SMOTE/class imbalance handling, replace it with Chawla et al. (2002), the original SMOTE paper
- [43] "Handling Imbalanced Data for Classification. GeeksforGeeks." — replace with a proper academic source

#### Task 7.3 — Add missing references

The following claims in the manuscript need citations that are either missing or should be strengthened:
- RA 10173 (Data Privacy Act of 2012) — add as a reference for the ethical considerations section
- Cox Proportional Hazards original paper: Cox, D.R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society. Series B (Methodological), 34(2), 187–202
- Chawla et al. (2002) for SMOTE: Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357
- Any SDG source (see SDG section below)

---

### SDG ALIGNMENT SECTION

#### Task 8.1 — Add SDG section to the paper

Insert a new section in Chapter 1 (after Significance of the Study, Section 1.5 — making it Section 1.6, with Scope pushed to 1.7) titled:

**1.6 Alignment with United Nations Sustainable Development Goals**

Write ~250–300 words covering how this thesis contributes to the following SDGs:

- **SDG 4: Quality Education** — By helping schools maintain financial sustainability, the prediction system protects the institution's capacity to deliver quality education. Schools that face cash flow crises may cut staff, reduce resources, or close programs. Proactive receivable management directly protects educational continuity.
- **SDG 8: Decent Work and Economic Growth** — Predictive AR management reduces financial uncertainty for small and medium-sized private educational institutions, supporting their role as employers and contributors to local economic activity.
- **SDG 10: Reduced Inequalities** — The model enables schools to identify at-risk families earlier and offer targeted payment assistance (restructuring, DSWD referral) rather than punitive enforcement — consistent with the equity-oriented intent of RA 11984.
- **SDG 16: Peace, Justice, and Strong Institutions** — The use of pseudonymized data, compliance with RA 10173, and transparent ML methodology demonstrate responsible institutional governance and data stewardship.
- **SDG 17: Partnerships for the Goals** — The open benchmarking framework and the Dash prototype are designed to be replicable by other institutions, supporting knowledge sharing and institutional capacity building.

Cite the United Nations SDG framework (United Nations. (2015). Transforming our world: The 2030 agenda for sustainable development. United Nations General Assembly.) as a reference.

---

## MVP App — Implemented Features (for Thesis Section 4.6)

The following screens and features are **implemented** in the MVP and should be described as such in Section 4.6. Do NOT describe these as "future work" or "incomplete."

### App Feature A — KPI Dashboard (`src/app/screens/dashboard.py`)
**Status: IMPLEMENTED**
Loads live data from the results database on mount. Shows four KPI cards (best model F1, best model AUC, total experiments, deployed model name), a payment bracket distribution bar chart from the processed invoice cache, and a top-models DataTable sorted by enhanced F1. Falls back gracefully if no training has been run yet.

### App Feature B — Invoice Prediction Drilldown (`src/app/screens/invoice_drilldown.py`)
**Status: IMPLEMENTED**
Loads `credit_sales_cache.pkl`, runs the deployed `InferencePipeline.predict()` and `.predict_proba()` on all invoice records, and displays a paginated table with: invoice number, due date, amount, predicted payment bracket, confidence %, and actual bracket. Supports filtering by predicted bracket and CSV export. Logs each prediction run to the audit log.

### App Feature C — Comparative Model Dashboard (`src/app/screens/model_analysis/`)
**Status: IMPLEMENTED (pre-existing)**
Full model benchmarking comparison dashboard. Allows filtering by model type and balance strategy; shows ROC curves, confusion matrices, F1/AUC comparison charts across 1,092 experiment configurations.

### App Feature D — Audit Logs (`src/app/screens/audit_logs.py`)
**Status: IMPLEMENTED**
Displays all system events (predictions run, settings saved, models loaded) with timestamp, action, and details. Auto-refreshes every 30 seconds.

### App Feature E — Settings (`src/app/screens/settings.py`)
**Status: IMPLEMENTED**
Allows users to configure the undersample threshold, default balance strategy, and late invoice cutoff. Saves preferences to `data/user_settings.json` and logs each change to the audit log.

### App Feature F — Kaggle Dataset Adapter
**Status: DEFERRED — Future Development**
A mapping layer to accept Kaggle-format invoice datasets (external columns like `invoice_date`, `due_date`, `amount`, `customer_id`, `paid_date`) and convert them to the `CreditSalesProcessor` input schema. This would allow the system to be benchmarked against public invoice datasets and used by institutions without the proprietary school data format. Planned for a future release. Reference this in Section 5.4 (Future Directions).

### App Note — Screenshots
Section 4.6 should include screenshots of each implemented screen. **Placeholder:** Insert screenshots here once the app is running. Describe each screen in a short paragraph before the placeholder: what it shows, how a school administrator would use it, and what value it provides.

---

## Checklist for the Executing Agent

Before marking the revision complete, verify:

- [ ] Every table caption is above the table
- [ ] Every figure caption is below the figure  
- [ ] Every figure has an intro paragraph above and a discussion paragraph below
- [ ] Every table has an intro sentence above and an explanation paragraph after
- [ ] No bullet lists appear in Chapter 1 or Chapter 2 narrative prose
- [ ] Significance of the Study section exists (Section 1.5)
- [ ] SDG alignment section exists (Section 1.6)
- [ ] Ethical Considerations section exists (Section 3.7) with two subsections: data privacy and ethical use
- [ ] Research Design section exists before the first DFD figure (Section 3.1)
- [ ] Data acquisition explains how data was collected, pseudonymized, and split
- [ ] Comparative analysis table of existing technologies exists in Chapter 2 (Section 2.13)
- [ ] Table 1.1 (existing approaches comparison) exists in Chapter 1
- [ ] Chapter 4 section titles reference the research questions
- [ ] Section 4.6 Prototype Demonstration exists, describes the MVP screens (A–E above), and includes screenshot placeholders
- [ ] Section 4.7 Kaggle generalizability exists
- [ ] Section 5.5 Conclusion exists (NOT a summary)
- [ ] Future Directions (5.4) mentions the Kaggle dataset adapter as a deferred future item
- [ ] Each appendix starts on its own page
- [ ] Appendix G (authorization letter) exists — placeholder: "Letter on file — co-author to provide"
- [ ] Appendix H (dataset dictionary) exists
- [ ] Appendix A/B/C have intro + discussion text for their figures
- [ ] References are renumbered in order of first mention
- [ ] References [41], [42], [43] are properly cited or replaced
- [ ] RA 10173, Cox (1972), Chawla et al. (2002), and UN SDG (2015) are added to references
- [ ] Total manuscript is approximately 90–110 pages

---

## Notes on Page Count

Current manuscript is ~60 pages. Target is ~100 pages. The additions that will contribute most to page count:

| Addition | Estimated Pages |
|---|---|
| SDG alignment section | +0.5 |
| Significance of study | +0.5 |
| Comparative analysis table + discussion (Ch 2) | +1.5 |
| Expanded Chapter 2 subsections | +3–4 |
| Research Design section | +0.5 |
| Expanded data acquisition | +0.5 |
| Ethical considerations | +1 |
| Intro/discussion paragraphs for all figures (Ch 3) | +3 |
| Intro/discussion paragraphs for all figures (Ch 4) | +3 |
| RQ-linked section splits in Ch 4 | +1 |
| Demo section (4.6) | +1 |
| Kaggle section (4.7) | +0.5 |
| Conclusion section (5.5) | +0.5 |
| Expanded future directions | +0.5 |
| Appendix G + H + A/B/C expansions | +2–3 |
| **Total estimated addition** | **~20–22 pages** |

This brings the manuscript to approximately 80–82 pages. To reach 100 pages, the executing agent should also:
- Add more citation depth to all Chapter 2 sections (each subsection should have 5–8 in-text citations minimum)
- Add a more thorough discussion of the LDA findings (currently in README but not in the paper — the finding that LD1 explains 79.5% of variance with `opening_balance_flag` as the top feature is worth a dedicated paragraph in Chapter 4)
- Expand the experimental design section (3.6 formerly 3.5) with more detail about the temporal split rationale, class imbalance statistics, and evaluation metric justification (~0.5–1 page)

---

*Last updated: 2026-06-12. Plan authored by Anne Reyes based on advisor/reviewer feedback.*
