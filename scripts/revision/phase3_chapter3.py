"""Phase 3 — Chapter 3 revisions (Tasks 3.1–3.5).

- 3.1: new "3.1 Research Design" section; existing 3.1–3.5 renumbered 3.2–3.6
- 3.3: expanded data acquisition narrative (sources, consent, range, labels)
- 3.2: every figure/table gets intro above + discussion below; table captions
  moved above tables; figure captions made literal (fixes the mislabeled ERD
  caption and the "Figure 3.:" SEQ stubs); broken cross-references fixed
- 3.6: expanded experimental design (split rationale, imbalance, metrics);
  cites SMOTE [41], Borderline-SMOTE [42], SMOTE-Tomek [43], He & Garcia [44]
- 3.4/3.5: new 3.7 Ethical Considerations (3.7.1 Data Privacy cites RA 10173
  [46] and Appendix G; 3.7.2 Ethical Use of Predictions)
"""
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          insert_par_before, set_par_text, append_text,
                          move_before, BODY, CAPTION)
from docx.oxml.ns import qn

doc = load()
assert not par_exists(doc, "3.7 Ethical Considerations"), "phase 3 already applied"


def figure_par_before(cap):
    """Walk previous siblings of a caption to find the paragraph holding the image."""
    el = cap._p.getprevious()
    while el is not None:
        if el.tag == qn("w:p") and el.findall(".//" + qn("w:drawing")):
            return el
        el = el.getprevious()
    raise LookupError("no figure paragraph before caption")


def caption_above_table(cap):
    """Move a caption paragraph above its preceding table; return the tbl element."""
    el = cap._p.getprevious()
    while el is not None and el.tag != qn("w:tbl"):
        el = el.getprevious()
    assert el is not None, "no table before caption"
    move_before(cap._p, el)
    return el


def extract_drawings(par, doc):
    """Move drawing runs out of par into a fresh No Spacing paragraph after it."""
    runs = [r for r in par._p.findall(qn("w:r")) if r.findall(".//" + qn("w:drawing"))]
    fig = insert_par_after(doc, par._p, "", "No Spacing")
    for r in runs:
        par._p.remove(r)
        fig._p.append(r)
    return fig


# ----------------------------------------------- Section renumbering (3.x+1)
set_par_text(find_par(doc, "3.5 Experimental Design and Evaluation", style="Heading 2"),
             "3.6 Experimental Design and Evaluation")
set_par_text(find_par(doc, "3.4 Model Architectures", style="Heading 2"),
             "3.5 Model Architectures")
set_par_text(find_par(doc, "3.3 Survival-Analysis-Derived Features", style="Heading 2"),
             "3.4 Survival-Analysis-Derived Features")
set_par_text(find_par(doc, "3.2 Granular Feature Engineering", style="Heading 2"),
             "3.3 Granular Feature Engineering")
set_par_text(find_par(doc, "3.1 Data Acquisition and Preprocessing", style="Heading 2"),
             "3.2 Data Acquisition and Preprocessing")

# fix in-text references to renumbered sections
sec33 = find_par(doc, "The feature space comprises 40 variables")
p_surv = find_par(doc, "These features constitute the \"Enhanced\" feature regime")

# ---------------------------------------------------------------- Task 3.1
design = find_par(doc, "This study employs an experimental research design")
insert_par_before(doc, design._p, "3.1 Research Design", "Heading 2")
set_par_text(design, (
    "This study employs a quantitative, experimental research design. The "
    "research paradigm is a controlled benchmarking study: fifteen machine "
    "learning architectures are compared under identical data, split, and "
    "evaluation conditions, with the manipulated factors being model family, "
    "hyperparameter configuration, class-balancing strategy, and feature "
    "regime — a factorial space of 1,092 configurations. Experimental "
    "comparison of this kind is the standard methodology for evaluating machine "
    "learning systems, as established in large-scale credit-scoring benchmarks "
    "such as Lessmann et al. [24]."))
insert_par_after(doc, design._p, (
    "The data are observational institutional records rather than instruments "
    "administered to human subjects: the study draws on pseudonymized journal "
    "entries from a private school, so no surveys, interviews, or interventions "
    "on students were conducted. The design is therefore experimental with "
    "respect to model evaluation, not with respect to data collection. The data "
    "flow diagrams presented in Figures 3.1 through 3.5 illustrate the "
    "architecture of the developed system — its preprocessing, modeling, and "
    "analysis components — rather than the research design itself; the research "
    "design is the benchmarking protocol described in Section 3.6, which fixes "
    "the temporal train–test split, the evaluation metrics, and the experimental "
    "grid within which every architecture is assessed. The remainder of this "
    "chapter describes each component in the order data flows through the "
    "pipeline: data acquisition, granular feature engineering, survival analysis "
    "modeling, class balancing, and hierarchical ensemble classification."), BODY)

# ---------------------------------------------------------------- Task 3.3
acq1 = find_par(doc, "The primary dataset consists of 11,440 unique invoice records")
set_par_text(acq1, (
    "The primary dataset consists of 11,440 raw invoice records from a private "
    "school in Rizal, Philippines, spanning academic years 2019–2025 (with "
    "payment activity observed through March 31, 2026). The data originate from "
    "three Excel files manually exported from the school's internal enterprise "
    "resource planning (ERP) system. The revenues file contains itemized "
    "receivables — every billed amount with its due date, discounts, "
    "adjustments, and the dates and amounts of payments applied against it. The "
    "enrollees file records student enrollment per school year, including the "
    "installment plan selected, which allows enrollment streaks and plan-based "
    "features to be derived. The chart of accounts file maps every transaction "
    "category to its account classification and strategic business unit, "
    "enabling the category-level granularity central to this study. Each "
    "resulting record represents a discrete receivable item (e.g., Tuition, "
    "Miscellaneous Fee, E-Learning Platform) rather than an aggregate "
    "student-level balance."))
acq2 = insert_par_after(doc, acq1._p, (
    "Data were obtained with written institutional approval (Appendix G). All "
    "student identifiers were pseudonymized using a deterministic, irreversible "
    "hash prior to any analysis, implemented in a dedicated pseudonymization "
    "utility within the project codebase, and the original data files are "
    "excluded from the project's version-controlled repository. From the 11,440 "
    "raw records, 6,527 labeled records were retained after filtering for "
    "records with sufficient payment history to compute the days-to-payment "
    "(DTP) behavioral features; records lacking observable payment histories "
    "cannot support the lagged features the models require. The temporal "
    "train–test split is set at March 7, 2025: invoices due before this date "
    "form the training set, and invoices due on or after it form the test set."), BODY)
acq3 = find_par(doc, "Initial preprocessing involved: (1) pseudonymization")
set_par_text(acq3, (
    "Initial preprocessing involved three steps. First, student identifiers were "
    "pseudonymized as described above. Second, payment dates were temporally "
    "aligned against due dates to construct the target variable Days to Payment "
    "(DTP), computed as the number of days between an invoice's due date and the "
    "date its balance was fully settled, with negative or zero values denoting "
    "on-time settlement. Third, DTP was discretized into four ordinal brackets: "
    "Class 0 (On-Time, DTP <= 0), Class 1 (1–30 days late), Class 2 (31–60 days "
    "late), and Class 3 (61+ days late). Invoices still unpaid at the "
    "observation cutoff are flagged as censored and handled by the survival "
    "analysis described in Section 3.4."))

# ------------------------------------------------ Task 3.2 — Figures 3.1/3.2
erd_par = find_par(doc, "Below is the Entity Relationship Diagram")
fig1 = extract_drawings(erd_par, doc)
set_par_text(erd_par, (
    "Figure 3.1 presents the entity-relationship diagram (ERD) of the "
    "institutional dataset. Three core entities — Transactions, Categories, and "
    "Enrollees — are linked through foreign keys to form the basis for feature "
    "engineering. The Transactions table captures every discrete receivable item "
    "at category-level granularity, which is the primary innovation of this "
    "study's data model compared to aggregate student-level approaches in prior "
    "work."))
cap1 = find_par(doc, "Figure 3.: Level-1 DFD of the Pre-Processing component",
                style="Caption", nth=0)
set_par_text(cap1, "Figure 3.1: Entity-relationship diagram (ERD) of the "
                   "institutional dataset.")
fig1._p.addnext(cap1._p)
insert_par_after(doc, cap1._p, (
    "The relationships visible in the ERD determine how the modeling dataset is "
    "assembled. Each transaction references a category through category_id, "
    "allowing every receivable line item to be typed (tuition, miscellaneous "
    "fees, course materials, other services), while the pseudonymized student "
    "identifier links transactions to enrollment records across school years. "
    "The due_date and amount_paid fields jointly define the target variable: the "
    "gap between the due date and the date the obligation is fully offset yields "
    "Days to Payment, from which the four ordinal payment brackets are derived. "
    "The Credit Sales Fact Table consolidates these joins into a single "
    "analysis-ready view per receivable."), BODY)

cap2 = find_par(doc, "Figure 3.: Level-1 DFD of the Pre-Processing component",
                style="Caption", nth=0)  # nth 0 again: cap1 text was rewritten
fig2 = figure_par_before(cap2)
insert_par_before(doc, fig2, (
    "Figure 3.2 presents the Level-1 data flow diagram (DFD) of the "
    "pre-processing component, which transforms the three raw institutional "
    "exports into the analysis-ready credit sales dataset."), BODY)
set_par_text(cap2, "Figure 3.2: Level-1 DFD of the Pre-Processing component.")
insert_par_after(doc, cap2._p, (
    "The diagram traces the main processing steps: the raw revenues, enrollees, "
    "and chart-of-accounts files are first validated and pseudonymized, then "
    "merged into category-level transactions; payment applications are "
    "temporally aligned against due dates to compute Days to Payment; and the "
    "engineered behavioral, financial, and temporal features described in "
    "Section 3.3 are appended before the labeled records are written to the "
    "modeling cache. Each module is deterministic, so the same raw exports "
    "always reproduce the same modeling dataset — a property that supports the "
    "auditability objectives discussed in Section 3.7."), BODY)

# ------------------------------------------- Task 3.2 — Tables 3.1, 3.2, 3.3
t31_cap = find_par(doc, "Table 3.1.", style="Caption")
t31 = caption_above_table(t31_cap)
insert_par_before(doc, t31_cap._p, (
    "Table 3.1 presents the data dictionary for the Transactions table, "
    "detailing each attribute's format and role in the preprocessing pipeline."), BODY)
insert_par_after(doc, t31, (
    "Among these fields, due_date and amount_paid are the most consequential: "
    "together they are the source of the target variable, since Days to Payment "
    "is computed from the gap between an obligation's due date and the date it "
    "is fully offset. The category_id field is the link that enables granular "
    "line-item modeling — the central innovation of this study — while "
    "school_year and the pseudonymized student identifier allow behavioral "
    "features to be accumulated across a student's enrollment history."), BODY)

t32_cap = find_par(doc, "Table 3.2.", style="Caption")
t32 = caption_above_table(t32_cap)
insert_par_before(doc, t32_cap._p, (
    "Table 3.2 presents the data dictionary for the Categories table, detailing "
    "each attribute's format and role in the preprocessing pipeline."), BODY)
insert_par_after(doc, t32, (
    "Although structurally simple, this table is what gives the pipeline its "
    "granularity: category_name distinguishes tuition from miscellaneous and "
    "product-type charges, and strategic_business_unit groups categories into "
    "the operational units used for institutional reporting. Joining these "
    "attributes onto transactions is what allows the models to learn payment "
    "behavior per fee type rather than per aggregate balance."), BODY)

t33_cap = find_par(doc, "Table 3.3.", style="Caption")
t33 = caption_above_table(t33_cap)
insert_par_before(doc, t33_cap._p, (
    "Table 3.3 presents the data dictionary for the Credit Sales Fact Table, "
    "the consolidated analysis-ready view produced by the preprocessing "
    "pipeline."), BODY)
insert_par_after(doc, t33, (
    "The fact table decomposes each receivable into payment-aging buckets (paid "
    "before the due date, 1–30 days, 31–60 days, and so on), which is the "
    "representation from which the four ordinal payment brackets are labeled. "
    "The remaining_accounts_receivables field identifies invoices still unpaid "
    "at the observation cutoff; these censored records are exactly the cases "
    "handled by the survival analysis described in Section 3.4."), BODY)

# ------------------------------------------------- Task 3.2 — Figure 3.3
cap33 = find_par(doc, "Figure 3.: Level-1 DFD of the Modelling component",
                 style="Caption")
fig33 = figure_par_before(cap33)
insert_par_before(doc, fig33, (
    "Figure 3.3 presents the Level-1 DFD of the modelling component, covering "
    "class balancing, model training, and hyperparameter search across the "
    "three model families."), BODY)
set_par_text(cap33, "Figure 3.3: Level-1 DFD of the Modelling component.")
insert_par_after(doc, cap33._p, (
    "The diagram shows the experimental loop at the heart of the benchmark: the "
    "labeled dataset is split temporally, the training partition is resampled "
    "under the selected balancing strategy, and each of the fifteen model "
    "architectures is trained across its hyperparameter grid before evaluation "
    "metrics and feature importance scores are logged to the results database. "
    "Because every configuration follows the same path through this loop, "
    "results are directly comparable across model families, balancing "
    "strategies, and feature regimes."), BODY)

# --------------------------------------------- Task 3.2 — Tables 3.4–3.6
t34_cap = find_par(doc, "Table 3.4.", style="Caption")
t34 = caption_above_table(t34_cap)
insert_par_before(doc, t34_cap._p, (
    "The following table presents the hyperparameter grid searched for the base "
    "classifier configurations."), BODY)
insert_par_after(doc, t34, (
    "The ranges reflect three constraints. The computational budget capped each "
    "grid at roughly ten configurations per model so that the full "
    "1,092-experiment benchmark remained tractable; the specific values follow "
    "prior benchmarking literature, which finds diminishing returns beyond "
    "moderate tree depths and ensemble sizes [24]; and the XGBoost grid was "
    "additionally shaped by GPU memory constraints, favoring moderate depth "
    "with subsampling over exhaustive expansion."), BODY)

t35_cap = find_par(doc, "Table 3.5.", style="Caption")
t35 = caption_above_table(t35_cap)
insert_par_before(doc, t35_cap._p, (
    "The following table presents the hyperparameter grid searched for the "
    "ordinal classifier configurations."), BODY)
insert_par_after(doc, t35, (
    "The ordinal wrappers reuse the grids of their underlying base learners so "
    "that any performance difference is attributable to the Frank–Hall "
    "decomposition itself rather than to differential tuning effort. The "
    "scale_pos_weight flag is the one ordinal-specific addition, compensating "
    "for the shifting class ratios within each binary sub-problem."), BODY)

t36_cap = find_par(doc, "Table 3.6.", style="Caption")
t36 = caption_above_table(t36_cap)
insert_par_before(doc, t36_cap._p, (
    "The following table presents the hyperparameter grids searched for the "
    "two-stage model configurations, covering both the binary first stage and "
    "the multi-class second stage."), BODY)
insert_par_after(doc, t36, (
    "Because each two-stage pipeline trains two models, the per-stage grids "
    "were kept deliberately compact to contain combinatorial growth; the tuned "
    "values concentrate on the parameters with the largest observed effects "
    "(depth, learning rate, ensemble size), while secondary regularization "
    "parameters were fixed at the values established in the base-model grids."), BODY)

# --------------------------------------------------- Task 3.2 — Table 3.7
t37_intro = find_par(doc, "These influence scores are then fed into a dedicated")
set_par_text(t37_intro, (
    "These influence scores are then fed into a dedicated feature selection "
    "process in Module 9.0, where the most predictive variables are retained. "
    "Table 3.7 presents the feature importance method used for each model "
    "type."))
t37_cap = find_par(doc, "Table 3.7.", style="Caption")
t37 = caption_above_table(t37_cap)
insert_par_after(doc, t37, (
    "The methods differ because feature attribution must respect each model's "
    "internal structure: impurity-based measures suit tree learners, the "
    "gain/cover/frequency metrics decompose XGBoost's boosted splits, "
    "likelihood parameters expose Gaussian Naive Bayes' per-class evidence, and "
    "model-agnostic permutation importance covers learners such as KNN that "
    "lack intrinsic attribution. Normalizing these scores per experiment allows "
    "importance rankings to be compared across model families in Chapter 4."), BODY)

# -------------------------------- Task 3.2/notes — Figure 3.4 + Section 3.6
exp = find_par(doc, "The benchmark encompasses 1,092 unique configurations")
insert_par_after(doc, exp._p, (
    "The temporal split was chosen over random cross-validation deliberately: "
    "invoices are not exchangeable across time, and a random split would leak "
    "future payment behavior into training, inflating measured performance "
    "relative to deployment conditions. Training on all invoices due before "
    "March 7, 2025 and testing on those due afterward reproduces the situation "
    "the school faces in production — predicting forthcoming invoices from "
    "historical behavior — and exposes the models to any distribution drift "
    "across school years. The class imbalance statistics underline the metric "
    "choice: with Class 0 at roughly 76% of records, raw accuracy is dominated "
    "by the majority class, while the minority brackets that carry the greatest "
    "financial risk contribute least to it. Macro-averaged F1 corrects this by "
    "weighting all four classes equally, ROC-AUC complements it with a "
    "threshold-independent view of separability, and confusion matrices retain "
    "the per-class recall that a finance office ultimately acts on."), BODY)

cap34 = find_par(doc, "Figure 3.: Distribution of payment statuses", style="Caption")
fig34 = figure_par_before(cap34)
insert_par_before(doc, fig34, (
    "Figure 3.4 illustrates the distribution of the four payment brackets "
    "across the 6,527 training records used in this study."), BODY)
set_par_text(cap34, "Figure 3.4: Distribution of payment statuses.")
insert_par_after(doc, cap34._p, (
    "The distribution is extremely imbalanced: roughly 76% of invoices fall "
    "into Class 0 (on-time), while the three late brackets share the remainder, "
    "with the severest brackets the rarest. Left untreated, this imbalance lets "
    "a classifier achieve high accuracy by always predicting the majority class "
    "while failing entirely on the late invoices that matter operationally. It "
    "is this property that necessitates the resampling strategies evaluated in "
    "the benchmark — SMOTE [41], Borderline-SMOTE [42], SMOTE-Tomek [43], and "
    "threshold-based hybrid undersampling — and that motivates macro-averaged "
    "F1, which weights all four classes equally, as the primary evaluation "
    "metric, consistent with standard practice in imbalanced classification "
    "[44]."), BODY)

# --------------------------------------------------- Task 3.2 — Figure 3.5
p_analysis = find_par(doc, "The last module of the framework is the analysis phase")
set_par_text(p_analysis, (
    "The last module of the framework is the analysis phase, presented in "
    "Figure 3.5. After the benchmark completes, the analysis component "
    "aggregates the logged experiments for evaluation and comparison."))
cap35 = find_par(doc, "Figure 3.: Level-1 DFD of the Analysis component",
                 style="Caption")
set_par_text(cap35, "Figure 3.5: Level-1 DFD of the Analysis component.")
insert_par_after(doc, cap35._p, (
    "The analysis component reads the per-experiment metrics, feature "
    "importance scores, and class mappings from the results database, ranks "
    "configurations by macro-F1 and ROC-AUC, and renders the comparative "
    "figures presented in Chapter 4. Keeping analysis decoupled from training "
    "means new experiments extend the same results store without rerunning "
    "prior configurations, and every figure in Chapter 4 is reproducible from "
    "the 1,092 logged experiments."), BODY)

# ------------------------------------------------- Tasks 3.4 / 3.5 — Ethics
ch4 = find_par(doc, "CHAPTER 4:", style="Heading 1")
chain = [
    ("Heading 2", "3.7 Ethical Considerations"),
    (BODY,
     "Because the study processes sensitive student financial records and "
     "produces predictions about identifiable behavior, two ethical dimensions "
     "require explicit treatment: the privacy of the data used to train the "
     "models, and the manner in which the resulting predictions may be used."),
    ("Heading 3", "3.7.1 Data Privacy"),
    (BODY,
     "The dataset contains sensitive student financial records, and privacy was "
     "addressed before any analysis took place. All student identifiers were "
     "pseudonymized using a deterministic, irreversible hash implemented in a "
     "dedicated pseudonymization utility in the project codebase; no real names "
     "or student numbers appear in any processed file, and the mapping cannot "
     "be reversed from the published artifacts. Raw data files are never "
     "committed to version control — their exclusion is enforced through the "
     "repository's ignore rules — so the source records exist only within the "
     "institution's systems and the researchers' controlled working "
     "environment. These measures place the study in compliance with the Data "
     "Privacy Act of 2012 (Republic Act 10173) [46], which governs the "
     "processing of personal information in the Philippines and requires that "
     "processing be limited to declared, legitimate purposes with proportionate "
     "safeguards. Data access was granted under written institutional "
     "authorization, reproduced as Appendix G, which defines the scope of data "
     "access and the restrictions on its use. Finally, the predictive model "
     "outputs are intended for institutional receivables management only; they "
     "are not designed or released for individual student profiling for any "
     "public purpose."),
    ("Heading 3", "3.7.2 Ethical Use of Predictions"),
    (BODY,
     "There is an inherent risk that machine learning predictions of payment "
     "delinquency could be used to discriminate against students from families "
     "with poor payment histories, and the researchers acknowledge this risk "
     "explicitly. Three mitigations are recommended wherever the system is "
     "deployed. First, predictions should inform early outreach and support — "
     "earlier reminders, proactive installment restructuring, or referral to "
     "social welfare assistance — and never punitive action against the "
     "student. Second, any intervention workflow built on the model must comply "
     "with Republic Act 11984 [7], which prohibits academic penalties, "
     "including examination denial, for outstanding balances; the system's "
     "outputs therefore cannot lawfully gate any academic activity. Third, the "
     "model should be audited periodically for demographic bias, comparing "
     "error rates across student segments, and retrained where disparities "
     "emerge. The study's framing is deliberately supportive rather than "
     "punitive: the goal is to enable schools to offer targeted payment "
     "assistance earlier, not to flag students for enforcement, and this "
     "orientation is embedded in the recommendations of Chapter 5."),
]
for style, text in chain:
    insert_par_before(doc, ch4._p, text, style)

save(doc)
print("Phase 3 applied.")
