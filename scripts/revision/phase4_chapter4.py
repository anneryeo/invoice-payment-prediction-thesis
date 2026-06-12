"""Phase 4 — Chapter 4 revisions (Tasks 4.1–4.4).

- 4.1: retitle sections to reference RQ1–RQ4; carve out new 4.2 (survival
  features) and 4.5 (granular feature contribution); add chapter intro
  mapping RQs to sections
- 4.2: intro + discussion paragraphs for Table 4.1 and Figures 4.1–4.8;
  Table 4.1 caption moved above and restyled; figure captions made literal;
  LDA paragraph added in 4.5
- 4.3: new 4.6 Prototype Demonstration (MVP screens A–E, screenshot
  placeholders, deferred Kaggle adapter)
- 4.4: new 4.7 Benchmark Dataset Generalizability
"""
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          insert_par_before, set_par_text, delete_par,
                          move_before, BODY, CAPTION)
from docx.oxml.ns import qn

doc = load()
assert not par_exists(doc, "4.6 Prototype Demonstration"), "phase 4 already applied"


def figure_par_before(cap):
    el = cap._p.getprevious()
    while el is not None:
        if el.tag == qn("w:p") and el.findall(".//" + qn("w:drawing")):
            return el
        el = el.getprevious()
    raise LookupError("no figure paragraph before caption")


# ------------------------------------------------------- Chapter intro (4.1)
ch_intro = find_par(doc, "The experimental results from 1,092 benchmark configurations")
insert_par_after(doc, ch_intro._p, (
    "This chapter is organized around the four research questions posed in "
    "Section 1.3. Section 4.1 addresses RQ1 by comparing two-stage and ordinal "
    "ensemble architectures against single-stage base classifiers. Section 4.2 "
    "addresses RQ2, isolating the marginal contribution of "
    "survival-analysis-derived features. Section 4.3 addresses RQ3, evaluating "
    "seven class-balancing strategies across all model families. Section 4.4 "
    "examines the convergence of the two headline metrics and the resulting "
    "model selection criteria, and Section 4.5 addresses RQ4 by analyzing which "
    "granular features most consistently drive payment-bracket predictions. The "
    "chapter closes with a demonstration of the prototype system that "
    "operationalizes the best-performing model (Section 4.6) and an assessment "
    "of the pipeline's generalizability to public benchmark datasets "
    "(Section 4.7)."), BODY)

# ------------------------------------------------------------- Section 4.1
set_par_text(find_par(doc, "4.1 Model Family Comparison", style="Heading 2"),
             "4.1 RQ1 — Performance of Ensemble Architectures vs. Base "
             "Classifiers")

# Table 4.1: caption above, restyled; intro before; discussion after
t41_cap = find_par(doc, "Table 4.1. Peak performance per model family")
t41_cap.style = doc.styles[CAPTION]
el = t41_cap._p.getprevious()
while el is not None and el.tag != qn("w:tbl"):
    el = el.getprevious()
assert el is not None
move_before(t41_cap._p, el)
insert_par_before(doc, t41_cap._p, (
    "Table 4.1 presents the peak macro-F1 and ROC-AUC scores for each model "
    "family under the enhanced feature regime, identifying the best individual "
    "model and resampling strategy per family."), BODY)
insert_par_after(doc, el, (
    "The two-stage family leads on both headline metrics: its best "
    "configuration (two_stage_xgb_ada under Borderline-SMOTE) reaches a "
    "macro-F1 of 0.6003 and ROC-AUC of 0.8919, exceeding the best ordinal model "
    "by approximately 0.038 F1 and the best base classifier by approximately "
    "0.041 F1. With respect to RQ1, this gap indicates that architectural "
    "decomposition — separating the on-time/late decision from delay-severity "
    "classification — yields a larger improvement than either exploiting class "
    "ordering alone (the ordinal family) or tuning strong single-stage "
    "learners, though the modest margins also show that base ensembles remain "
    "competitive when paired with well-chosen resampling."), BODY)

# Figure 4.1: merge split caption, literal number, intro before image
cap41a = find_par(doc, "Figure 4.: Ordinal & Two-Stage Variants vs", style="Caption")
cap41b = find_par(doc, "Base Classifiers (A lift in F1 and AUC metrics)", style="Caption")
set_par_text(cap41a, "Figure 4.1: Ordinal & Two-Stage Variants vs. Base "
                     "Classifiers (lift in F1 and AUC metrics).")
delete_par(cap41b)
fig41 = figure_par_before(cap41a)
insert_par_before(doc, fig41, (
    "Figure 4.1 visualizes the full F1 and AUC lift of ordinal and two-stage "
    "variants over their corresponding base classifiers under the enhanced "
    "feature regime, making both the gains and the exceptions visible at a "
    "glance."), BODY)

# ------------------------------------------------- New Section 4.2 (RQ2)
bal_head = find_par(doc, "4.2 Impact of Class Balancing", style="Heading 2")
rq2_head = insert_par_before(doc, bal_head._p,
    "4.2 RQ2 — Impact of Survival-Analysis-Derived Features", "Heading 2")

surv_pars = [
    find_par(doc, "The inclusion of survival-derived features produced"),
    find_par(doc, "(a) Positive Lift: Distance-based (KNN) and probabilistic"),
    find_par(doc, "(b) Negligible-to-Negative Lift: High-capacity tree-based"),
    find_par(doc, "This finding suggests a conditional feature engineering strategy"),
]
anchor = rq2_head._p
for p in surv_pars:
    anchor.addnext(p._p)
    anchor = p._p
set_par_text(surv_pars[3], (
    "This finding suggests a conditional feature engineering strategy: survival "
    "features should be prioritized for simpler, faster models but may be "
    "omitted for high-capacity ensembles to reduce computational overhead. The "
    "magnitude of the contribution is quantified visually by comparing the "
    "baseline-regime distributions in Figure 4.4 against their enhanced-regime "
    "counterparts in Figure 4.2, and the feature importance analysis presented "
    "in Section 4.5 (Figures 4.5 and 4.6) confirms the relative weight of "
    "survival features across model types."))

# ------------------------------------------------------------- Section 4.3
set_par_text(bal_head, "4.3 RQ3 — Effect of Class-Balancing Strategies on "
                       "Model Performance")

cap42 = find_par(doc, "Figure 4.: F1 Macro by Model & Balance Strategy (enhanced",
                 style="Caption")
fig42 = figure_par_before(cap42)
insert_par_before(doc, fig42, (
    "Figure 4.2 presents macro-F1 distributions across all model types and "
    "balancing strategies under the enhanced feature regime."), BODY)
set_par_text(cap42, "Figure 4.2: F1 Macro by Model & Balance Strategy "
                    "(enhanced features).")
insert_par_after(doc, cap42._p, (
    "Borderline-SMOTE wins most consistently across model types, matching or "
    "exceeding vanilla SMOTE in nearly every column, while configurations "
    "without resampling trail for all but the strongest ensembles. The variance "
    "across strategies is itself informative: high-capacity ensembles are "
    "relatively robust to the choice of strategy, whereas KNN and Gaussian "
    "Naive Bayes swing substantially — strategy selection matters most for "
    "precisely the models least able to compensate internally."), BODY)

cap43 = find_par(doc, "Figure 4.: AUC Macro by Model & Balance Strategy (enhanced",
                 style="Caption")
fig43 = figure_par_before(cap43)
insert_par_before(doc, fig43, (
    "Figure 4.3 presents ROC-AUC distributions across all model types and "
    "balancing strategies under the enhanced feature regime."), BODY)
set_par_text(cap43, "Figure 4.3: AUC Macro by Model & Balance Strategy "
                    "(enhanced features).")
insert_par_after(doc, cap43._p, (
    "The AUC view largely mirrors the F1 ranking — the same two-stage "
    "configurations top both charts, with the best AUC of 0.882 achieved by the "
    "two-stage XGBoost-to-AdaBoost pipeline under Borderline-SMOTE — but the "
    "spread between strategies is narrower than under F1. This indicates that "
    "resampling primarily improves the placement of the decision threshold "
    "rather than the underlying separability of the classes."), BODY)

cap44 = find_par(doc, "Figure 4.: F1 Macro by Model & Balance Strategy (baseline",
                 style="Caption")
fig44 = figure_par_before(cap44)
insert_par_before(doc, fig44, (
    "Figure 4.4 presents macro-F1 distributions by model and balancing strategy "
    "under the baseline feature regime, that is, without the five "
    "survival-analysis-derived features."), BODY)
set_par_text(cap44, "Figure 4.4: F1 Macro by Model & Balance Strategy "
                    "(baseline features).")
insert_par_after(doc, cap44._p, (
    "Comparing Figure 4.4 against Figure 4.2 quantifies the survival-feature "
    "contribution discussed in Section 4.2: the distributions shift upward "
    "modestly under the enhanced regime for distance-based and probabilistic "
    "models (mean F1 lifts of +0.0050 to +0.0061 for KNN and Gaussian Naive "
    "Bayes), while tree-based ensembles are essentially unchanged. The strategy "
    "rankings, however, are stable across the two regimes — Borderline-SMOTE "
    "leads in both — indicating that resampling choice and feature regime act "
    "as largely independent levers."), BODY)

# ------------------------------------------------------------- Section 4.4
set_par_text(find_par(doc, "4.3 Convergence of Metrics", style="Heading 2"),
             "4.4 Metric Convergence and Model Selection Criteria")

cap47 = find_par(doc, "Figure 4.: ROC Curves for Top Models", style="Caption")
fig47 = figure_par_before(cap47)
insert_par_before(doc, fig47, (
    "Figure 4.7 presents ROC curves for the top-performing models, plotting "
    "the trade-off between true positive rate and false positive rate across "
    "classification thresholds."), BODY)
set_par_text(cap47, "Figure 4.7: ROC Curves for Top Models.")
insert_par_after(doc, cap47._p, (
    "The curves separate cleanly from the diagonal chance line, with the "
    "two-stage XGBoost-to-AdaBoost configuration achieving the largest area "
    "under the curve (approximately 0.89). Its advantage is most pronounced in "
    "the low-false-positive region, which is operationally the relevant regime: "
    "a finance office acting on predictions wants high recall of late invoices "
    "while contacting as few on-time payers as possible."), BODY)

cap48 = find_par(doc, "Figure 4.: Confusion Matrices for Top 3 Models", style="Caption")
fig48 = figure_par_before(cap48)
insert_par_before(doc, fig48, (
    "Figure 4.8 presents the confusion matrices of the top three models, "
    "detailing classification performance per payment bracket."), BODY)
set_par_text(cap48, "Figure 4.8: Confusion Matrices for Top 3 Models.")
insert_par_after(doc, cap48._p, (
    "The matrices reveal a consistent class-level pattern: Class 0 (on-time) "
    "recall is uniformly high and Class 1 (1–30 days late) is recovered "
    "reasonably well, but Classes 2 and 3 remain difficult under every "
    "strategy, with a substantial share of 31–60 day invoices absorbed into "
    "neighboring brackets. The errors are predominantly ordinal-adjacent — "
    "misclassifications land one bracket away rather than at the opposite "
    "extreme — which moderates their operational cost, since an invoice flagged "
    "one bracket early still receives an intervention."), BODY)

# ------------------------------------------------- New Section 4.5 (RQ4)
last44 = find_par(doc, "The relationship between F1-score and ROC-AUC is further")
rq4_head = insert_par_after(doc, last44._p,
    "4.5 RQ4 — Granular Feature Contribution", "Heading 2")
rq4_intro = insert_par_after(doc, rq4_head._p, (
    "Figures 4.5 and 4.6 present feature importance scores aggregated across "
    "model types, revealing which variables most consistently drive payment "
    "bracket predictions."), BODY)

cap45 = find_par(doc, "Figure 4.: Feature Importance across Model Types (1 of 2)",
                 style="Caption")
fig45 = figure_par_before(cap45)
cap46 = find_par(doc, "Figure 4.: Feature Importance across Model Types (2 of 2)",
                 style="Caption")
fig46 = figure_par_before(cap46)
# move the two figure/caption pairs into the new section
rq4_intro._p.addnext(fig45)
fig45.addnext(cap45._p)
cap45._p.addnext(fig46)
fig46.addnext(cap46._p)
set_par_text(cap45, "Figure 4.5: Feature Importance across Model Types (1 of 2).")
set_par_text(cap46, "Figure 4.6: Feature Importance across Model Types (2 of 2).")
disc45 = insert_par_after(doc, cap46._p, (
    "Three features dominate across model families: prev_bracket, the payment "
    "bracket of the student's most recent invoice; dtp_wavg, the "
    "recency-weighted average of historical days-to-payment; and "
    "opening_balance_flag, indicating whether the student carried an unpaid "
    "balance into the period. All three are behavioral-historical features "
    "engineered at the granular, line-item level described in Chapter 3, "
    "confirming the study's central design rationale with respect to RQ4: a "
    "student's payment future is best predicted by a compact summary of their "
    "payment past, and that summary is only available when invoices are tracked "
    "at category-level granularity rather than as aggregate balances."), BODY)
insert_par_after(doc, disc45._p, (
    "The exploratory linear discriminant analysis corroborates this ranking "
    "from a different angle. A four-class LDA on the training set finds that "
    "the first discriminant axis (LD1) explains 79.5% of between-class "
    "separation variance, and the features loading most heavily on it are "
    "opening_balance_flag, the log-transformed opening_balance, prev_bracket, "
    "and dtp_wavg — essentially the same compact behavioral set identified by "
    "the supervised importances. When the analysis is restricted to the three "
    "delinquent classes, LD1 explains 95.9% of separation, indicating that "
    "delay severity behaves as a nearly one-dimensional behavioral construct. "
    "The agreement between this projection-based view and the supervised "
    "importance rankings strengthens confidence that these variables capture "
    "genuine structure rather than model-specific artifacts."), BODY)

# ------------------------------------------------- Section 4.6 (Task 4.3)
ch5 = find_par(doc, "CHAPTER 5:", style="Heading 1")
sec46 = [
    ("Heading 2", "4.6 Prototype Demonstration"),
    (BODY,
     "The benchmarking results are operationalized in a working prototype: a "
     "Dash (Python/Plotly) web application that serves as the study's delivery "
     "vehicle for institutional users. The prototype is best characterized as a "
     "Minimum Viable Product (MVP): a functional, working system that "
     "demonstrates the core workflow end to end — from raw data upload through "
     "model training to invoice-level prediction — while deferring selected "
     "non-core integrations to future development. An administrator interacts "
     "with five principal screens, described below. [Screenshot placeholder: "
     "insert captures of each screen once the application is running.]"),
    (BODY,
     "The entry point is a multi-step Initial Setup Wizard that guides the "
     "administrator through uploading the three institutional source files "
     "(revenues, enrollees, and chart of accounts) and triggering the training "
     "pipeline, removing any need to interact with code or notebooks. Once "
     "training has been run, the KPI Dashboard loads live data from the results "
     "database on mount: four indicator cards report the best model's macro-F1, "
     "its ROC-AUC, the total number of logged experiments, and the name of the "
     "currently deployed model, alongside a payment-bracket distribution chart "
     "built from the processed invoice cache and a table of top models sorted "
     "by enhanced-regime F1. The screen falls back gracefully to an empty state "
     "when no training has yet been performed. For a school administrator, this "
     "view answers the first operational question — how reliable is the "
     "deployed model and what does the receivables mix look like — at a "
     "glance."),
    (BODY,
     "The Invoice Prediction Drilldown is the screen where predictions become "
     "actionable. It loads the processed credit sales cache, runs the deployed "
     "inference pipeline's class and probability predictions across all invoice "
     "records, and displays a paginated table of invoice number, due date, "
     "amount, predicted payment bracket, prediction confidence, and the actual "
     "bracket where known. The administrator can filter the table by predicted "
     "bracket — isolating, for example, all invoices expected to fall 61 or "
     "more days late — and export the result to CSV for use in collection "
     "workflows; every prediction run is recorded in the audit log. The "
     "Comparative Model Dashboard provides the analytical depth behind the "
     "deployment choice, allowing filtering by model type and balance strategy "
     "and rendering ROC curves, confusion matrices, and F1/AUC comparison "
     "charts across all 1,092 experiment configurations."),
    (BODY,
     "Two supporting screens round out the MVP. The Audit Logs screen lists "
     "all system events — predictions run, settings saved, models loaded — "
     "with timestamp, action, and details, refreshing automatically every 30 "
     "seconds; this provides the accountability trail required for responsible "
     "institutional use (Section 3.7). The Settings screen exposes the "
     "undersample threshold, the default balance strategy, and the late-invoice "
     "cutoff, persisting preferences to a configuration file and logging each "
     "change. All five screens are implemented and functional in the MVP. One "
     "integration is deliberately deferred: a compatibility adapter that would "
     "map external dataset schemas — such as public Kaggle invoice datasets "
     "with columns like invoice_date, due_date, amount, customer_id, and "
     "paid_date — onto the input schema expected by the feature engineering "
     "pipeline. At present only the proprietary three-file institutional schema "
     "is supported; the adapter and additional data source integrations are "
     "planned for future development (Section 5.4)."),
]
for style, text in sec46:
    insert_par_before(doc, ch5._p, text, style)

# ------------------------------------------------- Section 4.7 (Task 4.4)
sec47 = [
    ("Heading 2", "4.7 Benchmark Dataset Generalizability"),
    (BODY,
     "The pipeline as implemented is designed around a specific institutional "
     "schema: the three Excel exports (revenues, enrollees, chart of accounts) "
     "described in Section 3.2. Its strong results therefore demonstrate "
     "effectiveness for the partner institution's data model, but not yet "
     "generalizability beyond it. The natural validation step is to benchmark "
     "the pipeline against publicly available invoice and accounts-receivable "
     "datasets, such as those published on Kaggle under searches for \"B2B "
     "invoice payment prediction,\" \"accounts receivable dataset,\" or "
     "\"customer payment behavior.\" Doing so requires the compatibility "
     "adapter described in Section 4.6: a mapping layer that translates "
     "external columns onto the schema expected by the feature engineering "
     "pipeline and degrades gracefully where institution-specific variables — "
     "installment plan types, enrollment streaks, category-level fee structure "
     "— have no external equivalent. Because several of the study's strongest "
     "features are institution-specific, such benchmarking would also reveal "
     "how much of the measured performance derives from the granular school "
     "data model itself, which is an informative result in either direction. "
     "This evaluation is explicitly flagged as planned future work rather than "
     "a completed contribution of this study (Section 5.4)."),
]
for style, text in sec47:
    insert_par_before(doc, ch5._p, text, style)

save(doc)
print("Phase 4 applied.")
