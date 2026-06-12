"""Phase 1 — Chapter 1 revisions (Tasks 1.1, 1.2, 1.3, 1.4, 8.1).

- Expand opening introduction with survey of existing IPPP models (1.1)
- Insert Table 1.1 comparative table of existing approaches (1.4)
- Number the unnumbered section headings 1.1-1.3
- Insert 1.5 Significance of the Study (1.2)
- Insert 1.6 SDG alignment section (8.1); Scope renumbered to 1.7
- Convert limitations enumeration to flowing prose (1.3)

New provisional citation used here: [47] = United Nations (2015) SDG framework
(added to the References list in the references phase).
"""
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          insert_table_after, set_par_text, BODY, CAPTION)

doc = load()
assert not par_exists(doc, "1.5 Significance of the Study"), "phase 1 already applied"

# ---------------------------------------------------------------- Task 1.1
opening = find_par(doc, "Machine learning (ML), as a branch of artificial intelligence")

p2 = insert_par_after(doc, opening._p, (
    "A substantial body of machine learning research has addressed the IPPP and "
    "adjacent receivables-prediction problems, employing models such as Logistic "
    "Regression, Random Forest, XGBoost, LightGBM, and neural networks. Schoonbee, "
    "Moore, and van Vuuren [8] applied a machine-learning decision-support approach "
    "to invoice payment prediction in an educational setting and reported "
    "classification accuracies of approximately 80–85% using tree-based "
    "ensembles, while Moore and van Vuuren [4] combined survival analysis with "
    "machine learning in their Modelling Invoice Payment Predictions (MIPP) "
    "framework to forecast the settlement timing of customer invoices. In the "
    "educational domain, Mugorobin et al. [3] developed an estimation system for "
    "late payment of school tuition fees, and Martikainen [5] demonstrated that "
    "statistical learning methods such as logistic regression can predict late "
    "payment of sales invoices with actionable discrimination. At the customer "
    "level, Cheong [29] and Appel et al. [28] showed that ensemble models trained "
    "on transaction histories can reduce unnecessary collection interventions, and "
    "Abbas and Hussein [12] confirmed that modern boosting algorithms such as "
    "XGBoost and LightGBM significantly outperform conventional statistical "
    "baselines in loan default prediction."), BODY)

p3 = insert_par_after(doc, p2._p, (
    "Despite these advances, existing approaches share three limitations: first, "
    "they model payment behavior using aggregate customer- or student-level "
    "features, discarding the line-item granularity at which invoices are actually "
    "issued; second, they treat payment delay as a binary or nominal multi-class "
    "target, ignoring the inherent ordering of delay brackets; and third, they "
    "benchmark only a narrow set of five to eight standard classifiers, leaving "
    "ordinal decompositions, hierarchical two-stage architectures, and "
    "survival-analysis-derived features systematically untested. This study "
    "addresses these gaps by introducing a multi-stage pipeline across 1,092 "
    "experimental configurations."), BODY)

# ---------------------------------------------------------------- Task 1.4
intro = insert_par_after(doc, p3._p, (
    "Table 1.1 summarizes the most directly comparable invoice payment prediction "
    "studies, the datasets and models they employed, their reported performance, "
    "and the principal limitation each leaves unaddressed."), BODY)
cap = insert_par_after(doc, intro._p,
    "Table 1.1. Summary of existing machine learning approaches to invoice "
    "payment prediction.", CAPTION)
tbl = insert_table_after(doc, cap._p, [
    ["Study", "Dataset Type", "Models Used", "Best Accuracy / F1",
     "Key Limitation"],
    ["Moore & van Vuuren (2020) [4]", "Corporate customer invoices",
     "Survival analysis + ML (MIPP framework)",
     "Time-to-payment estimates; bracket-level F1 not reported",
     "Aggregate customer features; no ordinal target"],
    ["Schoonbee et al. (2021) [8]",
     "Educational institution invoices (South Africa)",
     "Logistic Regression, Random Forest, XGBoost, Neural Networks",
     "Accuracy ≈ 80–85%",
     "Broad student-level features; single-stage classifiers only"],
    ["Mugorobin et al. (2020) [3]", "School tuition fee records",
     "Rule-based estimation with classification",
     "Not reported on standardized metrics",
     "Heuristic system; no ensemble benchmarking"],
    ["Martikainen (2023) [5]", "B2B sales invoices (Finland)",
     "Logistic Regression, statistical learning",
     "Moderate discrimination (AUC-based)",
     "Few classifiers; no class-imbalance treatment"],
    ["Cheong (2022) [29]", "Corporate accounts receivable",
     "Customer-level gradient boosting",
     "Reduced intervention actions; per-class F1 not reported",
     "Customer-level aggregation; no line-item granularity"],
])
insert_par_after(doc, tbl._tbl, (
    "Two patterns emerge from Table 1.1. Across studies, predictive accuracy "
    "clusters around 80–85% for binary or coarse multi-class formulations, "
    "yet none of the cited works decomposes performance across ordered delay "
    "brackets, and none operates at the granularity of individual fee categories. "
    "These omissions motivate the granular, ordinal, and multi-stage design "
    "benchmarked in this study."), BODY)

# ------------------------------------------- Number the unnumbered headings
set_par_text(find_par(doc, "Context of the Study", style="Heading 2"),
             "1.1 Context of the Study")
set_par_text(find_par(doc, "The Problem, Gap, or Opportunity", style="Heading 2"),
             "1.2 The Problem, Gap, or Opportunity")
set_par_text(find_par(doc, "Research Questions", style="Heading 2"),
             "1.3 Research Questions")

# ------------------------------------------------- Tasks 1.2 and 8.1
scope_h = find_par(doc, "1.5 Scope and Limitations", style="Heading 2")
set_par_text(scope_h, "1.7 Scope and Limitations")

# Insert 1.5 and 1.6 before the (now) 1.7 Scope heading, by inserting after the
# last paragraph of Section 1.4 (i.e., immediately before scope heading).
sig_h = insert_par_after(doc, scope_h._p, "", style=None)  # placeholder; will move
# simpler: build the chain before scope by inserting before it in reverse order
from docx_helpers import insert_par_before, delete_par
delete_par(sig_h)

chain = []
chain.append(("Heading 2", "1.5 Significance of the Study"))
chain.append((BODY,
    "This study offers direct operational value to the partner institution. The "
    "bad-debt and collection trends documented in Table 2.1 — bad debts "
    "expense rising from PHP 248,650 in school year 2019 to PHP 352,075 in 2025, "
    "with Days Sales Outstanding persisting between 36 and 49 days in recent "
    "years — indicate that reactive collection practices have not contained "
    "receivable risk. By predicting the payment bracket of every invoice at "
    "issuance, the system enables finance officers to anticipate cash shortfalls, "
    "to target reminders and restructuring offers before due dates lapse, and to "
    "reduce bad-debt exposure by shifting from reactive to proactive cash flow "
    "management. Beyond the partner school, the framework is transferable to "
    "Philippine private schools broadly: such institutions share substantially "
    "similar installment-billing structures and operate under the same enforcement "
    "constraints imposed by Republic Act 11984 [7], so any school capable of "
    "exporting standard revenue and enrollment records can retrain the pipeline "
    "on its own data without architectural modification."))
chain.append((BODY,
    "For the academic community, this is the first study to benchmark ordinal "
    "classifiers and two-stage ensemble architectures specifically for the "
    "educational Invoice Payment Prediction Problem, directly addressing the "
    "methodological gaps identified in Schoonbee et al. [8] and Moore and van "
    "Vuuren [4]. For data science practice more generally, the study demonstrates "
    "that survival-analysis feature engineering has conditional rather than "
    "universal utility: Cox-derived hazard features improve distance-based and "
    "probabilistic learners while adding redundant or noisy signal to "
    "high-capacity tree ensembles. This finding carries implications beyond the "
    "IPPP, suggesting that feature engineering effort should be allocated "
    "according to the inductive capacity of the downstream learner rather than "
    "applied uniformly across model families."))
chain.append(("Heading 2",
    "1.6 Alignment with United Nations Sustainable Development Goals"))
chain.append((BODY,
    "This research contributes to several of the United Nations Sustainable "
    "Development Goals (SDGs) articulated in the 2030 Agenda for Sustainable "
    "Development [47]. With respect to SDG 4 (Quality Education), the prediction "
    "system protects an institution's capacity to deliver quality education by "
    "helping it maintain financial sustainability: schools facing cash flow "
    "crises may be forced to cut staff, reduce learning resources, or close "
    "programs, so proactive receivables management directly protects educational "
    "continuity. With respect to SDG 8 (Decent Work and Economic Growth), "
    "predictive accounts-receivable management reduces financial uncertainty for "
    "small and medium-sized private educational institutions, supporting their "
    "role as employers and contributors to local economic activity."))
chain.append((BODY,
    "The study likewise advances SDG 10 (Reduced Inequalities), because the model "
    "enables schools to identify at-risk families earlier and to offer targeted "
    "payment assistance — installment restructuring or referral for social "
    "welfare certification — rather than punitive enforcement, consistent "
    "with the equity-oriented intent of Republic Act 11984 [7]. With respect to "
    "SDG 16 (Peace, Justice, and Strong Institutions), the use of pseudonymized "
    "data, compliance with the Data Privacy Act of 2012, and a transparent, fully "
    "logged machine learning methodology demonstrate responsible institutional "
    "governance and data stewardship. Finally, in support of SDG 17 (Partnerships "
    "for the Goals), the open benchmarking framework and the accompanying web "
    "prototype are designed to be replicable by other institutions, supporting "
    "knowledge sharing and institutional capacity building across the sector."))

for style, text in chain:
    insert_par_before(doc, scope_h._p, text, style)

# ---------------------------------------------------------------- Task 1.3
lim = find_par(doc, "Limitations include: (1)")
set_par_text(lim, (
    "Several limitations bound the interpretation of these results. The study "
    "draws on data from a single institution, which may limit generalizability to "
    "schools with significantly different demographic or socioeconomic profiles. "
    "The predictive models are trained on historical journal entries and "
    "therefore cannot account for real-time external economic shocks, such as "
    "localized inflation or family-level crises, that are not captured in the "
    "pseudonymized financial records. Finally, the survival features are derived "
    "from a Cox Proportional Hazards model, which assumes a linear relationship "
    "between covariates and the log-hazard and may consequently miss non-linear "
    "temporal interactions."))

save(doc)
print("Phase 1 applied.")
