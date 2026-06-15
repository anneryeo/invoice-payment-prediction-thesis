"""Phase 2 — Chapter 2 revisions (Tasks 2.1–2.6).

- 2.1: convert bulleted chapter-structure list to two prose paragraphs
- 2.2: add narrative transitions at section boundaries (2.2→2.3→2.4, 2.5→2.6,
  gap statement in 2.6, post-table motivation paragraph in 2.4)
- 2.3: insert new "2.13 Comparative Analysis of Existing Technologies" with an
  8-study table; renumber Theoretical Framework to 2.14 and fix its figure
  (image moved out of the heading, intro above, literal caption, discussion below)
- 2.4: expand 2.9.1, 2.9.2, 2.9.3, 2.10, 2.11
- 2.5: map literature gaps to research objectives at end of 2.12
- 2.6: expand Philippine local studies (2.8)

Provisional citation used here: [45] = Cox (1972).
"""
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          insert_par_before, insert_table_after, set_par_text,
                          delete_par, append_text, BODY, CAPTION)
from docx.oxml.ns import qn

doc = load()
assert not par_exists(doc, "2.13 Comparative Analysis"), "phase 2 already applied"

# ---------------------------------------------------------------- Task 2.1
intro1 = find_par(doc, "This chapter reviews the body of literature")
set_par_text(intro1, (
    "This chapter reviews the body of literature and studies relevant to the "
    "prediction of invoice payment behaviors using machine learning (ML). It "
    "situates the study within the broader field of financial forecasting while "
    "narrowing the focus to the unique challenges faced by educational "
    "institutions. The review begins by establishing the institutional context of "
    "the study, including the financial and operational realities of the partner "
    "school, and by comparing invoicing and receivables practices across sectors "
    "to contextualize the position of educational institutions. It then examines "
    "the role of ML in financial forecasting and its application to invoice "
    "payment prediction, highlighting the importance of granular, student- and "
    "product-specific variables in improving predictive accuracy."))
intro2 = insert_par_after(doc, intro1._p, (
    "The chapter subsequently discusses the issue of tuition debt collection in "
    "learning institutions, covering industry practices, school-level strategies, "
    "and the operational consequences of unpaid fees, and explores the "
    "socioeconomic and behavioral factors that influence payment behaviors. It "
    "then assesses existing intervention strategies and justifies the study's "
    "chosen approach, before reviewing the methodological advances in ordinal "
    "classification, two-stage ensemble architectures, and survival analysis that "
    "inform this study's expanded model set. The chapter closes by synthesizing "
    "methods, findings, and gaps in the literature to position this research "
    "within the academic discourse."), BODY)

for bullet in [
    "Establish the institutional context of the study",
    "Examine the role of ML in financial forecasting",
    "Highlight the importance of granular, student- and product-specific",
    "Discuss the issue of tuition debt collection in learning institutions",
    "Compare invoicing models across sectors",
    "Explore socioeconomic and behavioral factors",
    "Assess existing intervention strategies",
    "Review methodological advances in ordinal classification",
    "Synthesize methods, findings, and gaps in the literature",
]:
    delete_par(find_par(doc, bullet))

# ---------------------------------------------------------------- Task 2.2
# 2.2 -> 2.3 transition
append_text(find_par(doc, "These cross-sector practices reveal two insights"), (
    "The following section examines how educational institutions worldwide have "
    "responded to this challenge through their tuition collection strategies."))

# 2.3 -> 2.4 transition
append_text(find_par(doc, "These international experiences show two consistent themes"), (
    "These global pressures take a specific institutional form in the "
    "Philippines, where regulatory constraints further narrow the collection "
    "options available to private schools, as the next section details."))

# 2.4: paragraph after Table 2.1 discussion stating how the data motivate the study
p93 = find_par(doc, "These patterns reveal that, despite offering flexible payment")
insert_par_after(doc, p93._p, (
    "Taken together, the institutional record in Table 2.1 motivates the "
    "prediction problem at the center of this study. Bad debts expense has risen "
    "sharply since 2021, the number of students carrying unpaid balances has "
    "grown even as enrollment stabilized, and Days Sales Outstanding has remained "
    "in the 36–49 day range despite the availability of installment plans. "
    "Because the school can neither screen enrollees for creditworthiness nor "
    "enforce payment through academic sanctions, the only remaining managerial "
    "lever is anticipation: knowing, at the moment an invoice is issued, how "
    "likely it is to be paid late and by how much. That is precisely the granular "
    "forecasting capability this study develops."), BODY)

# 2.5 -> 2.6 bridge
append_text(find_par(doc, "Despite these advances, challenges remain. Feature selection"), (
    "While these studies address credit risk in general, the same algorithmic "
    "advances form the technical foundation for the more specific task at hand, "
    "predicting when individual invoices will be paid, which the next section "
    "reviews."))

# 2.6: explicit gap paragraph
p101 = find_par(doc, "Other studies have explored statistical and hybrid approaches")
insert_par_after(doc, p101._p, (
    "Across these studies, a specific methodological gap emerges: none of the "
    "cited works benchmarked ordinal classifiers or two-stage hierarchical "
    "architectures at scale for invoice payment prediction. Reported comparisons "
    "are limited to small sets of single-stage classifiers, typically five to "
    "eight models evaluated under one or two preprocessing regimes. This leaves "
    "open the question of whether architectures that explicitly exploit the "
    "ordered, heavily imbalanced structure of payment-delay targets can "
    "outperform conventional designs — the question this study answers "
    "across 1,092 experimental configurations."), BODY)

# ---------------------------------------------------------------- Task 2.6
p_socio = find_par(doc, "At the macro level, political and economic stressors")
insert_par_after(doc, p_socio._p, (
    "Philippine evidence sharpens this picture at the household level. Carvajal "
    "et al. [21] surveyed financial literacy and debt management practices and "
    "found that low financial literacy is associated with poor debt management, "
    "a mechanism that translates directly into erratic tuition payment: families "
    "that do not budget for recurring obligations accumulate school balances even "
    "when income is nominally sufficient. Mencias-Tabernilla [22] documented the "
    "expenditure patterns and debt profiles of Filipino teachers, showing that "
    "chronic indebtedness extends even to salaried education professionals; if "
    "the households of stable wage earners struggle with recurring debt, "
    "tuition-paying families with irregular incomes are plausibly at greater "
    "risk. Finally, the implementing context of RA 11984 [7, 13, 23] shapes "
    "behavior on both sides of the transaction: students cannot be barred from "
    "examinations over unpaid fees, while schools must channel hardship cases "
    "through social welfare certification and restructuring procedures. These "
    "local realities make payment-timing prediction more valuable in the "
    "Philippine setting than in jurisdictions where enforcement remains "
    "available."), BODY)

# ---------------------------------------------------------------- Task 2.4
# 2.9.1
set_par_text(find_par(doc, "Educational institutions typically employ administrative reminders"), (
    "Educational institutions typically employ a layered set of collection "
    "mechanisms. The most common first-line instrument is the administrative "
    "reminder campaign, ranging from printed statements of account to scheduled "
    "text-message and email reminders ahead of due dates, which practitioner "
    "literature identifies as the cheapest and most scalable intervention "
    "[32, 33]. When reminders fail, schools escalate to formal instruments: "
    "promissory notes that document a committed payment date, late-payment fees "
    "or surcharge schedules intended to price delinquency, and installment "
    "restructuring that converts an overdue lump sum into smaller scheduled "
    "payments [33, 34]. Some institutions complement these with incentives such "
    "as early-payment discounts or sibling rebates to reward timely settlement "
    "[32]. These practices, consistent with the tuition collection literature "
    "[32, 33, 34], remain the dominant standard operating procedures in many "
    "schools, including the partner institution. Their common weakness is that "
    "all of them, except early-payment incentives, are triggered only after an "
    "invoice has already become delinquent."))

# 2.9.2
set_par_text(find_par(doc, "The effectiveness of these strategies is constrained by RA 11984"), (
    "The effectiveness of these strategies is procedurally constrained by RA "
    "11984 [7, 13, 23]. The law categorically prohibits schools from denying "
    "examinations to students with outstanding balances, removing the single "
    "strongest enforcement mechanism previously available to private "
    "institutions [7]. Its implementing guidelines add further procedural "
    "requirements: students invoking financial hardship must submit "
    "certifications from the Department of Social Welfare and Development "
    "(DSWD), and schools are in turn expected to provide installment "
    "restructuring or deferred payment arrangements rather than refuse service "
    "[13, 23]. Legal commentary has warned that this combination lengthens "
    "collection cycles, extends credit exposure, and may raise accounts "
    "receivable balances unless schools adopt proactive financial management "
    "[13]. As a result, institutions are structurally bound to continue "
    "providing instruction even as receivables accumulate, which converts "
    "collections from an enforcement problem into a forecasting problem."))

# 2.9.3 — expand to two paragraphs
p293 = find_par(doc, "Given the limitations of traditional strategies, machine learning")
set_par_text(p293, (
    "Given the limitations of traditional strategies, machine learning offers a "
    "structurally different alternative: rather than reacting to delinquency "
    "after a due date has lapsed, a predictive model assigns every invoice a "
    "payment-delay risk estimate at the moment it is issued. Studies in credit "
    "scoring and receivables management [6, 24, 29, 35] demonstrate that "
    "predictive models trained on granular behavioral features consistently "
    "outperform reactive heuristics, because they convert historical payment "
    "patterns into forward-looking early-warning signals."))
insert_par_after(doc, p293._p, (
    "The operational mechanism is straightforward. An invoice predicted to fall "
    "into a late bracket can be routed to a pre-due-date intervention, such as an "
    "earlier reminder, a proactive offer of installment restructuring, or a "
    "guidance conversation with the family, while invoices predicted to be paid "
    "on time require no action at all. This targeting matters under RA 11984: "
    "since post-due-date enforcement is legally restricted, the value of knowing "
    "which invoices are at risk before they become delinquent is amplified. "
    "Cheong [29] showed that customer-level prediction reduces unnecessary "
    "intervention actions, and Schoonbee et al. [8] demonstrated decision-support "
    "value in an educational context; this study extends that logic to "
    "category-level invoices in a Philippine private school."), "Normal")

# 2.10 — expand
p210 = find_par(doc, "A fundamental property of payment-delay targets is their inherent ordering")
set_par_text(p210, (
    "A fundamental property of payment-delay targets is their inherent ordering, "
    "from on-time settlement through progressively later payment brackets. Frank "
    "and Hall [9] introduced a simple and influential approach to ordinal "
    "classification that decomposes a k-class ordinal problem into k-1 binary "
    "sub-problems: for the four payment brackets used in this study, three binary "
    "classifiers are trained to estimate P(target > on-time), P(target > 1–30 "
    "days), and P(target > 31–60 days), and the per-class probabilities are then "
    "recovered by differencing the cumulative estimates. Because each binary "
    "sub-problem preserves the ordering of the label space, the decomposition "
    "allows any probabilistic base classifier to exploit ranking information that "
    "a nominal multi-class formulation discards."))
insert_par_after(doc, p210._p, (
    "Two-stage architectures approach the same label structure from a different "
    "angle: where the Frank–Hall scheme is a decomposed view of one ordinal "
    "problem, a two-stage design is hierarchical. A first-stage binary classifier "
    "separates on-time from late invoices, directly attacking the dominant source "
    "of class imbalance, and a second-stage multi-class model, trained only on "
    "delinquent records, determines delay severity. The hierarchy mirrors "
    "institutional decision-making (first, will this invoice be late; second, "
    "how late) and allows each stage to specialize on a better-balanced "
    "sub-problem. Architectures of this kind have proven effective in credit "
    "risk [24], medical diagnosis [25], and receivables management [35], but "
    "had not previously been benchmarked at scale for the IPPP."), BODY)

# 2.11 — expand
p211 = find_par(doc, "Survival analysis provides a framework for modeling the time until an invoice")
set_par_text(p211, (
    "Survival analysis provides a statistical framework for modeling the time "
    "until an event occurs — here, the time until an invoice is fully paid. Its "
    "defining advantage over standard classification is its treatment of "
    "censoring: at any observation cutoff, some invoices remain unpaid, and their "
    "eventual payment delay is unknown rather than missing. A conventional "
    "late/not-late variable must either discard these records or mislabel them, "
    "whereas survival methods retain censored invoices and extract partial "
    "information from the fact that they have survived unpaid up to the cutoff. "
    "The Cox Proportional Hazards model [45] formalizes this by expressing each "
    "invoice's hazard rate, the instantaneous probability of payment at time t "
    "given non-payment until t, as a baseline hazard scaled by an exponential "
    "function of the invoice's covariates. Each invoice therefore receives a "
    "continuous risk profile (partial hazard, expected settlement time, survival "
    "probability at reference horizons) rather than a single binary flag, a "
    "conceptually richer representation of payment timing."))
insert_par_after(doc, p211._p, (
    "Moore and van Vuuren [4] integrated survival analysis with machine learning "
    "in the MIPP framework, using time-to-payment modeling to inform invoice "
    "predictions. This study extends that work in a specific way: instead of "
    "using the survival model as the predictor, it uses Cox PH outputs (hazard "
    "rates, survival probabilities, expected settlement times) as engineered "
    "input features for downstream classifiers, and then measures their marginal "
    "value experimentally. Prior work in credit risk suggests such features help "
    "probabilistic models more than tree ensembles [25, 35], a hypothesis this "
    "study tests directly across feature regimes."), BODY)

# ---------------------------------------------------------------- Task 2.5
p212 = find_par(doc, "ML models outperform baseline statistical approaches in forecasting")
insert_par_after(doc, p212._p, (
    "Each of these gaps maps directly onto the research objectives stated in "
    "Section 1.4. Gap (1), the lack of granular student-product variables, is "
    "directly addressed by Objective (4), which evaluates how line-item features "
    "derived from category-level invoices influence payment timing relative to "
    "aggregate student-level variables. Gap (2), the limited application of "
    "invoice prediction in private educational institutions, is addressed by "
    "Objective (1), which conducts the benchmarking on six years of records from "
    "a Philippine private school and by Objective (4)'s institution-specific "
    "feature set. Gap (3), the scarcity of ordinal IPPP benchmarks, and gap (4), "
    "the absence of two-stage architectures and systematic survival-feature "
    "testing, are addressed jointly by Objectives (1), (2), and (3): the "
    "1,092-configuration benchmark includes three ordinal decompositions and six "
    "two-stage pipelines (Objective 2) and isolates the marginal contribution of "
    "survival-analysis-derived features across all model families (Objective 3). "
    "The synthesis of the literature thus leads directly to the methodology "
    "developed in Chapter 3."), BODY)

# ---------------------------------------------------------------- Task 2.3
tf_head = find_par(doc, "2.13 Theoretical Framework", style="Heading 2")

# --- new 2.13 section inserted before the Theoretical Framework heading
sec_h = insert_par_before(doc, tf_head._p,
    "2.13 Comparative Analysis of Existing Technologies", "Heading 2")
sec_i = insert_par_before(doc, tf_head._p, (
    "Before positioning this study's methodology, it is necessary to compare the "
    "existing technologies for invoice payment prediction side by side. The "
    "studies reviewed in the preceding sections differ in domain, feature "
    "granularity, algorithmic family, and evaluation protocol, and these "
    "differences are easy to lose in narrative form. A structured comparison "
    "serves three purposes. First, it reveals which methodological choices "
    "recur across otherwise unrelated domains, distinguishing genuine best "
    "practices from domain-specific conventions. Second, it exposes the "
    "evaluation gaps, such as metrics that are rarely reported and architectures "
    "that are never tested, that limit the comparability of published results. "
    "Third, it establishes the baseline against which the contribution of the "
    "present study can be judged. Table 2.2 consolidates the eight studies most "
    "relevant to the IPPP, spanning educational, corporate, and consumer-credit "
    "domains."), BODY)
sec_c = insert_par_before(doc, tf_head._p,
    "Table 2.2. Comparative analysis of existing invoice payment prediction "
    "and related studies.", CAPTION)
rows = [
    ["Study", "Year", "Domain", "Method", "Features Used",
     "Performance Metrics", "Limitations"],
    ["Schoonbee et al. [8]", "2021", "Educational invoices (South Africa)",
     "LR, RF, XGBoost, NN decision support",
     "Student-level payment history, demographics",
     "Accuracy ≈ 80–85%",
     "Aggregate features; nominal targets"],
    ["Moore & van Vuuren [4]", "2020", "Customer invoices",
     "Survival analysis + ML (MIPP)",
     "Invoice history, customer attributes",
     "Time-to-payment estimates",
     "No ordinal brackets; no resampling study"],
    ["Mugorobin et al. [3]", "2020", "School tuition (Indonesia)",
     "Rule-based estimation with classification",
     "Tuition payment timeliness records",
     "Accuracy on small institutional sample",
     "Heuristic; no ensemble benchmarking"],
    ["Martikainen [5]", "2023", "B2B sales invoices (Finland)",
     "Statistical learning, logistic regression",
     "Invoice and customer attributes",
     "Moderate AUC discrimination",
     "Few classifiers; imbalance untreated"],
    ["Cheong [29]", "2022", "Corporate accounts receivable",
     "Customer-level gradient boosting",
     "Customer payment history, segmentation",
     "Reduction in unnecessary interventions",
     "Customer aggregation; no line-item detail"],
    ["Appel et al. [28]", "2021", "Corporate accounts receivable",
     "Supervised ML on transaction histories",
     "Transaction history, customer segmentation",
     "Precision on overdue-account detection",
     "Corporate context; binary target"],
    ["Thuy et al. [20]", "2025", "Student credit scoring (Vietnam)",
     "ML vs. deep learning comparison",
     "Academic, demographic, financial features",
     "ML ensembles competitive with deep learning",
     "Credit scoring, not invoice-level prediction"],
    ["Abbas & Hussein [12]", "2024", "Consumer loan default",
     "XGBoost, LightGBM",
     "Borrower financial attributes",
     "Boosting outperforms statistical baselines",
     "Default prediction; no payment-delay granularity"],
]
tbl = insert_table_after(doc, sec_c._p, rows)
insert_par_after(doc, tbl._tbl, (
    "Four patterns emerge from Table 2.2. First, tree-based ensembles and "
    "boosting algorithms are the consistent performance leaders across every "
    "domain in which they were tested, displacing both classical statistical "
    "models and, in the most recent comparison [20], deep learning alternatives "
    "on tabular financial data. Second, reported performance clusters in the "
    "80–85% accuracy range for binary or coarse multi-class formulations, "
    "suggesting a practical ceiling for aggregate-feature approaches. Third, "
    "every study operates on aggregate customer- or student-level features; none "
    "models receivables at the granularity of individual fee categories, despite "
    "evidence from retail analytics [30] that fine-grained features improve "
    "prediction. Fourth, and most consequentially for this study, none of the "
    "eight works evaluates ordinal decompositions or hierarchical two-stage "
    "architectures, and none systematically tests survival-derived features "
    "against resampling strategies. The present study is positioned precisely in "
    "this gap: it benchmarks fifteen architectures, including three ordinal and "
    "six two-stage designs, across seven balancing strategies and two feature "
    "regimes on category-level educational invoices."), BODY)

# --- fix the Theoretical Framework section (now 2.14): image out of heading
# Move the drawing run from the heading into its own paragraph.
drawing_runs = [r for r in tf_head._p.findall(qn("w:r"))
                if r.findall(".//" + qn("w:drawing"))]
fig_intro = insert_par_after(doc, tf_head._p, (
    "Figure 2.1 presents the conceptual framework that integrates the four "
    "theoretical lenses guiding this study, tracing how raw institutional data "
    "flow through preprocessing, predictive modeling, and decision support to "
    "inform receivables management."), BODY)
fig_par = insert_par_after(doc, fig_intro._p, "", "No Spacing")
for r in drawing_runs:
    tf_head._p.remove(r)
    fig_par._p.append(r)
set_par_text(tf_head, "2.14 Theoretical Framework")

# caption: make literal and ensure it sits after the figure paragraph
cap = find_par(doc, "Figure 2.", style="Caption")
set_par_text(cap, "Figure 2.1: Conceptual Framework.")
fig_par._p.addnext(cap._p)

save(doc)
print("Phase 2 applied.")
