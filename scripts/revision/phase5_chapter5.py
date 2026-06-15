"""Phase 5 — Chapter 5 revisions (Tasks 5.1, 5.2).

- 5.2: expand 5.4 Future Directions (SHAP, time-varying Cox, federated
  learning, Dash prototype extension incl. Kaggle adapter, public-dataset
  testing, annual recalibration cadence)
- 5.1: add 5.5 Conclusion (~450 words; demonstrates, not summarizes)
"""
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          insert_par_before, set_par_text, BODY)

doc = load()
assert not par_exists(doc, "5.5 Conclusion"), "phase 5 already applied"

# ---------------------------------------------------------------- Task 5.2
fd = find_par(doc, "Future research should explore: (1) the inclusion of time-varying")
set_par_text(fd, (
    "Future research and development should proceed along six directions. "
    "First, explainable AI integration: a SHAP value layer over the deployed "
    "two-stage model would let administrators see why an individual invoice is "
    "flagged as high risk, increasing the administrative trust on which "
    "adoption depends. Second, time-varying covariates within the Cox model "
    "would allow hazard estimates to update as partial payments arrive during "
    "the term, rather than fixing survival features at invoice issuance. "
    "Third, multi-institution federated learning would improve model "
    "generalizability while preserving student data privacy, since schools "
    "could train shared models without exchanging raw records. Fourth, the "
    "Dash prototype should be extended beyond its MVP scope — most notably "
    "through the Kaggle dataset compatibility adapter described in Sections "
    "4.6 and 4.7, a mapping layer that would translate external invoice "
    "schemas onto the input format expected by the feature engineering "
    "pipeline, together with additional data source integrations beyond the "
    "three-file institutional export. Fifth, and enabled by that adapter, the "
    "pipeline should be tested on publicly available accounts-receivable and "
    "invoice datasets to establish generalizability beyond the partner "
    "institution. Finally, deployments should adopt the retraining cadence "
    "recommended in Section 5.3: an annual recalibration of the Cox PH model "
    "and downstream classifiers, so that survival features and decision "
    "thresholds continue to track each new cohort's payment behavior."))

# ---------------------------------------------------------------- Task 5.1
app_head = find_par(doc, "APPENDICES", style="Heading 1")
conclusion = [
    ("Heading 2", "5.5 Conclusion"),
    (BODY,
     "This study demonstrated that hierarchical ensemble decomposition is a "
     "viable architectural paradigm for the Invoice Payment Prediction "
     "Problem, outperforming both ordinal classifiers and single-stage "
     "baselines across the majority of resampling and feature regimes tested. "
     "Across 1,092 controlled configurations, the two-stage "
     "XGBoost-to-AdaBoost pipeline established the performance ceiling at a "
     "macro-F1 of 0.6003 and ROC-AUC of 0.8919, and the experiments further "
     "demonstrated two structural results: that granular, line-item behavioral "
     "features — a student's previous bracket, weighted payment history, and "
     "carried balance — are the dominant predictive signal, and that "
     "survival-analysis feature engineering has conditional rather than "
     "universal utility, helping probabilistic and distance-based learners "
     "while adding little to high-capacity tree ensembles."),
    (BODY,
     "These numbers have a concrete operational meaning for the partner "
     "school, connecting back to the significance articulated in Section 1.5. "
     "A macro-F1 of 0.60 on a four-class, heavily imbalanced problem means "
     "that if one hundred invoices fall due next month, the model places "
     "roughly sixty of the eventually-late ones in their correct delay "
     "bracket, and the confusion analysis in Section 4.4 shows that most of "
     "its remaining errors land in adjacent brackets rather than at the "
     "opposite extreme. Measured against the status quo — no forward "
     "visibility into receivables at all — this allows the finance office to "
     "concentrate outreach on the highest-risk invoices before due dates "
     "lapse: earlier reminders, proactive restructuring offers, and social "
     "welfare referrals, in place of after-the-fact collection that RA 11984 "
     "has rendered largely unenforceable. Prediction, in other words, converts "
     "bad-debt exposure into earlier and gentler intervention."),
    (BODY,
     "Beyond the single institution, the contribution is methodological. The "
     "study leaves behind a replicable benchmarking framework — a factorial "
     "protocol over model families, balancing strategies, and feature regimes, "
     "with every experiment logged to a queryable results store — together "
     "with a working prototype that operationalizes the winning configuration. "
     "Any institution facing the IPPP, educational or otherwise, can rerun the "
     "same protocol on its own records and obtain a defensible, evidence-based "
     "model selection rather than an imported assumption about what works."),
    (BODY,
     "Predictive receivables management, finally, should be read as an "
     "instrument of institutional responsibility rather than financial "
     "surveillance. A school that can see payment difficulty coming can keep "
     "its programs funded without leaning on enforcement mechanisms the law "
     "has rightly removed, and can direct assistance to the families who need "
     "it while there is still time to help. In aligning financial "
     "sustainability with educational equity, this study argues, predictive "
     "analytics becomes part of responsible governance — and the gap between "
     "what schools can know and what they currently act on is precisely where "
     "data science has the most to offer."),
]
for style, text in conclusion:
    insert_par_before(doc, app_head._p, text, style)

save(doc)
print("Phase 5 applied.")
