# CHAPTER 3:
METHODOLOGY
## 3.1 Research Design
This study employs a quantitative, experimental research design. The research paradigm is a controlled benchmarking study: fifteen machine learning architectures are compared under identical data, split, and evaluation conditions, with the manipulated factors being model family, hyperparameter configuration, class-balancing strategy, and feature regime, a factorial space of 1,092 configurations. Experimental comparison of this kind is the standard methodology for evaluating machine learning systems, as established in large-scale credit-scoring benchmarks such as Lessmann et al. [27].
The data are observational institutional records rather than instruments administered to human subjects: the study draws on pseudonymized journal entries from a private school, so no surveys, interviews, or interventions on students were conducted. The design is therefore experimental with respect to model evaluation, not with respect to data collection. The data flow diagrams presented in Figures 3.1 through 3.5 illustrate the architecture of the developed system (its preprocessing, modeling, and analysis components) rather than the research design itself; the research design is the benchmarking protocol described in Section 3.6, which fixes the temporal train–test split, the evaluation metrics, and the experimental grid within which every architecture is assessed. The remainder of this chapter describes each component in the order data flows through the pipeline: data acquisition, granular feature engineering, survival analysis modeling, class balancing, and hierarchical ensemble classification.
## 3.2 Data Acquisition and Preprocessing
The primary dataset consists of 11,440 raw invoice records from a private school in Rizal, Philippines, spanning academic years 2019–2025 (with payment activity observed through March 31, 2026). The data originates from three Excel files manually exported from the school's internal enterprise resource planning (ERP) system. The revenues file contains itemized receivables: every billed amount with its due date, discounts, adjustments, and the dates and amounts of payments applied against it. The enrollees file records student enrollment per school year, including the installment plan selected, which allows enrollment streaks and plan-based features to be derived. The chart of accounts file maps every transaction category to its account classification and strategic business unit, enabling the category-level granularity central to this study. Each resulting record represents a discrete receivable item (e.g., Tuition, Miscellaneous Fee, E-Learning Platform) rather than an aggregate student-level balance.
Data were obtained with written institutional approval (Appendix G). All student identifiers were pseudonymized using a deterministic, irreversible hash prior to any analysis, implemented in a dedicated pseudonymization utility within the project codebase, and the original data files are excluded from the project's version-controlled repository. From the 11,440 raw records, 6,527 labeled records were retained after filtering for records with sufficient payment history to compute the days-to-payment (DTP) behavioral features; records lacking observable payment histories cannot support the lagged features the models require. The temporal train–test split is set at March 7, 2025: invoices due before this date form the training set, and invoices due on or after it form the test set.
Initial preprocessing involved three steps. First, student identifiers were pseudonymized as described above. Second, payment dates were temporally aligned against due dates to construct the target variable Days to Payment (DTP), computed as the number of days between an invoice's due date and the date its balance was fully settled, with negative or zero values denoting on-time settlement. Third, DTP was discretized into four ordinal brackets: Class 0 (On-Time, DTP <= 0), Class 1 (1–30 days late), Class 2 (31–60 days late), and Class 3 (61+ days late). Invoices still unpaid at the observation cutoff are flagged as censored and handled by the survival analysis described in Section 3.4.
*Figure 3.1 presents the entity-relationship diagram (ERD) of the institutional dataset. Three core entities (Transactions, Categories, and Enrollees) are linked through foreign keys to form the basis for feature engineering. The Transactions table captures every discrete receivable item at category-level granularity, which is the primary innovation of this study's data model compared to aggregate student-level approaches in prior work.*

*Figure 3.1: Entity-relationship diagram (ERD) of the institutional dataset.*
The relationships visible in the ERD determine how the modeling dataset is assembled. Each transaction references a category through category_id, allowing every receivable line item to be typed (tuition, miscellaneous fees, course materials, other services), while the pseudonymized student identifier links transactions to enrollment records across school years. The due_date and amount_paid fields jointly define the target variable: the gap between the due date and the date the obligation is fully offset yields Days to Payment, from which the four ordinal payment brackets are derived. The Credit Sales Fact Table consolidates these joins into a single analysis-ready view per receivable.

*Figure 3.X presents the Level-2 DFD of Process 1.0 (Data Importation), detailing how the two raw Excel uploads are decoded, typed, and datetime-normalised before being forwarded to feature engineering.*

*Figure 3.X: Level-2 DFD of Process 1.0 — Data Importation.*
The diagram decomposes Process 1.0 into four sequential sub-processes. Sub-process 1.1 performs asynchronous parallel loading of the revenue and enrollee Excel files using base64 decoding and the calamine Excel engine, reducing blocking during file reads. Sub-process 1.2 applies data type conversion to ensure numeric, categorical, and date fields conform to the types required by downstream feature engineering. Sub-process 1.3 handles datetime parsing, standardising all date representations into a uniform format from which time differences can be reliably computed. Sub-process 1.4 performs lookup-based due date updates, resolving any date overrides that originate from institutional adjustments or payment plan renegotiations. The outputs of this process are two typed and datetime-normalised DataFrames — one for revenue records and one for enrollee records — forwarded to Process 2.0 (Time to Payment) and Process 3.0 (Student Information) for feature derivation.

*Figure 3.2 presents the Level-1 data flow diagram (DFD) of the pre-processing component, which transforms the three raw institutional exports into the analysis-ready credit sales dataset.*

*Figure 3.2: Level-1 DFD of the Pre-Processing component.*
The diagram traces the main processing steps: the raw revenues, enrollees, and chart-of-accounts files are first validated and pseudonymized, then merged into category-level transactions; payment applications are temporally aligned against due dates to compute Days to Payment; and the engineered behavioral, financial, and temporal features described in Section 3.3 are appended before the labeled records are written to the modeling cache as two distinct data streams — an ML training data stream and a survival analysis data stream — forwarded to separate downstream processes. Each module is deterministic, so the same raw exports always reproduce the same modeling dataset, a property that supports the auditability objectives discussed in Section 3.7.
*Table 3.1 presents the data dictionary for the Transactions table, detailing each attribute's format and role in the preprocessing pipeline.*
*Table 3.1.  Entity data dictionary of the Transactions table.*

| Attribute Name | Format | Description |
|---|---|---|
| transaction_entry_id | AutoNumber | Automatic numbering for a unique identifier |
| entry_date | Date | The date the journal entry was entered |
| due_date | Date | The date the specific product/service category is due to be paid |
| school_year | smallInt(4) | The school year the transaction is for |
| student_id_pseudonymized | mediumInt(6) | The identifier of the student the transaction is related to |
| category_id | smallInt(4) | The category of the transaction |
| amount_due | Float | The amount needed to be paid |
| amount_paid | Float | The amount paid to offset the amount due |
| account_name | Char(30) | The account the money was sent (e.g. Cash on Hand, Bank Transfer, or GCash) |

Among these fields, due_date and amount_paid are the most consequential: together they are the source of the target variable, since Days to Payment is computed from the gap between an obligation's due date and the date it is fully offset. The category_id field is the link that enables granular line-item modeling, the central innovation of this study, while school_year and the pseudonymized student identifier allow behavioral features to be accumulated across a student's enrollment history.
*Table 3.2 presents the data dictionary for the Categories table, detailing each attribute's format and role in the preprocessing pipeline.*
*Table 3.2.  Entity data dictionary of the Categories table.*

| Attribute Name | Format | Description |
|---|---|---|
| category_id | smallInt(4) | Automatic numbering for a unique identifier |
| category_name | Char(60) | The name of the category |
| strategic_business_unit | Char(35) | The business unit the category is accounted in (e.g. Teaching, Goods, Other services, or Administrative) |

Although structurally simple, this table is what gives the pipeline its granularity: category_name distinguishes tuition from miscellaneous and product-type charges, and strategic_business_unit groups categories into the operational units used for institutional reporting. Joining these attributes onto transactions is what allows the models to learn payment behavior per fee type rather than per aggregate balance.
*Table 3.3 presents the data dictionary for the Credit Sales Fact Table, the consolidated analysis-ready view produced by the preprocessing pipeline.*
*Table 3.3.  Entity data dictionary of the Credit Sales Fact Table (FT).*

| Attribute Name | Format | Description |
|---|---|---|
| credit_entry_id | autoNumber | Automatic numbering for a unique identifier |
| student_id_pseudonymized | mediumInt(6) | The pseudonymized identifier for the student number |
| credit_sale_amount | Currency | The amount payable for the respective student id |
| due_date | Date | The date when the obligation is due to be paid |
| school_year | smallInt(4) | The school year the credit sale occurred |
| paid_before_due_date | Currency | The amount paid in advance before the due date |
| paid_1_to_30_days | Currency | The amount paid after the due date, which is grouped from a span of [1 to 30], [31 to 60], etc. days. |
| paid_31_to_60_days | Currency | The amount paid after the due date, which is grouped from a span of [1 to 30], [31 to 60], etc. days. |
| paid_61_to_90_days | Currency | The amount paid after the due date, which is grouped from a span of [1 to 30], [31 to 60], etc. days. |
| paid_91_to_120_days | Currency | The amount paid after the due date, which is grouped from a span of [1 to 30], [31 to 60], etc. days. |
| paid_121_to_150_days | Currency | The amount paid after the due date, which is grouped from a span of [1 to 30], [31 to 60], etc. days. |
| paid_151_to_180_days | Currency | The amount paid after the due date, which is grouped from a span of [1 to 30], [31 to 60], etc. days. |
| paid_180_above_days | Currency | The amount paid after the due date, which is grouped from a span of [1 to 30], [31 to 60], etc. days. |
| remaining_accounts_receivables | Currency | The amount of accounts receivable left that needs to be paid as of generating this fact table |

The fact table decomposes each receivable into payment-aging buckets (paid before the due date, 1–30 days, 31–60 days, and so on), which is the representation from which the four ordinal payment brackets are labeled. The remaining_accounts_receivables field identifies invoices still unpaid at the observation cutoff; these censored records are exactly the cases handled by the survival analysis described in Section 3.4.

*Figure 3.X presents the Level-2 DFD of Process 5.0 (Data Cleaning), detailing how the CreditSalesProcessor pipeline transforms merged raw records into the two distinct data streams used by the modelling component.*

*Figure 3.X: Level-2 DFD of Process 5.0 — Data Cleaning.*
The diagram decomposes Process 5.0 into four sequential sub-processes executed by the CreditSalesProcessor pipeline. Sub-process 5.1 (InvoiceBuilder) allocates every payment amount into one of eight aging brackets — paid before due, 1–30 days, 31–60 days, 61–90 days, 91–120 days, 121–150 days, 151–180 days, and 180-plus days — using sequential conditional bucketing based on elapsed days between the due date and each payment date; this step runs in parallel threads via a ThreadPool to contain processing time over the full 11,440-record dataset. Sub-process 5.2 (FeatureEngineer) derives the behavioral, financial, and temporal features described in Section 3.3, including the four DTP lag features, weighted and simple averages, cumulative balance fields, enrollment plan encodings, and temporal indicators. Sub-process 5.3 (InvoicePostProcessor) applies post-processing filters, including school year filtering, winsorization of outlier financial values, and removal of columns not required for modeling. Sub-process 5.4 performs the data stream split: records where the censor indicator equals one have their survival-specific columns removed and are written to the ML training stream (df_data), while all records with survival columns intact are written to the survival analysis stream (df_data_surv). These two streams are the sole outputs of Process 5.0 and flow to different downstream processes.

## 3.3 Granular Feature Engineering
The feature space comprises 40 variables across six categories, derived from institutional journal entries and engineered to capture granular payment behaviors:
(a) Raw Financial: `gross_receivables`, `amount_discounted`, `adjustments`, `credit_sale_amount` (net receivable)
(b) Days-to-Payment (DTP) Historical: `dtp_1` through `dtp_4` (most recent 4 invoices), `dtp_avg`, `dtp_wavg` (weighted 0.4,0.3,0.2,0.1), `dtp_2_trend`, `dtp_3_trend`, `days_since_last_payment`, `dtp_rolling_std`, `dtp_max`
(c) Financial/Cumulative: `amount_due_cumsum`, `amount_paid_cumsum`, `opening_balance`, `opening_balance_flag`, `payment_ratio` (paid/due ratio)
(d) Behavioral/Historical: `prev_bracket` (most recent payment bracket), `early_payer_flag` (binary), `on_time_streak` (consecutive on-time payments)
(e) Payment Plan: One-hot encoded `plan_type_Plan-A` through `plan_type_Plan-E` and `plan_type_nan`, plus ordinal `plan_type_risk_score` (A=0→NaN=5)
(f) Temporal: `due_month` (1-12), `due_quarter` (1-4)
These features were generated via the `CreditSalesProcessor` pipeline, transforming raw transactional data into analysis-ready inputs. All features were retained for modeling.
## 3.4 Survival-Analysis-Derived Features
*Figure 3.X presents the Level-2 DFD of Process 7.5 (Cox Survival Analysis Tuning), detailing the hyperparameter search and time-point derivation steps that precede survival feature generation.*

*Figure 3.X: Level-2 DFD of Process 7.5 — Cox Survival Analysis Tuning.*
The diagram decomposes Process 7.5 into six sequential sub-processes executed by the CoxHyperparameterTuner. Sub-process 7.5.1 initialises the hyperparameter search grid: six values of the regularisation strength alpha (0.001, 0.01, 0.05, 0.1, 0.5, 1.0) are combined with three values of the elastic-net mixing parameter l1_ratio (0.5, 0.75, 1.0), yielding eighteen candidate configurations. Sub-process 7.5.2 partitions the training portion of the survival analysis stream into k folds for cross-validation. Sub-process 7.5.3 fits a CoxnetSurvivalAnalysis model (scikit-survival) on the training folds of each configuration. Sub-process 7.5.4 scores the fitted model on each held-out validation fold using Harrell's concordance index (C-index), a rank-based measure of how well the model orders samples by risk. Sub-process 7.5.5 selects the configuration with the highest mean C-index across all folds as the best hyperparameter set; the full grid search results are also written to cox_tuning_report.xlsx for audit. Sub-process 7.5.6 derives nine optimal evaluation time points from the observed event distribution using a slope-change algorithm applied to the Kaplan-Meier survival curve; these time points represent periods of the greatest informational density in the observed settlement distribution and are passed alongside the best hyperparameters to Process 8.5.

*Figure 3.X presents the Level-2 DFD of Process 8.5 (Survival Feature Generation), detailing how the tuned Cox model is applied to produce the survival-derived features that augment the original feature set.*

*Figure 3.X: Level-2 DFD of Process 8.5 — Survival Feature Generation.*
The diagram decomposes Process 8.5 into five sequential sub-processes. Sub-process 8.5.1 fits a CoxnetSurvivalAnalysis model using the best hyperparameters from Process 7.5 exclusively on the training partition of the survival analysis stream, preventing data leakage from test-set survival patterns. Sub-process 8.5.2 applies the fitted model to both partitions and computes, at each of the nine optimal time points, the survival probability S(t): the estimated likelihood that an invoice remains unpaid beyond time t. Sub-process 8.5.3 computes the cumulative hazard H(t) at each of the nine time points: the accumulated probability of settlement up to and including time t. Sub-process 8.5.4 computes two scalar summaries per sample: the risk score, derived from the exponentiated log-hazard ratio, and the expected survival time E[T]; a safeguard clips the linear predictor to the range [−10, +10] before exponentiation to prevent numerical overflow. Sub-process 8.5.5 concatenates the original features with the 18 time-varying features (9 × S(t) + 9 × H(t)) and the two scalar features (risk score, E[T]) to produce the enhanced dataset alongside the unmodified baseline dataset; both datasets are forwarded to Process 8.0 (Model Building) to enable a direct paired comparison between baseline and survival-augmented classification performance.

To capture the temporal risk of non-payment, the researchers’ applied survival analysis using a Cox Proportional Hazards (Cox PH) model via the `lifelines` library. The model was fitted on a censored training-set subset where the "event" is defined as full invoice settlement, preventing data leakage when applied to test splits. Five time-dependent hazard features were derived:
(1) `partial_hazard`: Exponentiated linear predictor (exp(coef)) from the Cox model
(2) `log_partial_hazard`: Natural log of the partial hazard
(3) `expected_survival_time`: Mean predicted days until full settlement under the Cox model
(4) `survival_probability`: Likelihood the invoice remains unpaid beyond a reference time point
(5) `cumulative_hazard`: Cumulative hazard at the reference time point
These features constitute the "Enhanced" feature regime, evaluated against the "Baseline" regime (all features except these five) to assess survival analysis' marginal utility in payment prediction.

## 3.5 Model Architectures
*Figure 3.X presents the Level-2 DFD of Process 8.0 (Model Building), detailing the three sub-processes that train the fifteen classifier variants across baseline and enhanced feature sets.*

*Figure 3.X: Level-2 DFD of Process 8.0 — Model Building.*
The diagram decomposes Process 8.0 into three concurrent sub-processes that collectively train fifteen classifier variants. Sub-process 8.1 trains six base classifiers — AdaBoost, Random Forest, XGBoost, Decision Tree, Gaussian Naive Bayes, and K-Nearest Neighbours — on both the baseline and enhanced feature sets. Sub-process 8.2 trains three ordinal classifiers using the Frank and Hall (2001) K-1 binary decomposition: for the four payment outcome classes, three binary classifiers are trained to estimate P(y > 0), P(y > 1), and P(y > 2) respectively; class probabilities are recovered as sequential differences and renormalised after clipping negative values to zero. Only tree-based estimators are used for ordinal classifiers because the feature selection step (Process 9.0) relies on mean decrease in impurity (MDI) importance scores, which require the estimator to expose a feature_importances_ attribute unavailable in K-Nearest Neighbours and Gaussian Naive Bayes. Sub-process 8.3 trains six two-stage ensemble classifiers: each pipeline first trains a binary Stage 1 model to classify an invoice as on-time versus late, then trains a multiclass Stage 2 model on the late-only subset to discriminate the 30-day, 60-day, and 90-day brackets; final class probabilities are computed as P(class k, k > 0) = P(late) × P(class k | late). All fifteen classifiers are trained in parallel using scikit-learn's joblib backend (n_jobs=−1), and their evaluation metrics and feature importance scores are written to the results database for aggregation in Process 12.0.


The researchers’ benchmarked 15 models categorized into three families:
(a) Base Classifiers: Six standard algorithms including AdaBoost, Random Forest, XGBoost, Decision Tree, Gaussian Naive Bayes, and K-Nearest Neighbors (KNN).
(b) Ordinal Classifiers: Three architectures (Ordinal AdaBoost, Ordinal RF, Ordinal XGBoost) utilizing a Frank and Hall decomposition [12] to exploit the ordered nature of the payment brackets.
(c) Two-Stage Ensemble Classifiers: Six hierarchical pipelines designed to mirror institutional decision-making.
- Stage 1 (Binary): A high-capacity learner (e.g., XGBoost) classifies an invoice as either "On-Time" (Class 0) or "Late" (Classes 1-3).
- Stage 2 (Multi-class): If an invoice is predicted "Late," it is passed to a second-stage model (e.g., AdaBoost) trained specifically on the subset of delinquent records to determine the severity (Class 1, 2, or 3).
*Figure 3.3 presents the Level-1 DFD of the modelling component, covering class balancing, model training, and hyperparameter search across the three model families.*

*Figure 3.3: Level-1 DFD of the Modelling component.*
The modelling component operates two concurrent training pipelines. The baseline pipeline trains all fifteen classifier architectures directly on the ML training data stream received from Process 5.0. The survival-augmented pipeline first routes the survival analysis data stream through Process 7.5 (Cox Survival Analysis Tuning), which identifies optimal hyperparameters and nine evaluation time points, and then through Process 8.5 (Survival Feature Generation), which appends 20 survival-derived features per sample to produce an enhanced dataset; the same fifteen architectures are then trained on this augmented feature set. Both pipelines share the same temporal split, balancing strategy, and hyperparameter grid, ensuring that any performance difference between the baseline and enhanced conditions is attributable solely to the survival-derived features. Evaluation metrics and feature importance scores from both pipelines are logged to the results database for aggregation in Process 12.0.
Grid search hyperparameter tuning was employed to identify the optimal model configurations. The specific parameter grids applied to each model type and family are presented in Tables 3.4, 3.5, and 3.6 found on the next pages, corresponding to the base, ordinal, and multi-step classifiers, respectively.
The following table presents the hyperparameter grid searched for the base classifier configurations.
*Table 3.4.  Hyperparameters used (Base Models).*

| Model | Tuned Parameters | Fixed Parameters | Configs |
|---|---|---|---|
| AdaBoost | learning_rate: 		{0.01, 0.1, 0.5, 1.0} n_estimators: 		{50, 150} | algorithm: SAMME.R | 7 |
| Decision Tree | max_depth: 		{10, 20} min_samples_leaf:	{1, 3, 5} | criterion: gini | 6 |
| Gaussian Naive Bayes | var_smoothing: 	{1e-9, 1e-8, 1e-7, 1e-6, 			1e-5, 1e-4, 1e-3, 1e-2} | N/A | 8 |
| KNN | n_neighbors: 		{3, 5, 7} weights: 			{uniform, distance} | metric: minkowski | 6 |
| Random Forest | max_depth: 		{10, 20, 30} min_samples_leaf:	{1, 3, 5} n_estimators: 		{100, 200, 300} | criterion: gini | 21 |
| XGBoost | max_depth:              	{3, 5, 6} learning_rate:              	{0.01, 0.05} n_estimators:              	{300, 500} subsample:               	{0.7, 0.8} colsample_bytree:      	{0.7, 0.8} | min_child_weight: 3, reg_alpha: 0.0, reg_lambda: 1.0 | 11 |
| Total |  |  | 59 |

The ranges reflect three constraints. The computational budget capped each grid at roughly ten configurations per model so that the full 1,092-experiment benchmark remained tractable; the specific values follow prior benchmarking literature, which finds diminishing returns beyond moderate tree depths and ensemble sizes [27]; and the XGBoost grid was additionally shaped by GPU memory constraints, favoring moderate depth with subsampling over exhaustive expansion.
The following table presents the hyperparameter grid searched for the ordinal classifier configurations.
*Table 3.5.  Hyperparameters used (Ordinal Models).*

| Model | Tuned Parameters | Fixed Parameters | Configs |
|---|---|---|---|
| Ordinal AdaBoost | learning_rate: 		{0.01, 0.1, 0.5, 1.0} n_estimators:          	{50, 150} | scale_pos_weight: false | 7 |
| Ordinal Random Forest | max_depth:             	{10, 20, 30}  min_samples_leaf: 	{1, 3, 5} n_estimators:           	{100, 200, 300} | scale_pos_weight: false | 17 |
| Ordinal XGBoost | max_depth:              	{3, 5, 6} learning_rate:           	{0.01, 0.05} n_estimators:            	{300, 500} subsample:               	{0.7, 0.8} colsample_bytree:    	{0.7, 0.8} | min_child_weight: 3, reg_alpha: 0.0, reg_lambda: 1.0, scale_pos_weight: true | 11 |
| Total |  |  | 35 |

The ordinal wrappers reuse the grids of their underlying base learners so that any performance difference is attributable to the Frank–Hall decomposition itself rather than to differential tuning effort. The scale_pos_weight flag is the one ordinal-specific addition, compensating for the shifting class ratios within each binary sub-problem.

The table below shows the hyperparameter grids for the two-stage model (binary first stage and multi-class second stage).
*Table 3.6. Hyperparameters used (Two-Stage Models).*

| Model | Stage 1 Parameters | Stage 1 Parameters | Stage 2 Parameters | Stage 2 Parameters | Configs |
|---|---|---|---|---|---|
| Model | Tuned | Fixed | Tuned | Fixed | Configs |
| Two-Stage XGB → XGB | max_depth: 	{3} lr: 	{0.01, 0.05} n_estimators: {300, 500} | subsample: 0.8 colsample:  0.8 mcw:           3 | max_depth: {3, 5} lr: {0.01, 0.05} n_estimators: {300, 500} | subsample: 0.8 colsample: 0.8 mcw: 3 | 12 |
| Two-Stage XGB → RF | max_depth:	{3} lr: 	{0.01, 0.05} n_estimators: {300, 500} | subsample: 0.8 colsample:  0.8 mcw:           3 | max_depth: {10, 20} min_samples_leaf: {1, 3} n_estimators: {200, 300} | criterion: gini | 12 |
| Two-Stage RF → RF | max_depth: {10, 20}  min_samples_leaf: {1, 3}  n_estimators: {200} | criterion: gini | max_depth: {10, 20} min_samples_leaf: {1, 3} n_estimators: {200, 300} | criterion: gini | 12 |
| Two-Stage XGB → Ada | max_depth: {3, 5} lr: {0.01, 0.05} n_estimators: {300, 500} | subsample:0.8 colsample: 0.8 mcw:          3 | learning_rate: {0.01, 0.1} n_estimators: {50, 150} | algorithm: SAMME.R | 10 |
| Two-Stage RF → Ada | max_depth: {10, 20} min_samples_leaf: {1, 3} n_estimators: {200, 300} | criterion: gini | learning_rate: {0.01, 0.1} n_estimators: {50, 150} | algorithm: SAMME.R | 10 |
| Two-Stage Ada → XGB | learning_rate: {0.01, 0.1, 0.5} n_estimators: {50, 150} | algorithm: SAMME.R | max_depth: {3, 5} lr: 	{0.01, 0.05} n_estimators: {300, 500} | subsample: 0.8 colsample: 0.8 mcw: 3 | 10 |
| Total |  |  |  |  | 66 |


Because each two-stage pipeline trains two models, the per-stage grids were kept deliberately compact to contain combinatorial growth; the tuned values concentrate on the parameters with the largest observed effects (depth, learning rate, ensemble size), while secondary regularization parameters were fixed at the values established in the base-model grids.
These influence scores are then fed into a dedicated feature selection process in Module 9.0, where the most predictive variables are retained. Table 3.7 presents the feature importance method used for each model type.
*Table 3.7.  List of the feature importance methods used per model.*

| Model | Feature Importance Method |
|---|---|
| Decision Tree | Gini importance |
| Random Forest | Mean decrease in impurity (MDI) or Permutation importance |
| AdaBoost | Importance derived from the weighted sum of weak learners (decision stumps/trees). |
| XGBoost | Multiple metrics:  Gain (improvement in accuracy from splits), Cover (number of samples affected), and Frequency (split count). |
| Gaussian Naive Bayes | Feature influence comes from likelihoods (mean, variance per class). |
| KNN | Permutation importance |


The methods differ because feature attribution must respect each model's internal structure: impurity-based measures suit tree learners, the gain/cover/frequency metrics decompose XGBoost's boosted splits, likelihood parameters expose Gaussian Naive Bayes' per-class evidence, and model-agnostic permutation importance covers learners such as KNN that lack intrinsic attribution. Normalizing these scores per experiment allows importance rankings to be compared across model families in Chapter 4.

## 3.6 Experimental Design and Evaluation
The benchmark encompasses 1,092 unique configurations (15 models × 7 balancing strategies × ~10 parameter sets × 2 feature regimes). Models were evaluated using a temporal 70/30 train–test split, where the training set comprised invoices with due dates prior to March 7, 2025, and the test set included invoices due on or after that date. Given the extreme class imbalance (Class 0 dominates at ~76%), macro-averaged F1-score was selected as the primary performance metric, supplemented by Area Under the Receiver Operating Characteristic Curve (ROC-AUC) and confusion matrix analysis to assess per-class recall.
The temporal split was chosen over random cross-validation deliberately: invoices are not exchangeable across time, and a random split would leak future payment behavior into training, inflating measured performance relative to deployment conditions. Training on all invoices due before March 7, 2025 and testing on those due afterward reproduces the situation the school faces in production, predicting forthcoming invoices from historical behavior, and exposes the models to any distribution drift across school years. The class imbalance statistics underline the metric choice: with Class 0 at roughly 76% of records, raw accuracy is dominated by the majority class, while the minority brackets that carry the greatest financial risk contribute least to it. Macro-averaged F1 corrects this by weighting all four classes equally, ROC-AUC complements it with a threshold-independent view of separability, and confusion matrices retain the per-class recall that a finance office ultimately acts on.
*Figure 3.4 illustrates the distribution of the four payment brackets across the 6,527 training records used in this study.*

*Figure 3.4: Distribution of payment statuses.*
The distribution is extremely imbalanced: roughly 76% of invoices fall into Class 0 (on-time), while the three late brackets share the remainder, with the severest brackets the rarest. Left untreated, this imbalance lets a classifier achieve high accuracy by always predicting the majority class while failing entirely on the late invoices that matter operationally. It is this property that necessitates the resampling strategies evaluated in the benchmark (SMOTE [43], Borderline-SMOTE [44], SMOTE-Tomek [45], and threshold-based hybrid undersampling) and that motivates macro-averaged F1, which weights all four classes equally, as the primary evaluation metric, consistent with standard practice in imbalanced classification [46].
The last module of the framework is the analysis phase, presented in Figure 3.5. After the benchmark completes, the analysis component aggregates the logged experiments for evaluation and comparison.


*Figure 3.5: Level-1 DFD of the Analysis component.*
The analysis component reads the per-experiment metrics, feature importance scores, and class mappings from the results database, ranks configurations by macro-F1 and ROC-AUC, and renders the comparative figures presented in Chapter 4. Keeping analysis decoupled from training means new experiments extend the same results store without rerunning prior configurations, and every figure in Chapter 4 is reproducible from the 1,092 logged experiments.
## 3.7 Ethical Considerations
Because the study processes sensitive student financial records and produces predictions about identifiable behavior, two ethical dimensions require explicit treatment: the privacy of the data used to train the models, and the manner in which the resulting predictions may be used.
### 3.7.1 Data Privacy
The dataset contains sensitive student financial records, and privacy was addressed before any analysis took place. All student identifiers were pseudonymized using a deterministic, irreversible hash implemented in a dedicated pseudonymization utility in the project codebase; no real names or student numbers appear in any processed file, and the mapping cannot be reversed from the published artifacts. Raw data files are never committed to version control; their exclusion is enforced through the repository's ignore rules, so the source records exist only within the institution's systems and the researchers' controlled working environment. These measures place the study in compliance with the Data Privacy Act of 2012 (Republic Act 10173) [47], which governs the processing of personal information in the Philippines and requires that processing be limited to declared, legitimate purposes with proportionate safeguards. Data access was granted under written institutional authorization, reproduced as Appendix G, which defines the scope of data access and the restrictions on its use. Finally, the predictive model outputs are intended for institutional receivables management only; they are not designed or released for individual student profiling for any public purpose.
### 3.7.2 Ethical Use of Predictions
There is an inherent risk that machine learning predictions of payment delinquency could be used to discriminate against students from families with poor payment histories, and the researchers acknowledge this risk explicitly. Three mitigations are recommended wherever the system is deployed. First, predictions should inform early outreach and support (earlier reminders, proactive installment restructuring, or referral to social welfare assistance) and never punitive action against the student. Second, any intervention workflow built on the model must comply with Republic Act 11984 [11], which prohibits academic penalties, including examination denial, for outstanding balances; the system's outputs therefore cannot lawfully gate any academic activity. Third, the model should be audited periodically for demographic bias, comparing error rates across student segments, and retrained where disparities emerge. The study's framing is deliberately supportive rather than punitive: the goal is to enable schools to offer targeted payment assistance earlier, not to flag students for enforcement, and this orientation is embedded in the recommendations of Chapter 5.
