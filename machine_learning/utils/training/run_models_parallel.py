import pandas as pd
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from machine_learning.utils.training.load_parameters import ParameterLoader
from machine_learning.utils.data.data_preparation import DataPreparer
from machine_learning.utils.features.generate_survival_features import generate_survival_features
from machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period
from machine_learning.models.ordinal_classifier import OrdinalPipeline
from machine_learning.models.two_stage_classifier import TwoStagePipeline


# When the user selects "ordinal", the runner expands it into these four
# base-estimator variants automatically. Each key maps to the sklearn class
# whose constructor receives the param dict (minus scale_pos_weight).
_ORDINAL_ESTIMATOR_MAP = {
    "ordinal_xgboost": XGBClassifier,
    "ordinal_random_forest": RandomForestClassifier,
    "ordinal_ada_boost": AdaBoostClassifier,
}

_TWO_STAGE_ESTIMATOR_PAIRS = {
    "two_stage_xgb_xgb": (XGBClassifier, XGBClassifier),
    "two_stage_xgb_rf":  (XGBClassifier, RandomForestClassifier),
    "two_stage_rf_rf":   (RandomForestClassifier, RandomForestClassifier),
    "two_stage_xgb_ada": (XGBClassifier, AdaBoostClassifier),
    "two_stage_rf_ada":  (RandomForestClassifier, AdaBoostClassifier),
    "two_stage_ada_xgb": (AdaBoostClassifier, XGBClassifier),
}

# Used for the progress bar in Dash
progress_state = {"completed": 0, "total": 0}


class SurvivalExperimentRunner:
    """
    SurvivalExperimentRunner
    ------------------------
    A class to manage and execute survival model experiments with baseline and enhanced
    survival features. Supports parallel and sequential execution, dataset preparation,
    and feature generation.

    This class encapsulates:
    - Dataset preparation per balance strategy and threshold
    - Survival feature generation (cached per dataset)
    - Model training and evaluation (baseline vs. enhanced)
    - Parallel execution with tqdm progress bars

    Attributes
    ----------
    df_data_surv : pd.DataFrame
        Input survival dataset.
    models : dict
        Dictionary mapping model names to pipeline classes.
    balance_strategies : list
        List of balancing strategies to evaluate (e.g., "undersample", "oversample", "hybrid").
    args : Namespace
        Experiment configuration arguments (parameters_dir, target_feature, test_size, etc.).
    best_penalty : float
        Penalty parameter used in survival feature generation.
    thresholds : list or None
        Thresholds for hybrid balancing strategy.
    n_jobs : int
        Number of parallel jobs for joblib execution.
    output_path : str
        Path to save the results Excel file.
    do_not_parallel_compute : list
        List of model names to run sequentially instead of in parallel.
    feature_selection : bool
        Whether to perform feature selection during model training.
    loader : ParameterLoader
        Loader for model parameters.
    parameters_by_model : dict
        Cached parameters for each model.
    preparer : SurvivalDataPreparer
        Preparer object for dataset splitting and balancing.
    """

    def __init__(
        self,
        df_data,
        df_data_surv,
        models,
        balance_strategies,
        args,
        best_parameters,
        thresholds=None,
        n_jobs=-1,
        do_not_parallel_compute=None,
        feature_selection_baseline=True,
        feature_selection_enhanced=True,
        use_lda: bool = False,
        lda_mode: str = "append",
    ):
        self.df_data = df_data
        self.df_data_surv = df_data_surv
        self.models = models
        self.balance_strategies = balance_strategies
        self.args = args
        self.best_parameters = best_parameters
        self.thresholds = thresholds
        self.n_jobs = n_jobs
        self.do_not_parallel_compute = do_not_parallel_compute or []
        self.feature_selection_baseline = feature_selection_baseline
        self.feature_selection_enhanced = feature_selection_enhanced
        self.use_lda  = use_lda
        self.lda_mode = lda_mode

        # Initialize parameter loader
        self.loader = ParameterLoader(args.parameters_dir)

        # Expand ordinal and two-stage model aliases
        self.models = {}
        for model_name, pipeline_class in models.items():
            if model_name == "ordinal":
                for variant_key in _ORDINAL_ESTIMATOR_MAP:
                    self.models[variant_key] = OrdinalPipeline
            elif model_name == "two_stage":
                for variant_key in _TWO_STAGE_ESTIMATOR_PAIRS:
                    self.models[variant_key] = TwoStagePipeline
            else:
                self.models[model_name] = pipeline_class

        self.parameters_by_model = {m: self.loader.get_parameters(m) for m in self.models}

        self.preparer = DataPreparer(
            df_data,
            target_feature=args.target_feature,
            test_size=args.test_size,
            verbose=False,
        ).encode_labels().train_test_split()

        self.class_mappings = self.preparer.class_mapping

    # -----------------------------
    # Dataset preparation helper
    # -----------------------------

    def prepare_dataset(self, balance_strategy, threshold):
        """
        Prepare dataset for a given balance strategy and threshold.

        Runs data preparation, splits train/test sets, and generates survival features.
        Results are cached and reused across models and parameters.
        """
        self.preparer.resample(balance_strategy=balance_strategy, undersample_threshold=threshold)

        # Snapshot BEFORE LDA — Cox scaler was fitted on original features
        # and will raise if it sees LD1/LD2/LD3 columns.
        X_train_raw = self.preparer.X_train.copy()
        X_test_raw  = self.preparer.X_test.copy()
        y_train, y_test = self.preparer.y_train.copy(), self.preparer.y_test.copy()

        X_surv = self.df_data_surv.drop(columns=["days_elapsed_until_fully_paid", "censor"])
        T = adjust_payment_period(self.df_data_surv["days_elapsed_until_fully_paid"])
        E = self.df_data_surv["censor"]

        # Pass original features to Cox — survival features generated first
        X_surv_train, X_surv_test = generate_survival_features(
            X_surv, T, E, X_train_raw, X_test_raw,
            best_params=self.best_parameters, time_points=self.args.time_points
        )

        # Apply LDA AFTER survival feature generation.
        # Two separate transformers: one for the enhanced (survival) set,
        # one for the baseline set — each fitted only on its own X_train.
        if self.use_lda:
            from machine_learning.utils.features.lda_transformer import LDATransformer

            lda_enhanced = LDATransformer(mode=self.lda_mode)
            X_surv_train = lda_enhanced.fit_transform(X_surv_train, y_train)
            X_surv_test  = lda_enhanced.transform(X_surv_test)

            lda_baseline = LDATransformer(mode=self.lda_mode)
            X_train = lda_baseline.fit_transform(X_train_raw, y_train)
            X_test  = lda_baseline.transform(X_test_raw)
        else:
            X_train = X_train_raw
            X_test  = X_test_raw

        return X_train, X_test, y_train, y_test, X_surv_train, X_surv_test

    # -----------------------------
    # Pipeline construction helper
    # -----------------------------

    def _build_pipelines(
        self,
        model_name,
        pipeline_class,
        param,
        X_train, X_test,
        X_surv_train, X_surv_test,
        y_train, y_test,
    ):
        """
        Construct baseline and enhanced pipeline instances for a single experiment.
        """

        if model_name in _ORDINAL_ESTIMATOR_MAP:
            estimator_cls = _ORDINAL_ESTIMATOR_MAP[model_name]
            scale_pos_weight = param.get("scale_pos_weight", True)
            estimator_params = {k: v for k, v in param.items() if k != "scale_pos_weight"}

            pipeline_baseline = OrdinalPipeline(
                X_train, X_test, y_train, y_test, self.args,
                parameters={"scale_pos_weight": scale_pos_weight},
                base_estimator=estimator_cls(**estimator_params),
            )

            pipeline_enhanced = OrdinalPipeline(
                X_surv_train, X_surv_test, y_train, y_test, self.args,
                parameters={"scale_pos_weight": scale_pos_weight},
                base_estimator=estimator_cls(**estimator_params),
            )

        elif model_name in _TWO_STAGE_ESTIMATOR_PAIRS:
            if "stage1" not in param or "stage2" not in param:
                raise ValueError(f"{model_name} parameters must define 'stage1' and 'stage2'")

            s1_cls, s2_cls = _TWO_STAGE_ESTIMATOR_PAIRS[model_name]
            stage1_params = param["stage1"]
            stage2_params = param["stage2"]

            pipeline_baseline = TwoStagePipeline(
                X_train, X_test, y_train, y_test, self.args,
                stage1_estimator=s1_cls(**stage1_params),
                stage2_estimator=s2_cls(**stage2_params),
            )

            pipeline_enhanced = TwoStagePipeline(
                X_surv_train, X_surv_test, y_train, y_test, self.args,
                stage1_estimator=s1_cls(**stage1_params),
                stage2_estimator=s2_cls(**stage2_params),
            )

        else:
            pipeline_baseline = pipeline_class(X_train, X_test, y_train, y_test, self.args, param)
            pipeline_enhanced = pipeline_class(X_surv_train, X_surv_test, y_train, y_test, self.args, param)

        return pipeline_baseline, pipeline_enhanced

    # -----------------------------
    # Single experiment runner
    # -----------------------------

    def run_model_experiment(self, model_name, pipeline_class, param, dataset, balance_strategy, threshold):
        """
        Run a single experiment for a given model, parameter set, and dataset.

        Executes both baseline and enhanced pipelines, performs training and
        evaluation, then returns a flat result dict suitable for direct
        concatenation into a pandas DataFrame.

        Parameters
        ----------
        model_name : str
            Name of the model being tested.
        pipeline_class : class
            Pipeline class implementing the model workflow.
        param : dict
            Parameter set for the model.
        dataset : tuple
            Prepared dataset of the form:
            (X_train, X_test, y_train, y_test, X_survival_train, X_survival_test).
        balance_strategy : str
            Balancing strategy used to prepare the dataset.
        threshold : float or None
            Threshold for hybrid balancing strategy. None if not applicable.

        Returns
        -------
        dict
            Flat dictionary with the following keys:
            - model, parameters, balance_strategy, undersample_threshold
            - baseline_accuracy, baseline_precision_macro, baseline_recall_macro,
            baseline_f1_macro, baseline_roc_auc_macro, baseline_confusion_matrix,
            baseline_roc_curve, baseline_pr_curve
            - baseline_feature_method, baseline_feature_selected, baseline_feature_weights
            - enhanced_accuracy, enhanced_precision_macro, enhanced_recall_macro,
            enhanced_f1_macro, enhanced_roc_auc_macro, enhanced_confusion_matrix,
            enhanced_roc_curve, enhanced_pr_curve
            - enhanced_feature_method, enhanced_feature_selected, enhanced_feature_weights
        """
        (X_train, X_test, y_train, y_test,
        X_survival_train, X_survival_test) = dataset

        pipeline_baseline, pipeline_enhanced = self._build_pipelines(
            model_name, pipeline_class, param,
            X_train, X_test, X_survival_train, X_survival_test,
            y_train, y_test,
        )

        pipeline_baseline.initialize_model().fit(use_feature_selection=self.feature_selection_baseline)
        result_baseline = pipeline_baseline.evaluate().show_results()

        # Enhanced pipeline
        pipeline_enhanced.initialize_model().fit(use_feature_selection=self.feature_selection_enhanced)
        result_enhanced = pipeline_enhanced.evaluate().show_results()

        if model_name in _TWO_STAGE_ESTIMATOR_PAIRS:
            param_str = f"stage1={param['stage1']}, stage2={param['stage2']}"
        else:
            param_str = str(sorted({k: v for k, v in param.items() if k != "scale_pos_weight"}.items()))

        results = {
            "model": model_name,
            "parameters": param_str,
            "balance_strategy": balance_strategy,
            **{f"baseline_{k}": v for k, v in result_baseline.items()},
            "baseline_feature_method": pipeline_baseline.features.method_text,
            "baseline_feature_parameters": pipeline_baseline.features.method_parameters,
            "baseline_feature_selected": pipeline_baseline.features.selected,
            "baseline_feature_weights": pipeline_baseline.features.weights,
            **{f"enhanced_{k}": v for k, v in result_enhanced.items()},
            "enhanced_feature_method": pipeline_enhanced.features.method_text,
            "enhanced_feature_parameters": pipeline_enhanced.features.method_parameters,
            "enhanced_feature_selected": pipeline_enhanced.features.selected,
            "enhanced_feature_weights": pipeline_enhanced.features.weights,
        }

        return results

    # -----------------------------
    # Main experiment runner
    # -----------------------------

    def run(self):
        """
        Run all survival model experiments.

        Iterates over balance strategies and thresholds, prepares datasets,
        builds tasks for all models and parameters, executes them in parallel
        (or sequentially if specified), and aggregates all results into a
        flat pandas DataFrame suitable for immediate analysis or export.

        Returns
        -------
        tuple
            A 2-tuple of (results_df, class_mappings):

            results_df : pd.DataFrame
                Each row is one experiment run. Columns are:
                - model, parameters, balance_strategy, undersample_threshold
                - baseline_accuracy, baseline_precision_macro, baseline_recall_macro,
                baseline_f1_macro, baseline_roc_auc_macro, baseline_confusion_matrix,
                baseline_roc_curve, baseline_pr_curve
                - baseline_feature_method, baseline_feature_selected, baseline_feature_weights
                - enhanced_accuracy, enhanced_precision_macro, enhanced_recall_macro,
                enhanced_f1_macro, enhanced_roc_auc_macro, enhanced_confusion_matrix,
                enhanced_roc_curve, enhanced_pr_curve
                - enhanced_feature_method, enhanced_feature_selected, enhanced_feature_weights

            class_mappings : dict
                Dictionary mapping original class labels to their encoded
                integer representations, as produced by the label encoder
                during dataset preparation.
        """
        parallel_tasks = []
        sequential_tasks = []

        for balance_strategy in self.balance_strategies:
            # Guard against thresholds=None when hybrid is selected
            strategy_thresholds = (self.thresholds or [None]) if balance_strategy == "hybrid" else [None]

            for threshold in strategy_thresholds:
                dataset = self.prepare_dataset(balance_strategy, threshold)

                for model_name, pipeline_class in self.models.items():
                    for param in self.parameters_by_model[model_name]:
                        task_args = (model_name, pipeline_class, param, dataset, balance_strategy, threshold)

                        if model_name in self.do_not_parallel_compute:
                            sequential_tasks.append(task_args)
                        else:
                            parallel_tasks.append(task_args)

        progress_state["total"] = len(parallel_tasks) + len(sequential_tasks)
        progress_state["completed"] = 0

        all_results = []

        if parallel_tasks:
            import threading
            lock = threading.Lock()

            def _run_and_track(task_args):
                result = self.run_model_experiment(*task_args)
                with lock:
                    progress_state["completed"] += 1
                return result

            n_workers = cpu_count() if self.n_jobs == -1 else self.n_jobs
            with ThreadPool(processes=n_workers) as pool:
                all_results.extend(pool.map(_run_and_track, parallel_tasks))

        for task_args in sequential_tasks:
            all_results.append(self.run_model_experiment(*task_args))
            progress_state["completed"] += 1

        return pd.DataFrame(all_results), self.class_mappings

# ══════════════════════════════════════════════════════════════════════════════
#  FINALIZATION RUNNER
#  Single-model trainer for Step 5 deployment.
#
#  Reuses _ORDINAL_ESTIMATOR_MAP, _TWO_STAGE_ESTIMATOR_PAIRS, and the
#  pipeline-construction logic already established in SurvivalExperimentRunner,
#  eliminating the duplication that previously existed in step_5.py.
#
#  Key differences from SurvivalExperimentRunner:
#    - Uses prep_full_data (no train/test split): the entire dataset is used
#      for training because this pipeline is destined for deployment, not
#      evaluation.
#    - Trains exactly ONE pipeline (enhanced only): baseline is not needed for
#      the deployed model.
#    - Accepts a pre-fitted Cox model (fitted_cph) so Cox regression is not
#      re-run from scratch; step_5 can pass the model it already fitted.
#    - Returns (pipeline, label_encoder) ready to be pickled for deployment.
# ══════════════════════════════════════════════════════════════════════════════

class FinalizationRunner:
    """
    Trains a single selected model on the full dataset for deployment.

    Mirrors the pipeline-construction logic of
    ``SurvivalExperimentRunner._build_pipelines`` but adapts it for the
    Step 5 finalization context:

    - **Full-dataset training** — ``DataPreparer.prep_full_data`` is used
      instead of ``train_test_split``.  The deployed model sees every
      available training sample.
    - **Enhanced pipeline only** — survival features are generated and
      injected; no separate baseline pipeline is built.
    - **Accepts a pre-fitted Cox model** — if ``fitted_cph`` is supplied,
      ``generate_survival_features`` reuses it directly.  Pass ``None`` to
      let the function refit from ``best_surv_params``.

    Parameters
    ----------
    df_data : pd.DataFrame
        Cleaned credit-sales dataframe with survival columns dropped
        (i.e. ``censor == 1`` rows only, without ``days_elapsed_*`` /
        ``censor`` columns).
    df_data_surv : pd.DataFrame
        Full credit-sales dataframe with survival columns present but
        non-survival auxiliary columns (``due_date``, ``dtp_bracket``)
        dropped.
    model_key : str
        Fully-qualified model name as stored in results, e.g.
        ``"two_stage_xgb_ada"`` or ``"ordinal_xgboost"``.
    balance_strategy : str
        Resampling strategy to apply (e.g. ``"smote"``, ``"none"``).
    args : object
        Config object with attributes:
        ``parameters_dir``, ``target_feature``, ``time_points``,
        ``parameters`` (the hyperparameter dict for this model).
    best_surv_params : dict
        Best Cox hyperparameters ``{"alpha": float, "l1_ratio": float}``
        as saved by step 3.
    fitted_cph : sksurv estimator or None, default None
        A pre-fitted ``CoxnetSurvivalAnalysis`` instance.  When supplied,
        ``generate_survival_features`` skips refitting and uses this
        directly — saving significant time in step 5.

    Examples
    --------
    >>> runner = FinalizationRunner(
    ...     df_data, df_data_surv,
    ...     model_key="two_stage_xgb_ada",
    ...     balance_strategy="smote",
    ...     args=args,
    ...     best_surv_params={"alpha": 0.01, "l1_ratio": 1.0},
    ...     fitted_cph=cox_model,
    ... )
    >>> pipeline, label_encoder = runner.train()
    """

    def __init__(
        self,
        df_data,
        df_data_surv,
        model_key: str,
        balance_strategy: str,
        args,
        best_surv_params: dict,
        fitted_cph=None,
        use_lda: bool = False,
        lda_mode: str = "append",
    ):
        self.df_data          = df_data
        self.df_data_surv     = df_data_surv
        self.model_key        = model_key
        self.balance_strategy = balance_strategy
        self.args             = args
        self.best_surv_params = best_surv_params
        self.fitted_cph       = fitted_cph
        self.use_lda          = use_lda
        self.lda_mode         = lda_mode

        # Validate model_key early so the caller gets a clear error before
        # any expensive data preparation runs.
        self._plain_map = {
            "ada_boost":            None,  # resolved at train() time via ML imports
            "decision_tree":        None,
            "gaussian_naive_bayes": None,
            "knn":                  None,
            "random_forest":        None,
            "xgboost":              None,
            "stacked_ensemble":     None,
            "nn_mlp":               None,
            "nn_transformer":       None,
        }
        if (model_key not in self._plain_map
                and model_key not in _ORDINAL_ESTIMATOR_MAP
                and model_key not in _TWO_STAGE_ESTIMATOR_PAIRS):
            raise ValueError(
                f"Unknown model_key {model_key!r}.\n"
                f"  Plain models:     {sorted(self._plain_map)}\n"
                f"  Ordinal variants: {sorted(_ORDINAL_ESTIMATOR_MAP)}\n"
                f"  Two-stage combos: {sorted(_TWO_STAGE_ESTIMATOR_PAIRS)}"
            )

    def train(self):
        """
        Prepare data, build the enhanced pipeline, fit it, and return it.

        Returns
        -------
        pipeline : fitted pipeline instance
            Ready for ``pickle.dump`` to the deployment directory.
        label_encoder : LabelEncoder
            The label encoder fitted by ``DataPreparer.prep_full_data``,
            needed to decode predictions at inference time.
        """
        from machine_learning import (
            AdaBoostPipeline, DecisionTreePipeline, GaussianNaiveBayesPipeline,
            KNearestNeighborPipeline, RandomForestPipeline, XGBoostPipeline,
            StackedEnsemblePipeline, MultiLayerPerceptronPipeline, TransformerPipeline,
            OrdinalPipeline, TwoStagePipeline,
        )
        from machine_learning.utils.features.generate_survival_features import (
            generate_survival_features,
        )
        from machine_learning.utils.data.data_preparation import DataPreparer

        PLAIN_PIPELINE_MAP = {
            "ada_boost":            AdaBoostPipeline,
            "decision_tree":        DecisionTreePipeline,
            "gaussian_naive_bayes": GaussianNaiveBayesPipeline,
            "knn":                  KNearestNeighborPipeline,
            "random_forest":        RandomForestPipeline,
            "xgboost":              XGBoostPipeline,
            "stacked_ensemble":     StackedEnsemblePipeline,
            "nn_mlp":               MultiLayerPerceptronPipeline,
            "nn_transformer":       TransformerPipeline,
        }

        # ── 1. Full-dataset data preparation ─────────────────────────────────
        # prep_full_data applies resampling to the entire dataset with no
        # train/test split.  This maximises the training signal for the
        # deployed model.
        preparer = DataPreparer(
            df_data=self.df_data,
            target_feature=self.args.target_feature,
            verbose=True,
        )

        if self.use_lda:
            preparer.prep_full_data_with_lda(
                balance_strategy=self.balance_strategy,
                lda_mode=self.lda_mode,
                lda_verbose=True,
            )
        else:
            preparer.prep_full_data(balance_strategy=self.balance_strategy)

        X_full = preparer.X_train
        y_full = preparer.y_train

        # ── 2. Survival feature generation ───────────────────────────────────
        X_surv = self.df_data_surv.drop(
            columns=["days_elapsed_until_fully_paid", "censor"]
        )
        T = adjust_payment_period(
            self.df_data_surv["days_elapsed_until_fully_paid"]
        )
        E = self.df_data_surv["censor"]

        X_enhanced = generate_survival_features(
            X_surv=X_surv,
            T=T,
            E=E,
            X_train=X_full,
            X_test=None,          # no test split — full dataset only
            best_params=self.best_surv_params,
            time_points=self.args.time_points,
            fitted_cph=self.fitted_cph,   # reuse pre-fitted Cox when available
        )

        # ── 3. Pipeline construction ─────────────────────────────────────────
        # Mirrors SurvivalExperimentRunner._build_pipelines exactly,
        # adapted for the single-pipeline / full-dataset context.
        model_params = getattr(self.args, "parameters", {}) or {}

        if self.model_key in _TWO_STAGE_ESTIMATOR_PAIRS:
            if "stage1" not in model_params or "stage2" not in model_params:
                raise ValueError(
                    f"Parameters for {self.model_key!r} must define 'stage1' "
                    f"and 'stage2' dicts. Got keys: {list(model_params)}"
                )
            s1_cls, s2_cls = _TWO_STAGE_ESTIMATOR_PAIRS[self.model_key]
            pipeline = TwoStagePipeline(
                X_enhanced, X_enhanced,
                y_full, y_full,
                self.args,
                stage1_estimator=s1_cls(**model_params["stage1"]),
                stage2_estimator=s2_cls(**model_params["stage2"]),
            )

        elif self.model_key in _ORDINAL_ESTIMATOR_MAP:
            estimator_cls    = _ORDINAL_ESTIMATOR_MAP[self.model_key]
            scale_pos_weight = model_params.get("scale_pos_weight", True)
            estimator_params = {
                k: v for k, v in model_params.items() if k != "scale_pos_weight"
            }
            pipeline = OrdinalPipeline(
                X_enhanced, X_enhanced,
                y_full, y_full,
                self.args,
                parameters={"scale_pos_weight": scale_pos_weight},
                base_estimator=estimator_cls(**estimator_params),
            )

        else:
            pipeline_cls = PLAIN_PIPELINE_MAP[self.model_key]
            pipeline = pipeline_cls(
                X_enhanced, X_enhanced,
                y_full, y_full,
                self.args,
                model_params,
            )

        # ── 4. Fit ────────────────────────────────────────────────────────────
        pipeline.initialize_model()
        pipeline.fit(use_feature_selection=True)

        return pipeline, preparer.label_encoder