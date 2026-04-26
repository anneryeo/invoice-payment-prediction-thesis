import os
import pickle
import pandas as pd
import time
import joblib
import warnings
from multiprocessing import Pool, cpu_count
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# Suppress sklearn joblib warnings in all processes
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")


def _worker_init():
    """Pool initializer: suppresses noisy warnings inside each worker process."""
    import warnings as _w
    _w.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
    _w.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")


def _make_task_key(model_name, param, balance_strategy, threshold, fs_baseline, fs_enhanced):
    """Deterministic string key uniquely identifying one experiment task."""
    if model_name in _TWO_STAGE_ESTIMATOR_PAIRS:
        param_str = f"stage1={param['stage1']}, stage2={param['stage2']}"
    else:
        param_str = str(sorted({k: v for k, v in param.items() if k != "scale_pos_weight"}.items()))
    return f"{model_name}|{balance_strategy}|{threshold}|{param_str}|{int(fs_baseline)}|{int(fs_enhanced)}"


def _load_checkpoint(path):
    """Load checkpoint from disk. Returns (completed_keys: set, results: list)."""
    if path is None or not os.path.exists(path):
        return set(), []
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        completed_keys = set(data.get("completed_keys", []))
        results = data.get("results", [])
        print(f"[checkpoint] Loaded {len(results)} prior results from: {path}", flush=True)
        return completed_keys, results
    except Exception as e:
        print(f"[checkpoint] Could not load checkpoint ({e}); starting fresh.", flush=True)
        return set(), []


def _save_checkpoint(path, completed_keys, results):
    """Atomically save checkpoint to disk using write-then-rename."""
    if path is None:
        return
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump({"completed_keys": list(completed_keys), "results": results}, f)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[checkpoint] Warning: could not save checkpoint ({e})", flush=True)


def _make_feature_cache_key(balance_strategy: str, threshold=None, test_size: float = 0.2) -> str:
    """Deterministic key for caching survival features by strategy and threshold."""
    threshold_str = f"@{threshold}" if threshold is not None else ""
    return f"features_{balance_strategy}{threshold_str}_test{int(test_size*100)}.pkl"


def _load_cached_features(cache_dir: str, balance_strategy: str, threshold=None, test_size: float = 0.2) -> tuple:
    """Load cached survival features. Returns (X_surv_train, X_surv_test) or (None, None) if not found."""
    if cache_dir is None or not os.path.exists(cache_dir):
        return None, None
    
    cache_key = _make_feature_cache_key(balance_strategy, threshold, test_size)
    cache_path = os.path.join(cache_dir, cache_key)
    
    if not os.path.exists(cache_path):
        return None, None
    
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"[feature-cache] Hit: {balance_strategy}{('@'+str(threshold)) if threshold else ''}", flush=True)
        return data.get("X_surv_train"), data.get("X_surv_test")
    except Exception as e:
        print(f"[feature-cache] Could not load {cache_key} ({e}); regenerating", flush=True)
        return None, None


def _save_cached_features(cache_dir: str, balance_strategy: str, threshold, 
                         X_surv_train: pd.DataFrame, X_surv_test: pd.DataFrame, 
                         test_size: float = 0.2) -> None:
    """Save survival features to disk cache."""
    if cache_dir is None:
        return
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _make_feature_cache_key(balance_strategy, threshold, test_size)
        cache_path = os.path.join(cache_dir, cache_key)
        tmp_path = cache_path + ".tmp"
        
        with open(tmp_path, "wb") as f:
            pickle.dump({"X_surv_train": X_surv_train, "X_surv_test": X_surv_test}, f)
        os.replace(tmp_path, cache_path)
        print(f"[feature-cache] Saved: {cache_key}", flush=True)
    except Exception as e:
        print(f"[feature-cache] Warning: could not save features ({e})", flush=True)


from src.modules.machine_learning.utils.training.load_parameters import ParameterLoader
from src.modules.machine_learning.utils.data.data_preparation import DataPreparer
from src.modules.machine_learning.utils.features.generate_survival_features import generate_survival_features
from src.modules.machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period
from src.modules.machine_learning.models.ordinal_classifier import OrdinalPipeline
from src.modules.machine_learning.models.two_stage_classifier import TwoStagePipeline


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
progress_state = {"completed": 0, "total": 0, "start_time": None}


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xm Ys'."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _build_pipelines_fn(
    model_name, pipeline_class, param,
    X_train, X_test, X_surv_train, X_surv_test,
    y_train, y_test,
    args, use_lda, lda_mode,
):
    """Module-level pipeline builder — picklable, shared by worker processes and instance methods."""
    if model_name in _ORDINAL_ESTIMATOR_MAP:
        estimator_cls = _ORDINAL_ESTIMATOR_MAP[model_name]
        scale_pos_weight = param.get("scale_pos_weight", True)
        estimator_params = {k: v for k, v in param.items() if k != "scale_pos_weight"}

        pipeline_baseline = OrdinalPipeline(
            X_train, X_test, y_train, y_test, args,
            parameters={"scale_pos_weight": scale_pos_weight},
            base_estimator=estimator_cls(**estimator_params),
        )
        pipeline_enhanced = OrdinalPipeline(
            X_surv_train, X_surv_test, y_train, y_test, args,
            parameters={"scale_pos_weight": scale_pos_weight},
            base_estimator=estimator_cls(**estimator_params),
        )

    elif model_name in _TWO_STAGE_ESTIMATOR_PAIRS:
        if "stage1" not in param or "stage2" not in param:
            raise ValueError(f"{model_name} parameters must define 'stage1' and 'stage2'")

        s1_cls, s2_cls = _TWO_STAGE_ESTIMATOR_PAIRS[model_name]
        pipeline_baseline = TwoStagePipeline(
            X_train, X_test, y_train, y_test, args,
            stage1_estimator=s1_cls(**param["stage1"]),
            stage2_estimator=s2_cls(**param["stage2"]),
            use_lda=use_lda,
            lda_mode=lda_mode,
        )
        pipeline_enhanced = TwoStagePipeline(
            X_surv_train, X_surv_test, y_train, y_test, args,
            stage1_estimator=s1_cls(**param["stage1"]),
            stage2_estimator=s2_cls(**param["stage2"]),
            use_lda=use_lda,
            lda_mode=lda_mode,
        )

    else:
        pipeline_baseline = pipeline_class(X_train, X_test, y_train, y_test, args, param)
        pipeline_enhanced = pipeline_class(X_surv_train, X_surv_test, y_train, y_test, args, param)

    return pipeline_baseline, pipeline_enhanced


def _run_model_experiment_fn(
    model_name, pipeline_class, param, dataset,
    balance_strategy, threshold,
    args, use_lda, lda_mode,
    fs_baseline, fs_enhanced,
):
    """Module-level experiment runner — picklable, called by both Pool workers and instance methods."""
    (X_train, X_test, y_train, y_test,
     X_survival_train, X_survival_test) = dataset

    pipeline_baseline, pipeline_enhanced = _build_pipelines_fn(
        model_name, pipeline_class, param,
        X_train, X_test, X_survival_train, X_survival_test,
        y_train, y_test,
        args, use_lda, lda_mode,
    )

    # Use threading backend so nested joblib calls (e.g. permutation_importance
    # inside KNN) don't try to spawn Loky subprocesses from within a Pool worker.
    with joblib.parallel_backend('threading'):
        pipeline_baseline.initialize_model().fit(use_feature_selection=fs_baseline)
        result_baseline = pipeline_baseline.evaluate().show_results()

        pipeline_enhanced.initialize_model().fit(use_feature_selection=fs_enhanced)
        result_enhanced = pipeline_enhanced.evaluate().show_results()

    if model_name in _TWO_STAGE_ESTIMATOR_PAIRS:
        param_str = f"stage1={param['stage1']}, stage2={param['stage2']}"
    else:
        param_str = str(sorted({k: v for k, v in param.items() if k != "scale_pos_weight"}.items()))

    return {
        "model": model_name,
        "undersample_threshold": threshold,
        "parameters": param_str,
        "balance_strategy": balance_strategy,
        **{f"baseline_{k}": v for k, v in result_baseline.items()},
        "baseline_feature_method":      pipeline_baseline.features.method_text,
        "baseline_feature_parameters":  pipeline_baseline.features.method_parameters,
        "baseline_feature_selected":    pipeline_baseline.features.selected,
        "baseline_feature_weights":     pipeline_baseline.features.weights,
        **{f"enhanced_{k}": v for k, v in result_enhanced.items()},
        "enhanced_feature_method":      pipeline_enhanced.features.method_text,
        "enhanced_feature_parameters":  pipeline_enhanced.features.method_parameters,
        "enhanced_feature_selected":    pipeline_enhanced.features.selected,
        "enhanced_feature_weights":     pipeline_enhanced.features.weights,
    }


def _run_task(task_tuple):
    """Module-level worker for multiprocessing.Pool — fully self-contained, no instance pickling."""
    result = _run_model_experiment_fn(*task_tuple)
    key = _make_task_key(task_tuple[0], task_tuple[2], task_tuple[4], task_tuple[5], task_tuple[9], task_tuple[10])
    return key, result


from src.modules.machine_learning.utils.training.cache_manager import CacheManager

class SurvivalExperimentRunner:
    """
    SurvivalExperimentRunner
    ------------------------
    A class to manage and execute survival model experiments with baseline and enhanced
    survival features. Supports parallel and sequential execution, dataset preparation,
    and feature generation.
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
        n_jobs=4,
        do_not_parallel_compute=None,
        feature_selection_baseline=True,
        feature_selection_enhanced=True,
        use_lda: bool = False,
        lda_mode: str = "append",
        checkpoint_path=None,
        cache_dir="data/cache",
    ):
        self.df_data = df_data
        self.df_data_surv = df_data_surv
        self.models = models
        self.balance_strategies = balance_strategies
        self.args = args
        self.best_parameters = best_parameters
        self.thresholds = thresholds
        self.n_jobs = n_jobs
        self.do_not_parallel_compute = list(do_not_parallel_compute or [])
        self.feature_selection_baseline = feature_selection_baseline
        self.feature_selection_enhanced = feature_selection_enhanced
        self.use_lda  = use_lda
        self.lda_mode = lda_mode
        self.checkpoint_path = checkpoint_path
        self.cache = CacheManager(cache_root=cache_dir)

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

        # Auto-route any expanded model that uses XGBClassifier to sequential
        # execution to avoid CUDA context conflicts in the thread pool.
        # This covers: xgboost, ordinal_xgboost, two_stage_xgb_*, two_stage_ada_xgb.
        for expanded_name in self.models:
            if "xgb" in expanded_name and expanded_name not in self.do_not_parallel_compute:
                self.do_not_parallel_compute.append(expanded_name)

        self.parameters_by_model = {m: self.loader.get_parameters(m) for m in self.models}

        self.preparer = DataPreparer(
            df_data,
            target_feature=args.target_feature,
            test_size=args.test_size,
            verbose=False,
        ).encode_labels().train_test_split()

        self.class_mappings = self.preparer.class_mapping

        # Bug fix: snapshot the original train/test split so that prepare_dataset()
        # can reset before each balance strategy instead of resampling cumulatively.
        self._X_train_orig = self.preparer.X_train.copy()
        self._X_test_orig  = self.preparer.X_test.copy()
        self._y_train_orig = self.preparer.y_train.copy()
        self._y_test_orig  = self.preparer.y_test.copy()
        # Train-partition indices (before resampling resets them) — used to
        # restrict Cox fitting to train rows only, preventing test-set leakage.
        self._train_indices = self.preparer.X_train.index

        # Cache for fitted Cox model (fitted once, reused for all balance strategies)
        self._fitted_cox_model = None
        self._cox_scaler = None
        
        # Feature cache directory (disk cache for survival features to speed up reruns)
        self.feature_cache_dir = os.path.join(cache_dir, "survival_features") if cache_dir else None

    # ─────────────────────────────────────────────────────────────────────────────
    # Cox PH model fitting (cached, reused across all balance strategies)
    # ─────────────────────────────────────────────────────────────────────────────

    def _fit_cox_model_once(self):
        """
        Fit Cox PH model ONCE on original train data + all censored rows.
        Cached and reused for all balance strategies to avoid refitting.

        Returns: (fitted_cox_model, scaler) tuple
        """
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        import warnings
        from src.modules.machine_learning.utils.data.clean_survival_inputs import clean_survival_inputs

        print("[cox] Fitting Cox PH model on original train data ...", flush=True)
        cox_start = time.time()

        # Restrict Cox fitting to train-partition rows (plus all censored rows)
        _train_mask = (
            self.df_data_surv.index.isin(self._train_indices)
            | (self.df_data_surv["censor"] == 0)
        )
        _df_surv_train = self.df_data_surv[_train_mask]
        X_surv = _df_surv_train.drop(columns=["days_elapsed_until_fully_paid", "censor"])
        T = adjust_payment_period(_df_surv_train["days_elapsed_until_fully_paid"])
        E = _df_surv_train["censor"]

        # Clean and prepare survival inputs
        from src.modules.machine_learning.utils.data.clean_survival_inputs import clean_survival_inputs
        _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)
        feature_cols = [c for c in df_fit.columns if c not in ("T", "E")]
        X_fit_raw = df_fit[feature_cols]

        # Import _safe_scale from generate_survival_features
        from src.modules.machine_learning.utils.features.generate_survival_features import _safe_scale

        X_fit_scaled, fit_scaler = _safe_scale(X_fit_raw)

        y_fit = Surv.from_arrays(
            event=df_fit["E"].astype(bool).values,
            time=df_fit["T"].astype(float).values,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cox = CoxnetSurvivalAnalysis(
                l1_ratio=self.best_parameters["l1_ratio"],
                alphas=[self.best_parameters["alpha"]],
                fit_baseline_model=True,
                max_iter=100_000,
                tol=1e-7,
            )
            cox.fit(X_fit_scaled, y_fit)

        print(f"[cox] Done ({_format_duration(time.time() - cox_start)})", flush=True)
        return cox, fit_scaler

    # ─────────────────────────────────────────────────────────────────────────────
    # Dataset preparation helper
    # ─────────────────────────────────────────────────────────────────────────────

    def prepare_dataset(self, balance_strategy, threshold):
        """
        Prepare dataset for a given balance strategy and threshold.

        Runs data preparation, splits train/test sets, and generates survival features.
        Results are cached and reused across models and parameters.
        Cox PH model is cached to avoid refitting (saves ~2.5 min per strategy).
        """
        label = f"{balance_strategy}" + (f"@{threshold}" if threshold is not None else "")
        print(f"[dataset] Preparing: {label} ...", flush=True)
        _ds_start = time.time()

        # Bug fix: reset to the original split before each balance strategy so
        # strategies don't stack on top of each other's resampled output.
        self.preparer.X_train = self._X_train_orig.copy()
        self.preparer.y_train = self._y_train_orig.copy()

        self.preparer.resample(balance_strategy=balance_strategy, undersample_threshold=threshold)

        # Snapshot BEFORE LDA — Cox scaler was fitted on original features
        # and will raise if it sees LD1/LD2/LD3 columns.
        X_train_raw = self.preparer.X_train.copy()
        X_test_raw  = self.preparer.X_test.copy()
        y_train, y_test = self.preparer.y_train.copy(), self.preparer.y_test.copy()

        # Ensure Cox model is fitted ONCE before the first prepare_dataset call
        if self._fitted_cox_model is None:
            self._fitted_cox_model, self._cox_scaler = self._fit_cox_model_once()

        # Try to load from cache first (speeds up reruns)
        X_surv_train, X_surv_test = _load_cached_features(
            self.feature_cache_dir, 
            balance_strategy, 
            threshold,
            test_size=self.args.test_size
        )
        
        if X_surv_train is not None and X_surv_test is not None:
            # Cache hit — skip feature generation
            print(f"[dataset]   Using cached survival features ({_format_duration(time.time() - _ds_start)})", flush=True)
        else:
            # Cache miss — generate and save features
            # Bug fix: restrict Cox fitting to train-partition rows (plus all censored
            # rows, which don't belong to either classifier partition). Fitting Cox on
            # the full dataset — including test-partition rows' survival outcomes — is
            # data leakage that inflates test-set survival feature quality.
            _train_mask = (
                self.df_data_surv.index.isin(self._train_indices)
                | (self.df_data_surv["censor"] == 0)
            )
            _df_surv_train = self.df_data_surv[_train_mask]
            X_surv = _df_surv_train.drop(columns=["days_elapsed_until_fully_paid", "censor"])
            T = adjust_payment_period(_df_surv_train["days_elapsed_until_fully_paid"])
            E = _df_surv_train["censor"]

            # Pass cached Cox model to generate_survival_features to skip refitting
            print(f"[dataset]   Generating survival features (using cached Cox) ...", flush=True)
            X_surv_train, X_surv_test = generate_survival_features(
                X_surv, T, E, X_train_raw, X_test_raw,
                best_params=None,  # Not needed when using fitted_cox + cox_scaler
                time_points=self.args.time_points,
                fitted_cph=self._fitted_cox_model,  # Use cached model
                cox_scaler=self._cox_scaler,  # Use cached scaler
            )
            
            # Save to cache for future runs
            _save_cached_features(
                self.feature_cache_dir,
                balance_strategy,
                threshold,
                X_surv_train,
                X_surv_test,
                test_size=self.args.test_size
            )
            
        print(f"[dataset]   Done ({_format_duration(time.time() - _ds_start)})", flush=True)

        # Apply LDA AFTER survival feature generation.
        # Two separate transformers: one for the enhanced (survival) set,
        # one for the baseline set — each fitted only on its own X_train.
        if self.use_lda:
            from src.modules.machine_learning.utils.features.lda_transformer import LDATransformer

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
        model_name, pipeline_class, param,
        X_train, X_test, X_surv_train, X_surv_test,
        y_train, y_test,
    ):
        """Thin wrapper — delegates to the module-level _build_pipelines_fn."""
        return _build_pipelines_fn(
            model_name, pipeline_class, param,
            X_train, X_test, X_surv_train, X_surv_test,
            y_train, y_test,
            self.args, self.use_lda, self.lda_mode,
        )

    # -----------------------------
    # Single experiment runner
    # -----------------------------

    def run_model_experiment(self, model_name, pipeline_class, param, dataset, balance_strategy, threshold):
        """Thin wrapper — delegates to the module-level _run_model_experiment_fn."""
        return _run_model_experiment_fn(
            model_name, pipeline_class, param, dataset,
            balance_strategy, threshold,
            self.args, self.use_lda, self.lda_mode,
            self.feature_selection_baseline, self.feature_selection_enhanced,
        )

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
        completed_keys, prior_results = _load_checkpoint(self.checkpoint_path)

        for balance_strategy in self.balance_strategies:
            # Guard against thresholds=None when hybrid is selected
            strategy_thresholds = (self.thresholds or [None]) if balance_strategy == "hybrid" else [None]

            for threshold in strategy_thresholds:
                dataset = self.prepare_dataset(balance_strategy, threshold)

                for model_name, pipeline_class in self.models.items():
                    for param in self.parameters_by_model[model_name]:
                        # Full self-contained tuple — no SurvivalExperimentRunner instance needed
                        task_tuple = (
                            model_name, pipeline_class, param, dataset,
                            balance_strategy, threshold,
                            self.args, self.use_lda, self.lda_mode,
                            self.feature_selection_baseline, self.feature_selection_enhanced,
                        )

                        if model_name in self.do_not_parallel_compute:
                            sequential_tasks.append(task_tuple)
                        else:
                            parallel_tasks.append(task_tuple)

        parallel_tasks = [t for t in parallel_tasks if _make_task_key(t[0], t[2], t[4], t[5], t[9], t[10]) not in completed_keys]
        sequential_tasks = [t for t in sequential_tasks if _make_task_key(t[0], t[2], t[4], t[5], t[9], t[10]) not in completed_keys]

        if prior_results:
            print(f"[checkpoint] Resuming: skipping {len(prior_results)} already-completed experiments.", flush=True)

        progress_state["total"] = len(parallel_tasks) + len(sequential_tasks)
        progress_state["completed"] = 0
        progress_state["start_time"] = time.time()

        all_results = list(prior_results)

        if parallel_tasks:
            n_workers = cpu_count() if self.n_jobs == -1 else self.n_jobs
            with Pool(processes=n_workers, initializer=_worker_init, maxtasksperchild=4) as pool:
                for i, (key, result) in enumerate(
                    pool.imap_unordered(_run_task, parallel_tasks), start=1
                ):
                    all_results.append(result)
                    completed_keys.add(key)
                    _save_checkpoint(self.checkpoint_path, completed_keys, all_results)
                    progress_state["completed"] += 1
                    completed = progress_state["completed"]
                    total = progress_state["total"]
                    elapsed = time.time() - progress_state["start_time"]
                    avg_per_task = elapsed / completed
                    remaining = total - completed
                    eta = avg_per_task * remaining
                    print(
                        f"[{completed}/{total}] parallel task done | "
                        f"Elapsed: {_format_duration(elapsed)} | "
                        f"ETA: ~{_format_duration(eta)} remaining",
                        flush=True,
                    )

        for task_tuple in sequential_tasks:
            result = _run_model_experiment_fn(*task_tuple)
            key = _make_task_key(task_tuple[0], task_tuple[2], task_tuple[4], task_tuple[5], task_tuple[9], task_tuple[10])
            all_results.append(result)
            completed_keys.add(key)
            _save_checkpoint(self.checkpoint_path, completed_keys, all_results)
            progress_state["completed"] += 1
            completed = progress_state["completed"]
            total = progress_state["total"]
            elapsed = time.time() - progress_state["start_time"]
            avg_per_task = elapsed / completed
            remaining = total - completed
            eta = avg_per_task * remaining
            model_label = task_tuple[0]
            print(
                f"[{completed}/{total}] {model_label} | "
                f"Elapsed: {_format_duration(elapsed)} | "
                f"ETA: ~{_format_duration(eta)} remaining",
                flush=True,
            )

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                os.remove(self.checkpoint_path)
                print("[checkpoint] Run complete — checkpoint removed.", flush=True)
            except Exception:
                pass
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
        from src.modules.machine_learning import (
            AdaBoostPipeline, DecisionTreePipeline, GaussianNaiveBayesPipeline,
            KNearestNeighborPipeline, RandomForestPipeline, XGBoostPipeline,
            StackedEnsemblePipeline, MultiLayerPerceptronPipeline, TransformerPipeline,
            OrdinalPipeline, TwoStagePipeline,
        )
        from src.modules.machine_learning.utils.features.generate_survival_features import (
            generate_survival_features,
        )
        from src.modules.machine_learning.utils.data.data_preparation import DataPreparer

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
        # Always use prep_full_data (no LDA) here so that X_full contains
        # only the original feature columns.  LDA is applied in step 3,
        # AFTER survival features are generated, for the same reason as in
        # prepare_dataset(): the Cox scaler was fitted on the original columns
        # and raises ValueError if it encounters LD1/LD2/LD3.
        preparer = DataPreparer(
            df_data=self.df_data,
            target_feature=self.args.target_feature,
            verbose=True,
        )
        preparer.prep_full_data(balance_strategy=self.balance_strategy)

        X_full = preparer.X_train
        y_full = preparer.y_train

        # ── 2. Survival feature generation ───────────────────────────────────
        # Cox receives original-only features — no LD columns present yet.
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

        # ── 2b. LDA projection (after survival features) ─────────────────────
        # Applied here — not in step 1 — so Cox never sees LD columns.
        # Mirrors the ordering in SurvivalExperimentRunner.prepare_dataset().
        if self.use_lda:
            from src.modules.machine_learning.utils.features.lda_transformer import LDATransformer
            lda = LDATransformer(mode=self.lda_mode, verbose=True)
            X_enhanced = lda.fit_transform(X_enhanced, y_full)

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
            # use_lda / lda_mode forwarded so the deployed model's Stage 2
            # uses the same delinquent-only LDA as during experimentation.
            pipeline = TwoStagePipeline(
                X_enhanced, X_enhanced,
                y_full, y_full,
                self.args,
                stage1_estimator=s1_cls(**model_params["stage1"]),
                stage2_estimator=s2_cls(**model_params["stage2"]),
                use_lda=self.use_lda,
                lda_mode=self.lda_mode,
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

        # ── 5. Bundle into InferencePipeline ─────────────────────────────────
        # Wrap the fitted classifier together with every preprocessing
        # artifact so that the deployed pickle is fully self-contained.
        # The inference endpoint loads one file and calls inf.predict(X_raw)
        # with no caller-side preprocessing required.
        from src.modules.machine_learning.utils.inference.inference_pipeline import InferencePipeline

        inference_bundle = InferencePipeline(
            scaler              = preparer.scaler_,
            cox_model           = self.fitted_cph,
            time_points         = self.args.time_points,
            classifier_pipeline = pipeline,
            label_encoder       = preparer.label_encoder,
            lda_transformer     = lda if self.use_lda else None,
            model_key           = self.model_key,
            features            = pipeline.features,
            parameters          = getattr(self.args, "parameters", {}),
        )

        return inference_bundle, preparer.label_encoder