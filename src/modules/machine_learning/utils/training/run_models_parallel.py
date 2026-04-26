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


def _make_model_cache_key(model_name: str, param_str: str, balance_strategy: str, 
                         threshold=None, fs_mode: str = "enhanced") -> str:
    """Deterministic key for caching trained models."""
    threshold_str = f"@{threshold}" if threshold is not None else ""
    param_hash = str(hash(param_str) & 0x7FFFFFFF)[:8]
    return f"model_{model_name}_{balance_strategy}{threshold_str}_{fs_mode}_{param_hash}.pkl"


def _load_cached_model(cache_dir: str, model_name: str, param_str: str, 
                      balance_strategy: str, threshold=None, fs_mode: str = "enhanced"):
    """Load cached trained model and evaluation results."""
    if cache_dir is None or not os.path.exists(cache_dir):
        return None, None
    
    cache_key = _make_model_cache_key(model_name, param_str, balance_strategy, threshold, fs_mode)
    cache_path = os.path.join(cache_dir, cache_key)
    
    if not os.path.exists(cache_path):
        return None, None
    
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"[model-cache] Hit: {model_name} ({balance_strategy}{('@'+str(threshold)) if threshold else ''})", flush=True)
        return data.get("pipeline"), data.get("results")
    except Exception as e:
        print(f"[model-cache] Could not load {cache_key} ({e}); retraining", flush=True)
        return None, None


def _save_cached_model(cache_dir: str, model_name: str, param_str: str, 
                      balance_strategy: str, threshold, pipeline, results,
                      fs_mode: str = "enhanced") -> None:
    """Save trained model and evaluation results to disk cache."""
    if cache_dir is None:
        return
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _make_model_cache_key(model_name, param_str, balance_strategy, threshold, fs_mode)
        cache_path = os.path.join(cache_dir, cache_key)
        tmp_path = cache_path + ".tmp"
        
        with open(tmp_path, "wb") as f:
            pickle.dump({"pipeline": pipeline, "results": results}, f)
        os.replace(tmp_path, cache_path)
        print(f"[model-cache] Saved: {cache_key}", flush=True)
    except Exception as e:
        print(f"[model-cache] Warning: could not save model ({e})", flush=True)


from src.modules.machine_learning.utils.data.data_preparation import DataPreparer
from src.modules.machine_learning.utils.training.load_parameters import ParameterLoader
from src.modules.machine_learning.utils.features.generate_survival_features import generate_survival_features
from src.modules.machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period
from src.modules.machine_learning.models.ordinal_classifier import OrdinalPipeline
from src.modules.machine_learning.models.two_stage_classifier import TwoStagePipeline
from src.modules.machine_learning.utils.training.cache_manager import CacheManager


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
    model_cache_dir=None,
    cache_key=None,
):
    (X_train, X_test, y_train, y_test,
     X_survival_train, X_survival_test) = dataset

    if model_name in _TWO_STAGE_ESTIMATOR_PAIRS:
        param_str = f"stage1={param['stage1']}, stage2={param['stage2']}"
    else:
        param_str = str(sorted({k: v for k, v in param.items() if k != "scale_pos_weight"}.items()))

    pipeline_baseline, pipeline_enhanced = _build_pipelines_fn(
        model_name, pipeline_class, param,
        X_train, X_test, X_survival_train, X_survival_test,
        y_train, y_test,
        args, use_lda, lda_mode,
    )

    result_baseline = None
    result_enhanced = None
    
    if model_cache_dir:
        _, result_baseline = _load_cached_model(model_cache_dir, model_name, param_str, balance_strategy, threshold, "baseline")
        _, result_enhanced = _load_cached_model(model_cache_dir, model_name, param_str, balance_strategy, threshold, "enhanced")
    
    if result_baseline is not None and result_enhanced is not None:
        result = {
            "model": model_name, "underscore_threshold": threshold, "parameters": param_str,
            "balance_strategy": balance_strategy, "cache_key": cache_key,
            **{f"baseline_{k}": v for k, v in result_baseline.items()},
            **{f"enhanced_{k}": v for k, v in result_enhanced.items()},
        }
        return result

    with joblib.parallel_backend('threading'):
        if result_baseline is None:
            pipeline_baseline.initialize_model().fit(use_feature_selection=fs_baseline)
            result_baseline = pipeline_baseline.evaluate().show_results()
            if model_cache_dir:
                _save_cached_model(model_cache_dir, model_name, param_str, balance_strategy, threshold, pipeline_baseline, result_baseline, "baseline")

        if result_enhanced is None:
            pipeline_enhanced.initialize_model().fit(use_feature_selection=fs_enhanced)
            result_enhanced = pipeline_enhanced.evaluate().show_results()
            if model_cache_dir:
                _save_cached_model(model_cache_dir, model_name, param_str, balance_strategy, threshold, pipeline_enhanced, result_enhanced, "enhanced")

    result = {
        "model": model_name, "underscore_threshold": threshold, "parameters": param_str,
        "balance_strategy": balance_strategy, "cache_key": cache_key,
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
    return result


def _run_task(task_tuple):
    result = _run_model_experiment_fn(*task_tuple)
    key = _make_task_key(task_tuple[0], task_tuple[2], task_tuple[4], task_tuple[5], task_tuple[9], task_tuple[10])
    return key, result


class SurvivalExperimentRunner:
    def __init__(
        self, df_data, df_data_surv, models, balance_strategies, args,
        best_parameters, thresholds=None, n_jobs=4, do_not_parallel_compute=None,
        feature_selection_baseline=True, feature_selection_enhanced=True,
        use_lda=False, lda_mode="append", checkpoint_path=None, cache_dir="data/cache",
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
        self.use_lda = use_lda
        self.lda_mode = lda_mode
        self.checkpoint_path = checkpoint_path
        self.cache = CacheManager(cache_root=cache_dir)
        self.loader = ParameterLoader(args.parameters_dir)

        # Expand aliases
        expanded = {}
        for m, cls in models.items():
            if m == "ordinal":
                for v in _ORDINAL_ESTIMATOR_MAP: expanded[v] = OrdinalPipeline
            elif m == "two_stage":
                for v in _TWO_STAGE_ESTIMATOR_PAIRS: expanded[v] = TwoStagePipeline
            else: expanded[m] = cls
        self.models = expanded

        for m in self.models:
            if "xgb" in m and m not in self.do_not_parallel_compute:
                self.do_not_parallel_compute.append(m)

        self.parameters_by_model = {m: self.loader.get_parameters(m) for m in self.models}
        self.preparer = DataPreparer(df_data, target_feature=args.target_feature, test_size=args.test_size, verbose=False).encode_labels().train_test_split()
        self.class_mappings = self.preparer.class_mapping
        self._X_train_orig = self.preparer.X_train.copy()
        self._X_test_orig  = self.preparer.X_test.copy()
        self._y_train_orig = self.preparer.y_train.copy()
        self._y_test_orig  = self.preparer.y_test.copy()
        self._train_indices = self.preparer.X_train.index
        self._fitted_cox_model = None
        self._cox_scaler = None
        self.feature_cache_dir = os.path.join(cache_dir, "survival_features")
        self.model_cache_dir = os.path.join(cache_dir, "models")

    def _fit_cox_model_once(self):
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        from src.modules.machine_learning.utils.data.clean_survival_inputs import clean_survival_inputs
        from src.modules.machine_learning.utils.features.generate_survival_features import _safe_scale

        print("[cox] Fitting Cox PH model on original train data ...", flush=True)
        cox_start = time.time()
        _train_mask = self.df_data_surv.index.isin(self._train_indices) | (self.df_data_surv["censor"] == 0)
        _df_surv_train = self.df_data_surv[_train_mask]
        X_surv = _df_surv_train.drop(columns=["days_elapsed_until_fully_paid", "censor"])
        T = adjust_payment_period(_df_surv_train["days_elapsed_until_fully_paid"])
        E = _df_surv_train["censor"]

        _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)
        X_fit_raw = df_fit[[c for c in df_fit.columns if c not in ("T", "E")]]
        X_fit_scaled, fit_scaler = _safe_scale(X_fit_raw)
        y_fit = Surv.from_arrays(event=df_fit["E"].astype(bool).values, time=df_fit["T"].astype(float).values)

        cox = CoxnetSurvivalAnalysis(l1_ratio=self.best_parameters["l1_ratio"], alphas=[self.best_parameters["alpha"]], fit_baseline_model=True, max_iter=100_000, tol=1e-7)
        cox.fit(X_fit_scaled, y_fit)
        print(f"[cox] Done ({_format_duration(time.time() - cox_start)})", flush=True)
        return cox, fit_scaler

    def prepare_dataset(self, balance_strategy, threshold):
        label = f"{balance_strategy}" + (f"@{threshold}" if threshold is not None else "")
        import hashlib
        key_src = f"{balance_strategy}|{threshold}|{self.args.observation_end}|{self.args.test_size}"
        cache_key = hashlib.md5(key_src.encode()).hexdigest()

        print(f"[dataset] Preparing: {label} ...", flush=True)
        _ds_start = time.time()
        self.preparer.X_train = self._X_train_orig.copy()
        self.preparer.y_train = self._y_train_orig.copy()
        self.preparer.resample(balance_strategy=balance_strategy, undersample_threshold=threshold)
        
        X_train_raw, X_test_raw = self.preparer.X_train.copy(), self.preparer.X_test.copy()
        y_train, y_test = self.preparer.y_train.copy(), self.preparer.y_test.copy()

        if self._fitted_cox_model is None:
            self._fitted_cox_model, self._cox_scaler = self._fit_cox_model_once()

        X_surv_train, X_surv_test = _load_cached_features(self.feature_cache_dir, balance_strategy, threshold, self.args.test_size)
        
        if X_surv_train is None:
            _train_mask = self.df_data_surv.index.isin(self._train_indices) | (self.df_data_surv["censor"] == 0)
            _df_surv_train = self.df_data_surv[_train_mask]
            X_surv = _df_surv_train.drop(columns=["days_elapsed_until_fully_paid", "censor"])
            T = adjust_payment_period(_df_surv_train["days_elapsed_until_fully_paid"])
            E = _df_surv_train["censor"]
            X_surv_train, X_surv_test = generate_survival_features(X_surv, T, E, X_train_raw, X_test_raw, best_params=None, time_points=self.args.time_points, fitted_cph=self._fitted_cox_model, cox_scaler=self._cox_scaler)
            _save_cached_features(self.feature_cache_dir, balance_strategy, threshold, X_surv_train, X_surv_test, self.args.test_size)
            
        if self.use_lda:
            from src.modules.machine_learning.utils.features.lda_transformer import LDATransformer
            lda_enhanced = LDATransformer(mode=self.lda_mode)
            X_surv_train = lda_enhanced.fit_transform(X_surv_train, y_train)
            X_surv_test  = lda_enhanced.transform(X_surv_test)
            lda_baseline = LDATransformer(mode=self.lda_mode)
            X_train = lda_baseline.fit_transform(X_train_raw, y_train)
            X_test  = lda_baseline.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        print(f"[dataset] Done ({_format_duration(time.time() - _ds_start)})", flush=True)
        return (X_train, X_test, y_train, y_test, X_surv_train, X_surv_test), cache_key

    def run(self):
        parallel_tasks = []
        sequential_tasks = []
        completed_keys, prior_results = _load_checkpoint(self.checkpoint_path)

        for balance_strategy in self.balance_strategies:
            strategy_thresholds = (self.thresholds or [None]) if balance_strategy == "hybrid" else [None]
            for threshold in strategy_thresholds:
                dataset, cache_key = self.prepare_dataset(balance_strategy, threshold)
                for model_name, pipeline_class in self.models.items():
                    for param in self.parameters_by_model[model_name]:
                        task_tuple = (model_name, pipeline_class, param, dataset, balance_strategy, threshold, self.args, self.use_lda, self.lda_mode, self.feature_selection_baseline, self.feature_selection_enhanced, self.model_cache_dir, cache_key)
                        if model_name in self.do_not_parallel_compute: sequential_tasks.append(task_tuple)
                        else: parallel_tasks.append(task_tuple)

        parallel_tasks = [t for t in parallel_tasks if _make_task_key(t[0], t[2], t[4], t[5], t[9], t[10]) not in completed_keys]
        sequential_tasks = [t for t in sequential_tasks if _make_task_key(t[0], t[2], t[4], t[5], t[9], t[10]) not in completed_keys]

        all_results = list(prior_results)
        if parallel_tasks:
            n_workers = cpu_count() if self.n_jobs == -1 else self.n_jobs
            with Pool(processes=n_workers, initializer=_worker_init, maxtasksperchild=4) as pool:
                for i, (key, result) in enumerate(pool.imap_unordered(_run_task, parallel_tasks), start=1):
                    all_results.append(result)
                    completed_keys.add(key)
                    _save_checkpoint(self.checkpoint_path, completed_keys, all_results)
                    print(f"[{len(all_results)}/{progress_state['total']}] parallel task done", flush=True)

        for task_tuple in sequential_tasks:
            result = _run_model_experiment_fn(*task_tuple)
            key = _make_task_key(task_tuple[0], task_tuple[2], task_tuple[4], task_tuple[5], task_tuple[9], task_tuple[10])
            all_results.append(result)
            completed_keys.add(key)
            _save_checkpoint(self.checkpoint_path, completed_keys, all_results)
            print(f"[{len(all_results)}/{progress_state['total']}] {task_tuple[0]} done", flush=True)

        return pd.DataFrame(all_results), self.class_mappings


class FinalizationRunner:
    def __init__(self, df_data, df_data_surv, model_key, balance_strategy, args, best_surv_params, fitted_cph=None, use_lda=False, lda_mode="append", feature_metadata=None):
        self.df_data = df_data
        self.df_data_surv = df_data_surv
        self.model_key = model_key
        self.balance_strategy = balance_strategy
        self.args = args
        self.best_surv_params = best_surv_params
        self.fitted_cph = fitted_cph
        self.use_lda = use_lda
        self.lda_mode = lda_mode
        self.feature_metadata = feature_metadata or {}

    def train(self):
        from src.modules.machine_learning import AdaBoostPipeline, DecisionTreePipeline, GaussianNaiveBayesPipeline, KNearestNeighborPipeline, RandomForestPipeline, XGBoostPipeline, StackedEnsemblePipeline, MultiLayerPerceptronPipeline, TransformerPipeline, OrdinalPipeline, TwoStagePipeline
        from src.modules.machine_learning.utils.features.generate_survival_features import generate_survival_features
        from src.modules.machine_learning.utils.data.data_preparation import DataPreparer
        from src.modules.machine_learning.utils.inference.inference_pipeline import InferencePipeline

        preparer = DataPreparer(df_data=self.df_data, target_feature=self.args.target_feature, verbose=True)
        preparer.prep_full_data(balance_strategy=self.balance_strategy)
        X_full, y_full = preparer.X_train, preparer.y_train
        X_surv = self.df_data_surv.drop(columns=["days_elapsed_until_fully_paid", "censor"])
        T = adjust_payment_period(self.df_data_surv["days_elapsed_until_fully_paid"])
        E = self.df_data_surv["censor"]
        X_enhanced = generate_survival_features(X_surv, T, E, X_full, None, self.best_surv_params, self.args.time_points, self.fitted_cph)

        lda = None
        if self.use_lda:
            from src.modules.machine_learning.utils.features.lda_transformer import LDATransformer
            lda = LDATransformer(mode=self.lda_mode, verbose=True)
            X_enhanced = lda.fit_transform(X_enhanced, y_full)

        model_params = getattr(self.args, "parameters", {}) or {}
        if self.model_key in _TWO_STAGE_ESTIMATOR_PAIRS:
            s1_cls, s2_cls = _TWO_STAGE_ESTIMATOR_PAIRS[self.model_key]
            pipeline = TwoStagePipeline(X_enhanced, X_enhanced, y_full, y_full, self.args, stage1_estimator=s1_cls(**model_params["stage1"]), stage2_estimator=s2_cls(**model_params["stage2"]), use_lda=self.use_lda, lda_mode=self.lda_mode)
        elif self.model_key in _ORDINAL_ESTIMATOR_MAP:
            pipeline = OrdinalPipeline(X_enhanced, X_enhanced, y_full, y_full, self.args, parameters={"scale_pos_weight": model_params.get("scale_pos_weight", True)}, base_estimator=_ORDINAL_ESTIMATOR_MAP[self.model_key](**{k:v for k,v in model_params.items() if k!="scale_pos_weight"}))
        else:
            MAP = {"ada_boost": AdaBoostPipeline, "decision_tree": DecisionTreePipeline, "gaussian_naive_bayes": GaussianNaiveBayesPipeline, "knn": KNearestNeighborPipeline, "random_forest": RandomForestPipeline, "xgboost": XGBoostPipeline}
            pipeline = MAP[self.model_key](X_enhanced, X_enhanced, y_full, y_full, self.args, model_params)

        pipeline.initialize_model().fit(use_feature_selection=True)
        return InferencePipeline(preparer.scaler_, self.fitted_cph, self.args.time_points, pipeline, preparer.label_encoder, lda, self.model_key, pipeline.features, model_params, self.feature_metadata), preparer.label_encoder
