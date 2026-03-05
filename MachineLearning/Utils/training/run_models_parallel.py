import json
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm
from contextlib import contextmanager
from MachineLearning.Utils.training.load_parameters import ParameterLoader
from MachineLearning.Utils.data.data_preparation import DataPreparer
from MachineLearning.Utils.features.generate_survival_features import generate_survival_features
from MachineLearning.Utils.features.adjust_survival_time_periods import adjust_payment_period

# Ensures compatibility with Dash by resetting/shutting down the joblib executor
# so that new parallel tasks can be scheduled without hitting ShutdownExecutorError.
get_reusable_executor().shutdown(wait=True)

# Used for the progress bar in Dash
progress_state = {"completed": 0, "total": 0}

class SurvivalExperimentRunner:
    """
    SurvivalExperimentRunner
    ------------------------
    A class to manage and execute survival model experiments with baseline and enhanced
    survival features. Supports parallel and sequential execution, dataset preparation,
    feature generation, and results export.

    This class encapsulates:
    - Dataset preparation per balance strategy and threshold
    - Survival feature generation (cached per dataset)
    - Model training and evaluation (baseline vs. enhanced)
    - Parallel execution with tqdm progress bars
    - Flattened results export to Excel

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
    def __init__(self, df_data, df_data_surv, models, balance_strategies, args,
                 best_penalty, thresholds=None, n_jobs=-1,
                 do_not_parallel_compute=None,
                 feature_selection_baseline=True, feature_selection_enhanced=True):
        self.df_data = df_data
        self.df_data_surv = df_data_surv
        self.models = models
        self.balance_strategies = balance_strategies
        self.args = args
        self.best_penalty = best_penalty
        self.thresholds = thresholds
        self.n_jobs = n_jobs
        self.do_not_parallel_compute = do_not_parallel_compute or []
        self.feature_selection_baseline = feature_selection_baseline
        self.feature_selection_enhanced = feature_selection_enhanced

        # Initialize parameter loader and data preparer
        self.loader = ParameterLoader(args.parameters_dir)
        self.parameters_by_model = {m: self.loader.get_parameters(m) for m in models}
        self.preparer = DataPreparer(
            df_data,
            target_feature=args.target_feature,
            test_size=args.test_size,
            verbose=False
        ).encode_labels().train_test_split()

        self.class_mappings = self.preparer.class_mapping

    # -----------------------------
    # tqdm-joblib integration
    # -----------------------------
    @staticmethod
    @contextmanager
    def tqdm_joblib(tqdm_object):
        """
        Integrates tqdm progress bars with joblib parallel execution.

        Parameters
        ----------
        tqdm_object : tqdm
            A tqdm progress bar instance.

        Yields
        ------
        tqdm_object : tqdm
            The same tqdm object, patched to update with joblib task completion.
        """
        from joblib.parallel import BatchCompletionCallBack

        class TqdmBatchCompletionCallback(BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_callback = BatchCompletionCallBack
        try:
            import joblib.parallel
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_callback
            tqdm_object.close()

    # -----------------------------
    # tqdm-joblib integration for Dash
    # -----------------------------
    @staticmethod
    @contextmanager
    def tqdm_joblib(total):
        from joblib.parallel import BatchCompletionCallBack
        import joblib.parallel
        import threading

        class DashTqdm:
            def __init__(self, total):
                self.lock = threading.Lock()
                progress_state["completed"] = 0
                progress_state["total"] = total

            def update(self, n=1):
                with self.lock:
                    progress_state["completed"] += n

        dash_tqdm = DashTqdm(total)

        class TqdmBatchCompletionCallback(BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                dash_tqdm.update(self.batch_size)
                return super().__call__(*args, **kwargs)

        old_callback = BatchCompletionCallBack
        try:
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            yield dash_tqdm
        finally:
            joblib.parallel.BatchCompletionCallBack = old_callback

    # -----------------------------
    # Dataset preparation helper
    # -----------------------------
    def prepare_dataset(self, balance_strategy, threshold):
        """
        Prepare dataset for a given balance strategy and threshold.

        Runs data preparation, splits train/test sets, and generates survival features.
        Results are cached and reused across models and parameters.

        Parameters
        ----------
        balance_strategy : str
            Balancing strategy (e.g., "undersample", "oversample", "hybrid").
        threshold : float or None
            Threshold for hybrid balancing strategy.

        Returns
        -------
        tuple
            A tuple containing (X_train, X_test, y_train, y_test,
            X_survival_train, X_survival_test).
        """
        self.preparer.prep_data(balance_strategy=balance_strategy,
                                undersample_threshold=threshold)

        X_train, X_test = self.preparer.X_train, self.preparer.X_test
        y_train, y_test = self.preparer.y_train, self.preparer.y_test

        # Generate survival features
        X_surv = self.df_data_surv.drop(columns=['days_elapsed_until_fully_paid', 'censor'])
        T = adjust_payment_period(self.df_data_surv['days_elapsed_until_fully_paid'])
        E = self.df_data_surv['censor']

        X_survival_train, X_survival_test = generate_survival_features(
            X_surv, T, E, X_train, X_test, self.best_penalty, time_points=self.args.time_points
        )

        return (X_train, X_test, y_train, y_test,
                X_survival_train, X_survival_test)

    # -----------------------------
    # Single experiment runner
    # -----------------------------
    def run_model_experiment(self, model_name, pipeline_class, param,
                             dataset, balance_strategy, threshold):
        """
        Run a single experiment for a given model, parameter set, and dataset.

        Executes both baseline and enhanced pipelines, performs training, evaluation,
        and feature selection, then returns a result entry with a unique composite key.

        Parameters
        ----------
        model_name : str
            Name of the model being tested.
        pipeline_class : class
            Pipeline class implementing the model workflow.
        param : dict
            Parameter set for the model.
        dataset : tuple
            Prepared dataset including survival features.
        balance_strategy : str
            Balancing strategy used.
        threshold : float or None
            Threshold for hybrid balancing strategy.

        Returns
        -------
        tuple
            (unique_key, result_dict) where unique_key is a composite of
            model_name, balance_strategy, threshold, and param index.
        """
        (X_train, X_test, y_train, y_test,
         X_survival_train, X_survival_test) = dataset

        # Baseline pipeline
        pipeline_baseline = pipeline_class(X_train, X_test, y_train, y_test, self.args, param)
        pipeline_baseline.initialize_model().fit(use_feature_selection=self.feature_selection_baseline)
        result_baseline = pipeline_baseline.evaluate().show_results()
        features_baseline = pipeline_baseline.get_selected_features()

        # Enhanced pipeline
        pipeline_enhanced = pipeline_class(X_survival_train, X_survival_test, y_train, y_test, self.args, param)
        pipeline_enhanced.initialize_model().fit(use_feature_selection=self.feature_selection_enhanced)
        result_enhanced = pipeline_enhanced.evaluate().show_results()
        features_enhanced = pipeline_enhanced.get_selected_features()

        # Unique key: model + strategy + threshold + param fingerprint
        threshold_str = str(threshold) if threshold is not None else "none"
        param_str = str(sorted(param.items())) if isinstance(param, dict) else str(param)
        unique_key = f"{model_name}__{balance_strategy}__{threshold_str}__{param_str}"

        result = {
            "model": model_name,
            "parameters": param_str,
            "balance_strategy": balance_strategy,
            "undersample_threshold": threshold,
            "baseline": {
                "evaluation": result_baseline,
                "features": features_baseline,
            },
            "enhanced": {
                "evaluation": result_enhanced,
                "features": features_enhanced,
            },
        }

        return unique_key, result

    # -----------------------------
    # Main experiment runner
    # -----------------------------
    def run(self):
        """
        Run all survival model experiments.

        Iterates over balance strategies and thresholds, prepares datasets,
        builds tasks for all models and parameters, executes them in parallel
        (or sequentially if specified), and aggregates results.

        Returns
        -------
        tuple
            A 2-tuple of (all_results, class_mappings):

            all_results : dict
                Dictionary keyed by unique experiment key. Each entry contains:
                - parameters: str
                    Stringified parameter set used for the model.
                - balance_strategy: str
                    Balancing strategy applied to the dataset.
                - undersample_threshold: float or None
                    Threshold used for hybrid balancing strategy.
                - baseline: dict
                    Contains evaluation results (metrics + raw chart data)
                    and selected features for the baseline pipeline.
                - enhanced: dict
                    Contains evaluation results (metrics + raw chart data)
                    and selected features for the enhanced pipeline.

            class_mappings : dict
                Dictionary mapping original class labels to their encoded
                integer representations, as produced by the label encoder
                during dataset preparation.
        """
        parallel_tasks = []
        sequential_tasks = []

        for balance_strategy in self.balance_strategies:
            strategy_thresholds = self.thresholds if balance_strategy == "hybrid" else [None]

            for threshold in strategy_thresholds:
                dataset = self.prepare_dataset(balance_strategy, threshold)

                for model_name, pipeline_class in self.models.items():
                    for param in self.parameters_by_model[model_name]:
                        task_args = (model_name, pipeline_class, param, dataset, balance_strategy, threshold)
                        if model_name in self.do_not_parallel_compute:
                            sequential_tasks.append(task_args)
                        else:
                            parallel_tasks.append(delayed(self.run_model_experiment)(*task_args))

        # Run parallel tasks
        parallel_results = []
        if parallel_tasks:
            with self.tqdm_joblib(total=len(parallel_tasks)):
                parallel_results = Parallel(n_jobs=self.n_jobs)(parallel_tasks)

        # Run sequential tasks
        sequential_results = []
        if sequential_tasks:
            for task_args in tqdm(sequential_tasks, desc="Running sequential experiments", unit="exp"):
                result = self.run_model_experiment(*task_args)
                sequential_results.append(result)

        # Combine — each entry is now (unique_key, result_dict) so no key collisions
        all_results = {}
        for unique_key, result in parallel_results + sequential_results:
            all_results[unique_key] = result

        return all_results, self.class_mappings