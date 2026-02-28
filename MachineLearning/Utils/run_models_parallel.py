import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
from MachineLearning.Utils.load_parameters import ParameterLoader
from MachineLearning.Utils.data_preparation_survival_analysis import SurvivalDataPreparer
from MachineLearning.Utils.generate_survival_features import generate_survival_features


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
    def __init__(self, df_data_surv, models, balance_strategies, args,
                 best_penalty, thresholds=None, n_jobs=-1,
                 output_path="MachineLearning/Results/model_results.xlsx",
                 do_not_parallel_compute=None, feature_selection=True):
        self.df_data_surv = df_data_surv
        self.models = models
        self.balance_strategies = balance_strategies
        self.args = args
        self.best_penalty = best_penalty
        self.thresholds = thresholds
        self.n_jobs = n_jobs
        self.output_path = output_path
        self.do_not_parallel_compute = do_not_parallel_compute or []
        self.feature_selection = feature_selection

        # Initialize parameter loader and data preparer
        self.loader = ParameterLoader(args.parameters_dir)
        self.parameters_by_model = {m: self.loader.get_parameters(m) for m in models}
        self.preparer = SurvivalDataPreparer(
            df_data_surv,
            target_feature=args.target_feature,
            time_feature="days_elapsed_until_fully_paid",
            censor_feature="censor",
            test_size=args.test_size,
            verbose=False
        )

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
            T_train, T_test, E_train, E_test, X_survival_train, X_survival_test).
        """
        self.preparer.prep_data(balance_strategy=balance_strategy,
                                undersample_threshold=threshold)

        X_train, X_test = self.preparer.X_train, self.preparer.X_test
        y_train, y_test = self.preparer.y_train, self.preparer.y_test
        T_train, T_test = self.preparer.T_train, self.preparer.T_test
        E_train, E_test = self.preparer.E_train, self.preparer.E_test

        # Cache survival features once
        X_survival_train = generate_survival_features(
            X_train, T_train, E_train, self.best_penalty, time_points=self.args.time_points
        )
        X_survival_test = generate_survival_features(
            X_test, T_test, E_test, self.best_penalty, time_points=self.args.time_points
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
        and feature selection, then returns flattened results.

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
        dict
            Flattened dictionary of results including baseline and enhanced metrics,
            selected features, and metadata.
        """
        (X_train, X_test, y_train, y_test,
         X_survival_train, X_survival_test) = dataset

        # Baseline pipeline
        pipeline_baseline = pipeline_class(X_train, X_test, y_train, y_test, self.args, param)
        result_baseline = (
            pipeline_baseline.initialize_model()
            .fit(use_feature_selection=self.feature_selection)
            .evaluate()
            .show_results()
        )
        features_baseline = pipeline_baseline.get_selected_features()

        # Enhanced pipeline
        pipeline_enhanced = pipeline_class(X_survival_train, X_survival_test, y_train, y_test, self.args, param)
        result_enhanced = (
            pipeline_enhanced.initialize_model()
            .fit(use_feature_selection=self.feature_selection)
            .evaluate()
            .show_results()
        )
        features_enhanced = pipeline_enhanced.get_selected_features()

        # Flatten results directly
        return {
            "model": model_name,
            "parameters": str(param),
            "balance_strategy": balance_strategy,
            "undersample_threshold": threshold,
            **{f"baseline_{k}": v for k, v in result_baseline.items()},
            **{f"enhanced_{k}": v for k, v in result_enhanced.items()},
            "baseline_features": features_baseline,
            "enhanced_features": features_enhanced
        }

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
        pd.DataFrame
            Results DataFrame with flattened baseline and enhanced metrics,
            exported to Excel at the specified output path.
        """
        parallel_tasks = []
        sequential_tasks = []

        # Loop over balance strategies first (prep once per dataset)
        for balance_strategy in self.balance_strategies:
            strategy_thresholds = self.thresholds if balance_strategy == "hybrid" else [None]

            for threshold in strategy_thresholds:
                dataset = self.prepare_dataset(balance_strategy, threshold)

                # Build tasks for all models/params using this dataset
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
            with self.tqdm_joblib(tqdm(desc="Running parallel experiments", total=len(parallel_tasks), unit="exp")):
                parallel_results = Parallel(n_jobs=self.n_jobs)(parallel_tasks)

        # Run sequential tasks
        sequential_results = []
        if sequential_tasks:
            for task_args in tqdm(sequential_tasks, desc="Running sequential experiments", unit="exp"):
                result = self.run_model_experiment(*task_args)
                sequential_results.append(result)

        # Combine results
        all_results = parallel_results + sequential_results

        # Convert to DataFrame directly (already flattened)
        results_df = pd.DataFrame(all_results)

        # Export
        results_df.to_excel(self.output_path, index=False)
        print(f"All results saved to {self.output_path}")

        return results_df