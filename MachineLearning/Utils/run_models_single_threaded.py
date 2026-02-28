import pandas as pd
import numpy as np
from MachineLearning.Utils.load_parameters import ParameterLoader
from MachineLearning.Utils.data_preparation_survival_analysis import SurvivalDataPreparer
from MachineLearning.Utils.generate_survival_features import generate_survival_features

def run_survival_model_experiments(
    df_data_surv,
    models,
    balance_strategies,
    args,
    best_penalty,
    thresholds=None,
    output_path="MachineLearning/Results/model_results.xlsx"
):
    """
    Run survival model experiments with baseline and enhanced survival features.

    Parameters
    ----------
    df_data_surv : pd.DataFrame
        Input survival dataset.
    models : dict
        Dictionary mapping model names to pipeline classes.
    balance_strategies : list
        List of balancing strategies to test.
    args : Namespace
        Arguments containing parameters_dir, target_feature, test_size, time_points, etc.
    best_penalty : float
        Penalty parameter for survival feature generation.
    thresholds : dict or None
        Dictionary mapping balance_strategy -> list of thresholds.
        Example: {"hybrid": [0.1, 0.2, 0.3], "random": [None]}
        If None, defaults to np.arange(0.1, 1.0, 0.1) for "hybrid" and [None] otherwise.
    output_path : str
        Path to save the results Excel file.
    """

    # Load parameters from JSON
    loader = ParameterLoader(args.parameters_dir)

    # Initialize preparer class once
    preparer = SurvivalDataPreparer(
        df_data_surv,
        target_feature=args.target_feature,
        time_feature="days_elapsed_until_fully_paid",
        censor_feature="censor",
        test_size=args.test_size,
        verbose=False
    )

    all_results = []

    for balance_strategy in balance_strategies:
        # Decide thresholds based on strategy
        strategy_thresholds = thresholds if balance_strategy == "hybrid" else [None]

        for threshold in strategy_thresholds:
            print(f"Preparing data with balance_strategy={balance_strategy}, "
                f"undersample_threshold={threshold}")

            # Prepare data once per balance_strategy/threshold
            preparer.prep_data(
                balance_strategy=balance_strategy,
                undersample_threshold=threshold
            )

            X_train, X_test = preparer.X_train, preparer.X_test
            y_train, y_test = preparer.y_train, preparer.y_test
            T_train, T_test = preparer.T_train, preparer.T_test
            E_train, E_test = preparer.E_train, preparer.E_test

            # Generate survival features once per dataset
            X_survival_train = generate_survival_features(
                X_train, T_train, E_train, best_penalty, time_points=args.time_points
            )
            X_survival_test = generate_survival_features(
                X_test, T_test, E_test, best_penalty, time_points=args.time_points
            )

            # Now loop over models and parameters
            for model_name, pipeline_class in models.items():
                param_list = loader.get_parameters(model_name)

                for param in param_list:
                    print(
                        f"Running {model_name} with parameters: {param}, "
                        f"balance_strategy={balance_strategy}, "
                        f"undersample_threshold={threshold}"
                    )

                    # Baseline pipeline
                    pipeline_baseline = pipeline_class(
                        X_train, X_test, y_train, y_test, args, param
                    )
                    result_baseline = (
                        pipeline_baseline.initialize_model()
                        .fit(use_feature_selection=True)
                        .evaluate()
                        .show_results()
                    )
                    features_baseline = pipeline_baseline.get_selected_features()

                    # Enhanced pipeline
                    pipeline_enhanced = pipeline_class(
                        X_survival_train, X_survival_test, y_train, y_test, args, param
                    )
                    result_enhanced = (
                        pipeline_enhanced.initialize_model()
                        .fit(use_feature_selection=True)
                        .evaluate()
                        .show_results()
                    )
                    features_enhanced = pipeline_enhanced.get_selected_features()

                    # Merge results
                    result = {
                        "model": model_name,
                        "parameters": str(param),
                        "balance_strategy": balance_strategy,
                        "undersample_threshold": threshold,
                        "baseline": result_baseline,
                        "enhanced": result_enhanced,
                        "baseline_features": features_baseline,
                        "enhanced_features": features_enhanced
                    }
                    all_results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Normalize baseline and enhanced columns
    baseline_df = pd.json_normalize(results_df["baseline"]).add_prefix("baseline_")
    enhanced_df = pd.json_normalize(results_df["enhanced"]).add_prefix("enhanced_")

    # Concatenate back with the original dataframe
    results_df = pd.concat(
        [results_df.drop(columns=["baseline", "enhanced"]), baseline_df, enhanced_df],
        axis=1
    )

    # Export to Excel
    results_df.to_excel(output_path, index=False)
    print(f"All results saved to {output_path}")

    return results_df