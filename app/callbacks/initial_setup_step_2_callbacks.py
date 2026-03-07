import pandas as pd

from dash import Input, Output, State, html, dcc
from app import dash_app

html_step_2 = (
    html.Div([
        html.H3("Step 2: Model Training Selection", className="step-header"),

        # Models Section
        html.H4("Model Selection", className="sub-header"),
        html.P("Select from the list of models below to train:", className="section-description"),

        html.Div([
            html.H5("Tree-based Models", className="sub-sub-header"),
            dcc.Checklist(
                id="model_selection_trees",
                options=[
                    {"label": "Decision Tree", "value": "decision_tree"},
                    {"label": "Random Forest", "value": "random_forest"},
                    {"label": "XGBoost", "value": "xgboost"},
                    {"label": "AdaBoost", "value": "ada_boost"},
                ],
                value=["decision_tree", "random_forest", "xgboost", "ada_boost"],
                className="checklist"
            ),

            html.H5("Probabilistic Models", className="sub-sub-header"),
            dcc.Checklist(
                id="model_selection_prob",
                options=[{"label": "Gaussian Naive Bayes", "value": "gaussian_naive_bayes"}],
                value=["gaussian_naive_bayes"],
                className="checklist"
            ),

            html.H5("Neighbor-based Models", className="sub-sub-header"),
            dcc.Checklist(
                id="model_selection_neighbors",
                options=[{"label": "K Nearest Neighbors (KNN)", "value": "knn"}],
                value=["knn"],
                className="checklist"
            ),

            html.H5("Neural Network Models", className="sub-sub-header"),
            dcc.Checklist(
                id="model_selection_nn",
                options=[{"label": "Multi-layer Perceptron (MLP)", "value": "nn_mlp"}],
                value=["nn_mlp"],
                className="checklist"
            ),
        ], className="section-block"),

        # Balancing Strategies Section
        html.H4("Balancing Strategies", className="sub-header"),
        html.P("Select from the list of balancing strategies below to use in training:", className="section-description"),
        dcc.Checklist(
            id="balancing_selection",
            options=[
                {"label": "None (No balancing technique will be used)", "value": "none"},
                {"label": "SMOTE (Synthetic Minority Over Sampling Technique)", "value": "smote"},
                {"label": "Borderline SMOTE", "value": "borderline_smote"},
                {"label": "SMOTE-ENN (Edited Nearest Neighbors)", "value": "smote_enn"},
                {"label": "SMOTE Tomek (Tomek Links)", "value": "smote_tomek"},
            ],
            value=["none", "smote", "borderline_smote", "smote_enn", "smote_tomek"],
            className="checklist"
        ),

        html.Button("Confirm Parameters", id="model_parameters_confirm_btn", className="confirm-btn", disabled=True)
    ], className="step-container")
)

#########################################################
#  STEP 2 MODELS - SELECTING MODEL AND BAlANCE TECHNIQUES
#########################################################

@dash_app.callback(
    [
        Output("model_parameters_confirm_btn", "disabled"),
        Output("stored_models", "data"),
        Output("stored_balancing", "data"),
    ],
    [
        Input("model_selection_trees", "value"),
        Input("model_selection_prob", "value"),
        Input("model_selection_neighbors", "value"),
        Input("model_selection_nn", "value"),
        Input("balancing_selection", "value"),
    ],
    [
        State("stored_models", "data"),
        State("stored_balancing", "data"),
    ]
)
def toggle_confirm_button(trees, prob, neighbors, nn, balancing, stored_models, stored_balancing):
    # Combine all selected models
    selected_models = (trees or []) + (prob or []) + (neighbors or []) + (nn or [])
    selected_balancing = balancing or []

    # Use selected values directly so that unchecking all correctly yields an empty list.
    # Previously, the fallback to stored_* meant an empty selection was ignored,
    # preventing the button from ever being disabled after initial load.
    models_data = selected_models
    balancing_data = selected_balancing

    # Button enabled only if both lists are non-empty
    disabled = not (models_data and balancing_data)

    return disabled, models_data, balancing_data