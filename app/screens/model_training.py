from dash import html, dcc

class ModelTrainingScreen:
    def __init__(self, app):
        self.app = app

    def layout(self):
        return html.Div([
            html.H2("Model Training"),
            
            html.Label("Select Models"),
            dcc.Checklist(
                id="model_selection",
                options=[
                    {"label": "AdaBoost", "value": "ada_boost"},
                    {"label": "Decision Tree", "value": "decision_tree"},
                    {"label": "Gaussian Naive Bayes", "value": "gaussian_naive_bayes"},
                    {"label": "KNN", "value": "knn"},
                    {"label": "Random Forest", "value": "random_forest"},
                    {"label": "XGBoost", "value": "xgboost"},
                    {"label": "NN-MLP", "value": "nn_mlp"}
                ],
                value=[]
            ),
            
            html.Label("Select Balancing Strategy"),
            dcc.Checklist(
                id="balance_strategy",
                options=[
                    {"label": "SMOTE", "value": "smote"},
                    {"label": "Borderline-SMOTE", "value": "borderline_smote"},
                    {"label": "SMOTEENN", "value": "smoteenn"},
                    {"label": "SMOTETomek", "value": "smotetomek"}
                ],
                value=[]
            ),
            
            html.Button("Train Models", id="train_btn"),
            html.Div(id="training_progress"),
            
            html.H3("Model Summary"),
            dcc.Dropdown(id="model_summary_dropdown", placeholder="Select a model to view details"),
            html.Div(id="model_summary_table"),
            dcc.Graph(id="auc_graph")
        ])