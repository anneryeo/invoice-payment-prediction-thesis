from dash import html, dcc

class ModelComparisonScreen:
    def __init__(self, app):
        self.app = app

    def layout(self):
        return html.Div([
            html.H2("Model Comparison"),
            dcc.Dropdown(
                id="kpi_priority",
                options=[
                    {"label": "Accuracy", "value": "accuracy"},
                    {"label": "Recall", "value": "recall"},
                    {"label": "Precision", "value": "precision"},
                    {"label": "F1 Score", "value": "f1"}
                ],
                value="accuracy"
            ),
            dcc.Graph(id="comparison_chart")
        ])