from dash import html, dcc

class SettingsScreen:
    def __init__(self, app):
        self.app = app

    def layout(self):
        return html.Div([
            html.H2("Settings"),
            html.Label("Undersample Threshold"),
            dcc.Slider(id="threshold_slider", min=0, max=1, step=0.05, value=0.5),
            
            html.Label("Default Balance Strategy"),
            dcc.Dropdown(
                id="balance_strategy",
                options=[
                    {"label": "SMOTE", "value": "smote"},
                    {"label": "Borderline-SMOTE", "value": "borderline_smote"},
                    {"label": "SMOTEENN", "value": "smoteenn"},
                    {"label": "SMOTETomek", "value": "smotetomek"},
                    {"label": "Hybrid", "value": "hybrid"}
                ],
                value="smote"
            ),
            
            html.Label("Late Invoice Cutoff (days)"),
            dcc.Input(id="cutoff_days", type="number", value=60),
            
            html.Button("Save Settings", id="save_settings_btn")
        ])