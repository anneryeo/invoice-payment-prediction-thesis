# screens/settings.py
#
# Settings screen — saves user preferences to data/user_settings.json.
# Settings are loaded on mount and written on "Save" click.
# audit_logger records every save event.

from __future__ import annotations

import json
import os

from dash import html, dcc, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from src.app import dash_app
from src.app.utils.audit_logger import log_event

_SETTINGS_PATH = os.path.join("data", "user_settings.json")

_BALANCE_OPTIONS = [
    {"label": "SMOTE",            "value": "smote"},
    {"label": "Borderline-SMOTE", "value": "borderline_smote"},
    {"label": "SMOTEENN",         "value": "smoteenn"},
    {"label": "SMOTETomek",       "value": "smotetomek"},
    {"label": "Hybrid",           "value": "hybrid"},
]

_DEFAULTS = {
    "undersample_threshold": 0.5,
    "default_balance_strategy": "smote",
    "late_invoice_cutoff_days": 60,
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_settings() -> dict:
    if os.path.exists(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, encoding="utf-8") as fh:
                saved = json.load(fh)
            return {**_DEFAULTS, **saved}
        except Exception:
            pass
    return dict(_DEFAULTS)


def _save_settings(data: dict) -> None:
    os.makedirs(os.path.dirname(_SETTINGS_PATH) or ".", exist_ok=True)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ── Screen class ──────────────────────────────────────────────────────────────

class SettingsScreen:
    def __init__(self, app):
        self.app = app
        self._register_callbacks()

    def layout(self) -> html.Div:
        cfg = _load_settings()
        return html.Div(
            id="settings-root",
            children=[
                html.Div(className="page-header", children=[
                    html.H2("Settings", className="page-title"),
                    html.P(
                        "Configure system preferences. Changes are saved to disk and "
                        "applied on the next prediction run.",
                        className="page-subtitle",
                    ),
                ]),

                html.Div(className="card", style={"maxWidth": "640px"}, children=[
                    html.Div(className="section-header", children=[
                        html.H4("Prediction Parameters", className="section-title"),
                    ]),
                    html.Div(className="card-body", children=[

                        # Undersample threshold
                        html.Div(style={"marginBottom": "24px"}, children=[
                            html.Label(
                                "Undersample Threshold",
                                htmlFor="settings-threshold-slider",
                                style={"fontWeight": "600", "display": "block",
                                       "marginBottom": "4px", "fontSize": "14px"},
                            ),
                            html.P(
                                "Class balancing threshold for under-sampling the majority class.",
                                style={"fontSize": "12px", "color": "#888", "margin": "0 0 8px"},
                            ),
                            dcc.Slider(
                                id="settings-threshold-slider",
                                min=0, max=1, step=0.05,
                                value=cfg["undersample_threshold"],
                                marks={0: "0", 0.25: "0.25", 0.5: "0.5",
                                       0.75: "0.75", 1: "1"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]),

                        # Default balance strategy
                        html.Div(style={"marginBottom": "24px"}, children=[
                            html.Label(
                                "Default Balance Strategy",
                                htmlFor="settings-balance-dropdown",
                                style={"fontWeight": "600", "display": "block",
                                       "marginBottom": "4px", "fontSize": "14px"},
                            ),
                            html.P(
                                "Over-sampling strategy applied when training new model "
                                "configurations.",
                                style={"fontSize": "12px", "color": "#888", "margin": "0 0 8px"},
                            ),
                            dcc.Dropdown(
                                id="settings-balance-dropdown",
                                options=_BALANCE_OPTIONS,
                                value=cfg["default_balance_strategy"],
                                clearable=False,
                                style={"fontSize": "14px"},
                            ),
                        ]),

                        # Late invoice cutoff
                        html.Div(style={"marginBottom": "28px"}, children=[
                            html.Label(
                                "Late Invoice Cutoff (days)",
                                htmlFor="settings-cutoff-input",
                                style={"fontWeight": "600", "display": "block",
                                       "marginBottom": "4px", "fontSize": "14px"},
                            ),
                            html.P(
                                "Invoices delayed by more than this many days are flagged "
                                "as high-risk in the dashboard.",
                                style={"fontSize": "12px", "color": "#888", "margin": "0 0 8px"},
                            ),
                            dcc.Input(
                                id="settings-cutoff-input",
                                type="number",
                                value=cfg["late_invoice_cutoff_days"],
                                min=1, max=180, step=1,
                                debounce=True,
                                style={
                                    "width": "100px", "padding": "8px 12px",
                                    "border": "1px solid #d1ccc2",
                                    "borderRadius": "6px", "fontSize": "14px",
                                    "fontFamily": "-apple-system, 'Segoe UI', sans-serif",
                                },
                            ),
                        ]),

                        html.Div(style={"display": "flex", "gap": "12px",
                                        "alignItems": "center"}, children=[
                            html.Button(
                                "Save Settings",
                                id="settings-save-btn",
                                className="btn btn-primary",
                                n_clicks=0,
                            ),
                            html.Div(id="settings-status-msg",
                                     style={"fontSize": "13px", "color": "#2d6a4f"}),
                        ]),
                    ]),
                ]),
            ],
        )

    def _register_callbacks(self):
        @dash_app.callback(
            Output("settings-status-msg", "children"),
            Input("settings-save-btn", "n_clicks"),
            State("settings-threshold-slider", "value"),
            State("settings-balance-dropdown", "value"),
            State("settings-cutoff-input",     "value"),
            prevent_initial_call=True,
        )
        def _save(n_clicks, threshold, strategy, cutoff):
            if not n_clicks:
                raise PreventUpdate
            try:
                data = {
                    "undersample_threshold":     threshold,
                    "default_balance_strategy":  strategy,
                    "late_invoice_cutoff_days":  cutoff,
                }
                _save_settings(data)
                log_event(
                    "settings_saved",
                    f"threshold={threshold}, strategy={strategy}, cutoff={cutoff}",
                )
                return "✓ Settings saved."
            except Exception as exc:
                return f"⚠ Save failed: {exc}"
