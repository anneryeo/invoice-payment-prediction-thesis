# screens/invoice_drilldown.py
#
# Invoice Prediction Drilldown
# Loads the processed invoice cache, runs the deployed InferencePipeline,
# and displays per-invoice predictions with confidence scores.
# Supports filtering by predicted bracket and CSV export.

from __future__ import annotations

import io
import os
import glob
import pickle

import pandas as pd
from dash import html, dcc, dash_table, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from src.app import dash_app
from src.utils.data_loaders.read_settings_json import read_settings_json
from src.app.utils.audit_logger import log_event

# Bracket display
_BRACKET_LABELS = {
    "on_time": "On Time",
    "30_days": "1–30 Days",
    "60_days": "31–60 Days",
    "90_days": "61+ Days",
}
_BRACKET_COLORS = {
    "on_time": "#2d6a4f",
    "30_days": "#d97706",
    "60_days": "#c2410c",
    "90_days": "#b91c1c",
}
_BRACKET_OPTIONS = [{"label": "All Brackets", "value": "all"}] + [
    {"label": v, "value": k} for k, v in _BRACKET_LABELS.items()
]

# Columns that are metadata / target — not passed to the model.
# Used only as a fallback when the pipeline's fitted scaler doesn't expose
# feature_names_in_ (e.g. a legacy artifact); the normal path below derives
# the real feature list from the pipeline itself.
_NON_FEATURE_COLS = {"dtp_bracket", "due_date", "date_fully_paid",
                     "censor", "days_elapsed_until_fully_paid"}
# Columns we want to show as display info when present
_DISPLAY_COLS     = ["due_date", "gross_receivables", "net_receivables",
                     "school_year", "category_name", "student_id_pseudonimized"]


def _select_feature_columns(df_full: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Build the raw feature matrix the pipeline expects.

    Prefers the fitted scaler's feature_names_in_ (the actual columns seen
    at training time — see DataPreparer.normalize()) over the hand-maintained
    _NON_FEATURE_COLS exclusion list, since the latter can silently drift out
    of sync with what the model was trained on.
    """
    scaler   = getattr(pipeline, "scaler", None)
    expected = list(getattr(scaler, "feature_names_in_", []))
    if expected:
        missing = [c for c in expected if c not in df_full.columns]
        if missing:
            raise ValueError(
                f"Invoice cache is missing {len(missing)} column(s) required "
                f"by the deployed model: {missing}"
            )
        return df_full[expected].copy()

    feature_cols = [c for c in df_full.columns if c not in _NON_FEATURE_COLS]
    return df_full[feature_cols].copy()


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_pipeline() -> object | None:
    """Load the deployed InferencePipeline pickle."""
    try:
        settings = read_settings_json()
        deployed_dir = settings.get("Training", {}).get("DEPLOYED_MODELS", "")
        candidates = [
            p for p in glob.glob(os.path.join(deployed_dir, "finalized_*.pkl"))
            if not p.endswith("finalized_survival_model.pkl")
        ]
        if not candidates:
            return None
        with open(candidates[0], "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _load_cache() -> pd.DataFrame | None:
    """Load the credit_sales_cache.pkl feature DataFrame."""
    try:
        settings   = read_settings_json()
        cache_path = os.path.join(
            settings.get("Training", {}).get("RESULTS_ROOT", ""),
            "credit_sales_cache.pkl",
        )
        if not os.path.exists(cache_path):
            return None
        return pd.read_pickle(cache_path)
    except Exception:
        return None


def _run_predictions(
    df_full: pd.DataFrame,
    pipeline,
    page: int = 1,
    page_size: int = 15,
    bracket_filter: str = "all",
) -> tuple[list[dict], int]:
    """
    Run inference on the full cache DataFrame.

    Returns (table_rows, total_count) for the requested page.
    Rows include display columns + predicted bracket + top confidence %.
    """
    # Separate feature columns from metadata
    X_raw = _select_feature_columns(df_full, pipeline)

    labels = pipeline.predict(X_raw)
    probas = pipeline.predict_proba(X_raw)

    # Build display rows
    rows = []
    for i in range(len(df_full)):
        pred    = labels[i]
        conf    = float(probas.iloc[i][pred]) * 100
        actual  = df_full["dtp_bracket"].iloc[i] if "dtp_bracket" in df_full.columns else "—"
        due     = df_full["due_date"].iloc[i]     if "due_date"    in df_full.columns else "—"
        amount  = (
            df_full["net_receivables"].iloc[i]
            if "net_receivables" in df_full.columns
            else (df_full["gross_receivables"].iloc[i]
                  if "gross_receivables" in df_full.columns
                  else "—")
        )

        # Format
        try:
            due_str = pd.to_datetime(due).strftime("%Y-%m-%d") if pd.notna(due) else "—"
        except Exception:
            due_str = str(due)
        try:
            amt_str = f"₱{float(amount):,.2f}" if pd.notna(amount) else "—"
        except Exception:
            amt_str = str(amount)

        rows.append({
            "invoice_no":  i + 1,
            "due_date":    due_str,
            "amount":      amt_str,
            "prediction":  _BRACKET_LABELS.get(pred, pred),
            "confidence":  f"{conf:.1f}%",
            "actual":      _BRACKET_LABELS.get(actual, actual),
            "_pred_key":   pred,
        })

    # Filter
    if bracket_filter != "all":
        rows = [r for r in rows if r["_pred_key"] == bracket_filter]

    total   = len(rows)
    start   = (page - 1) * page_size
    visible = [{k: v for k, v in r.items() if k != "_pred_key"}
               for r in rows[start : start + page_size]]
    return visible, total


# ── Component builders ────────────────────────────────────────────────────────

def _empty(icon: str, title: str, body: str = "") -> html.Div:
    return html.Div(className="empty-state", children=[
        html.Span(icon,  className="empty-state-icon"),
        html.P(title,    className="empty-state-title"),
        html.P(body,     className="empty-state-text") if body else None,
    ])


def _summary_badges(labels: list[str]) -> html.Div:
    from collections import Counter
    counts = Counter(labels)
    total  = len(labels) or 1
    chips  = []
    for key in ["on_time", "30_days", "60_days", "90_days"]:
        n     = counts.get(key, 0)
        label = _BRACKET_LABELS[key]
        color = _BRACKET_COLORS[key]
        chips.append(html.Span(
            f"{label}: {n:,} ({n/total:.0%})",
            style={
                "backgroundColor": f"{color}18",
                "color":           color,
                "border":          f"1px solid {color}40",
                "borderRadius":    "20px",
                "padding":         "4px 12px",
                "fontSize":        "12px",
                "fontWeight":      "600",
                "marginRight":     "8px",
                "display":         "inline-block",
            },
        ))
    return html.Div(chips, style={"marginTop": "8px", "marginBottom": "4px"})


def _table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="drilldown-table",
        columns=[
            {"name": "#",             "id": "invoice_no"},
            {"name": "Due Date",      "id": "due_date"},
            {"name": "Amount",        "id": "amount"},
            {"name": "Prediction",    "id": "prediction"},
            {"name": "Confidence",    "id": "confidence"},
            {"name": "Actual Bracket","id": "actual"},
        ],
        data=[],
        page_current=0,
        page_size=15,
        page_action="custom",
        sort_action="custom",
        sort_mode="single",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1b3a6b", "color": "white",
            "fontWeight":      "600",     "fontSize": "11px",
            "textTransform":   "uppercase", "letterSpacing": "0.05em",
            "padding":         "10px 14px", "border": "none",
        },
        style_cell={
            "textAlign": "left",  "padding": "9px 14px",
            "fontSize":  "13px",  "color": "#2b2b2b",
            "border":    "1px solid #e5e1d8",
            "fontFamily": "-apple-system, 'Segoe UI', sans-serif",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafaf8"},
        ],
    )


# ── Screen class ──────────────────────────────────────────────────────────────

class InvoiceDrilldownScreen:
    def __init__(self, app):
        self.app = app
        self._register_callbacks()

    def layout(self) -> html.Div:
        return html.Div(
            id="drilldown-root",
            children=[
                dcc.Interval(id="drilldown-mount-interval",
                             interval=300, n_intervals=0, max_intervals=1),

                # Store predictions for CSV export
                dcc.Store(id="drilldown-predictions-store"),

                html.Div(className="page-header", children=[
                    html.H2("Invoice Prediction", className="page-title"),
                    html.P(
                        "Per-invoice payment delay predictions using the deployed model.",
                        className="page-subtitle",
                    ),
                ]),

                # Summary row + controls
                html.Div(className="card", style={"marginBottom": "16px"}, children=[
                    html.Div(className="section-header", children=[
                        html.H4("Prediction Summary", className="section-title"),
                    ]),
                    html.Div(id="drilldown-summary",
                             className="card-body",
                             children=[html.Div("Loading…", className="loading-state")]),
                ]),

                # Table card
                html.Div(className="card", children=[
                    html.Div(className="section-header", children=[
                        html.H4("Invoice Results", className="section-title"),
                        # Filter + export controls
                        html.Div(style={"display": "flex", "gap": "8px",
                                        "alignItems": "center"}, children=[
                            dcc.Dropdown(
                                id="drilldown-filter",
                                options=_BRACKET_OPTIONS,
                                value="all",
                                clearable=False,
                                style={"width": "180px", "fontSize": "13px"},
                            ),
                            html.Button(
                                "Export CSV",
                                id="drilldown-export-btn",
                                className="btn btn-outline",
                                n_clicks=0,
                            ),
                            dcc.Download(id="drilldown-download"),
                        ]),
                    ]),
                    html.Div(
                        id="drilldown-table-wrapper",
                        style={"padding": "0"},
                        children=[_table()],
                    ),
                ]),
            ],
        )

    def _register_callbacks(self):
        # ── Load on mount ─────────────────────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-summary",           "children"),
            Output("drilldown-table",             "data"),
            Output("drilldown-table",             "page_count"),
            Output("drilldown-predictions-store", "data"),
            Input("drilldown-mount-interval",     "n_intervals"),
            prevent_initial_call=True,
        )
        def _initial_load(_n):
            pipeline = _load_pipeline()
            df       = _load_cache()

            if pipeline is None:
                msg = _empty("🤖", "No deployed model found.",
                             "Complete the Setup Wizard to train and finalize a model.")
                return msg, [], 0, None
            if df is None:
                msg = _empty("📂", "Invoice data not found.",
                             "Complete the Setup Wizard to process the revenue dataset.")
                return msg, [], 0, None

            try:
                X_raw  = _select_feature_columns(df, pipeline)
                labels = pipeline.predict(X_raw)
                probas = pipeline.predict_proba(X_raw)

                # Build all rows for the store (no pagination here)
                all_rows = []
                for i in range(len(df)):
                    pred   = labels[i]
                    conf   = float(probas.iloc[i][pred]) * 100
                    actual = df["dtp_bracket"].iloc[i] if "dtp_bracket" in df.columns else "—"
                    due    = df["due_date"].iloc[i]     if "due_date"    in df.columns else "—"
                    amount = (
                        df["net_receivables"].iloc[i]
                        if "net_receivables" in df.columns
                        else df["gross_receivables"].iloc[i]
                        if "gross_receivables" in df.columns
                        else None
                    )
                    try:
                        due_str = pd.to_datetime(due).strftime("%Y-%m-%d") if pd.notna(due) else "—"
                    except Exception:
                        due_str = str(due)
                    try:
                        amt_str = f"₱{float(amount):,.2f}" if amount is not None and pd.notna(amount) else "—"
                    except Exception:
                        amt_str = str(amount)

                    all_rows.append({
                        "invoice_no": i + 1,
                        "due_date":   due_str,
                        "amount":     amt_str,
                        "prediction": _BRACKET_LABELS.get(pred, pred),
                        "confidence": f"{conf:.1f}%",
                        "actual":     _BRACKET_LABELS.get(str(actual), str(actual)),
                        "_pred_key":  pred,
                    })

                log_event(
                    "prediction_run",
                    f"model={getattr(pipeline, 'model_key', 'unknown')}, "
                    f"n_invoices={len(df)}",
                )
                summary = _summary_badges(list(labels))
                PAGE_SIZE = 15
                visible = [{k: v for k, v in r.items() if k != "_pred_key"}
                           for r in all_rows[:PAGE_SIZE]]
                page_count = (len(all_rows) + PAGE_SIZE - 1) // PAGE_SIZE
                return summary, visible, page_count, all_rows

            except Exception as exc:
                return (
                    html.Div(f"⚠️ Prediction error: {exc}",
                             className="alert alert-error"),
                    [], 0, None,
                )

        # ── Pagination + filter ───────────────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-table", "data",       allow_duplicate=True),
            Output("drilldown-table", "page_count", allow_duplicate=True),
            Input("drilldown-table",  "page_current"),
            Input("drilldown-filter", "value"),
            State("drilldown-predictions-store", "data"),
            prevent_initial_call=True,
        )
        def _page_filter(page_current, bracket_filter, all_rows):
            if not all_rows:
                raise PreventUpdate
            PAGE_SIZE = 15
            page = (page_current or 0) + 1

            filtered = all_rows
            if bracket_filter and bracket_filter != "all":
                filtered = [r for r in all_rows if r.get("_pred_key") == bracket_filter]

            total      = len(filtered)
            start      = (page - 1) * PAGE_SIZE
            visible    = [{k: v for k, v in r.items() if k != "_pred_key"}
                          for r in filtered[start : start + PAGE_SIZE]]
            page_count = (total + PAGE_SIZE - 1) // PAGE_SIZE
            return visible, page_count

        # ── CSV export ────────────────────────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-download", "data"),
            Input("drilldown-export-btn", "n_clicks"),
            State("drilldown-predictions-store", "data"),
            State("drilldown-filter", "value"),
            prevent_initial_call=True,
        )
        def _export(n_clicks, all_rows, bracket_filter):
            if not n_clicks or not all_rows:
                raise PreventUpdate
            rows = all_rows
            if bracket_filter and bracket_filter != "all":
                rows = [r for r in all_rows if r.get("_pred_key") == bracket_filter]
            export_rows = [{k: v for k, v in r.items() if k != "_pred_key"}
                           for r in rows]
            df_export = pd.DataFrame(export_rows)
            return dcc.send_data_frame(
                df_export.to_csv,
                filename="invoice_predictions.csv",
                index=False,
            )
