# screens/invoice_drilldown.py
#
# Invoice Prediction Drilldown
# Loads the processed invoice cache, runs the deployed InferencePipeline,
# and displays per-invoice predictions with confidence scores.
# Supports toggleable bracket filtering, column sorting, CSV export, and
# an expected cash flow projection chart.

from __future__ import annotations

import io
import os
import glob
import logging
import pickle
import re
from collections import Counter, defaultdict

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from src.app import dash_app
from src.utils.data_loaders.read_settings_json import read_settings_json
from src.app.utils.audit_logger import log_event

_logger = logging.getLogger(__name__)

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
_ALL_BRACKET_KEYS = ["on_time", "30_days", "60_days", "90_days"]

# Internal keys that live in the predictions store but must never reach the
# visible DataTable rows or CSV export.
_INTERNAL_KEYS = {"_pred_key", "_amount_raw", "prediction", "confidence"}

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


def _extract_amount(df_full: pd.DataFrame, i: int) -> float | None:
    """
    Pull the display amount for row *i*, preferring net_receivables over
    gross_receivables. Values may be stored as object dtype or contain
    edge cases that make a plain pd.notna() check fail, so each candidate
    is coerced through float() and validated with pd.isna() instead.
    """
    for col in ("net_receivables", "gross_receivables"):
        if col in df_full.columns:
            val = df_full[col].iloc[i]
            try:
                fval = float(val)
                if not pd.isna(fval):
                    return fval
            except (TypeError, ValueError):
                continue
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

        amount_raw = _extract_amount(df_full, i)

        # Format
        try:
            due_str = pd.to_datetime(due).strftime("%Y-%m-%d") if pd.notna(due) else "—"
        except Exception:
            due_str = str(due)
        amt_str = f"₱{amount_raw:,.2f}" if amount_raw is not None else "—"

        rows.append({
            "invoice_no":  i + 1,
            "due_date":    due_str,
            "amount":      amt_str,
            "pred_conf":   f"{_BRACKET_LABELS.get(pred, pred)} ({conf:.1f}%)",
            "actual":      _BRACKET_LABELS.get(actual, actual),
            "_pred_key":   pred,
            "_amount_raw": amount_raw,
        })

    # Filter
    if bracket_filter != "all":
        rows = [r for r in rows if r["_pred_key"] == bracket_filter]

    total   = len(rows)
    start   = (page - 1) * page_size
    visible = [{k: v for k, v in r.items() if k not in _INTERNAL_KEYS}
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


def _chip_style(color: str, active: bool) -> dict:
    base = {
        "display":       "inline-flex",
        "alignItems":    "center",
        "gap":           "6px",
        "padding":       "5px 14px",
        "borderRadius":  "20px",
        "fontSize":      "12px",
        "fontWeight":    "600",
        "cursor":        "pointer",
        "border":        f"1.5px solid {color}",
        "transition":    "all 0.15s ease",
        "fontFamily":    "-apple-system, 'Segoe UI', sans-serif",
        "letterSpacing": "0.01em",
    }
    if active:
        base.update({"backgroundColor": color, "color": "white"})
    else:
        base.update({"backgroundColor": f"{color}12", "color": color})
    return base


def _filter_bar() -> html.Div:
    return html.Div(id="drilldown-filter-bar", className="bracket-filter-bar", children=[
        html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center",
                        "flexWrap": "wrap"}, children=[
            html.Button("Select All", id="drilldown-filter-all-btn",
                        className="filter-chip filter-chip-all active",
                        style=_chip_style("#1b3a6b", True),
                        n_clicks=0),
            html.Button("On Time",     id="drilldown-filter-on-time",  n_clicks=0,
                        className="filter-chip", style=_chip_style("#2d6a4f", True)),
            html.Button("1–30 Days",   id="drilldown-filter-30",       n_clicks=0,
                        className="filter-chip", style=_chip_style("#d97706", True)),
            html.Button("31–60 Days",  id="drilldown-filter-60",       n_clicks=0,
                        className="filter-chip", style=_chip_style("#c2410c", True)),
            html.Button("61+ Days",    id="drilldown-filter-90",       n_clicks=0,
                        className="filter-chip", style=_chip_style("#b91c1c", True)),
        ])
    ])


def _table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="drilldown-table",
        columns=[
            {"name": "#",                       "id": "invoice_no",  "type": "numeric"},
            {"name": "Due Date",                "id": "due_date"},
            {"name": "Amount Due",              "id": "amount"},
            {"name": "Actual Bracket",          "id": "actual"},
            {"name": "Prediction (Confidence)", "id": "pred_conf"},
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

            # Prediction cell color coding by bracket label prefix
            {
                "if": {"filter_query": '{pred_conf} contains "On Time"', "column_id": "pred_conf"},
                "color": "#2d6a4f", "fontWeight": "600",
                "backgroundColor": "#2d6a4f18",
            },
            {
                "if": {"filter_query": '{pred_conf} contains "1–30"', "column_id": "pred_conf"},
                "color": "#d97706", "fontWeight": "600",
                "backgroundColor": "#d9770618",
            },
            {
                "if": {"filter_query": '{pred_conf} contains "31–60"', "column_id": "pred_conf"},
                "color": "#c2410c", "fontWeight": "600",
                "backgroundColor": "#c2410c18",
            },
            {
                "if": {"filter_query": '{pred_conf} contains "61+"', "column_id": "pred_conf"},
                "color": "#b91c1c", "fontWeight": "600",
                "backgroundColor": "#b91c1c18",
            },
            # Also color-code the Actual Bracket column
            {
                "if": {"filter_query": '{actual} = "On Time"',   "column_id": "actual"},
                "color": "#2d6a4f", "fontWeight": "500",
            },
            {
                "if": {"filter_query": '{actual} = "1–30 Days"', "column_id": "actual"},
                "color": "#d97706", "fontWeight": "500",
            },
            {
                "if": {"filter_query": '{actual} = "31–60 Days"',"column_id": "actual"},
                "color": "#c2410c", "fontWeight": "500",
            },
            {
                "if": {"filter_query": '{actual} = "61+ Days"',  "column_id": "actual"},
                "color": "#b91c1c", "fontWeight": "500",
            },
        ],
    )


def _kpi_card(label: str, value: float, sub: str = "", color: str = "#1b3a6b") -> html.Div:
    return html.Div(style={
        "background": "#f9f8f6", "borderRadius": "10px",
        "padding": "12px 20px", "minWidth": "160px", "flex": "1",
        "borderLeft": f"4px solid {color}",
    }, children=[
        html.P(label, style={"fontSize": "11px", "color": "#6b7280", "margin": "0 0 4px"}),
        html.P(f"₱{value:,.2f}", style={"fontSize": "18px", "fontWeight": "700",
                                         "color": "#1b3a6b", "margin": "0"}),
        html.P(sub, style={"fontSize": "11px", "color": "#6b7280", "margin": "4px 0 0"}) if sub else None,
    ])


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

                # Store predictions for CSV export / filtering / charting
                dcc.Store(id="drilldown-predictions-store"),
                dcc.Store(id="drilldown-active-brackets", data=list(_ALL_BRACKET_KEYS)),

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
                        # Filter chips + export controls
                        html.Div(style={"display": "flex", "gap": "8px",
                                        "alignItems": "center", "flexWrap": "wrap"}, children=[
                            _filter_bar(),
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

                # Cash flow card
                html.Div(className="card", style={"marginTop": "16px"}, children=[
                    html.Div(className="section-header", children=[
                        html.H4("Expected Cash Flow", className="section-title"),
                        html.P(
                            "Projected receivables by expected payment month, based on predicted delay brackets.",
                            style={"fontSize": "12px", "color": "#6b7280", "margin": "0"},
                        ),
                    ]),
                    html.Div(
                        id="drilldown-cashflow-wrapper",
                        style={"padding": "16px"},
                        children=[html.Div("Loading cash flow…", className="loading-state")],
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
            Output("drilldown-filter-on-time",    "children"),
            Output("drilldown-filter-30",         "children"),
            Output("drilldown-filter-60",         "children"),
            Output("drilldown-filter-90",         "children"),
            Input("drilldown-mount-interval",     "n_intervals"),
            prevent_initial_call=True,
        )
        def _initial_load(_n):
            pipeline = _load_pipeline()
            df       = _load_cache()

            empty_labels = ("On Time · 0", "1–30 Days · 0", "31–60 Days · 0", "61+ Days · 0")

            if pipeline is None:
                msg = _empty("🤖", "No deployed model found.",
                             "Complete the Setup Wizard to train and finalize a model.")
                return msg, [], 0, None, *empty_labels
            if df is None:
                msg = _empty("📂", "Invoice data not found.",
                             "Complete the Setup Wizard to process the revenue dataset.")
                return msg, [], 0, None, *empty_labels

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

                    amount_raw = _extract_amount(df, i)

                    try:
                        due_str = pd.to_datetime(due).strftime("%Y-%m-%d") if pd.notna(due) else "—"
                    except Exception:
                        due_str = str(due)
                    amt_str = f"₱{amount_raw:,.2f}" if amount_raw is not None else "—"

                    all_rows.append({
                        "invoice_no":  i + 1,
                        "due_date":    due_str,
                        "amount":      amt_str,
                        "pred_conf":   f"{_BRACKET_LABELS.get(pred, pred)} ({conf:.1f}%)",
                        "actual":      _BRACKET_LABELS.get(str(actual), str(actual)),
                        "_pred_key":   pred,
                        "_amount_raw": amount_raw,
                    })

                log_event(
                    "prediction_run",
                    f"model={getattr(pipeline, 'model_key', 'unknown')}, "
                    f"n_invoices={len(df)}",
                )
                summary = _summary_badges(list(labels))
                PAGE_SIZE = 15
                visible = [{k: v for k, v in r.items() if k not in _INTERNAL_KEYS}
                           for r in all_rows[:PAGE_SIZE]]
                page_count = (len(all_rows) + PAGE_SIZE - 1) // PAGE_SIZE

                pred_counts   = Counter(r["_pred_key"] for r in all_rows)
                label_on_time = f"On Time · {pred_counts.get('on_time', 0):,}"
                label_30      = f"1–30 Days · {pred_counts.get('30_days', 0):,}"
                label_60      = f"31–60 Days · {pred_counts.get('60_days', 0):,}"
                label_90      = f"61+ Days · {pred_counts.get('90_days', 0):,}"

                return (summary, visible, page_count, all_rows,
                        label_on_time, label_30, label_60, label_90)

            except Exception as exc:
                _logger.exception("Invoice drilldown prediction failed")
                return (
                    html.Div(f"⚠️ Prediction error: {exc}",
                             className="alert alert-error"),
                    [], 0, None, *empty_labels,
                )

        # ── Bracket filter toggle logic ──────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-active-brackets", "data"),
            Input("drilldown-filter-all-btn",  "n_clicks"),
            Input("drilldown-filter-on-time",  "n_clicks"),
            Input("drilldown-filter-30",       "n_clicks"),
            Input("drilldown-filter-60",       "n_clicks"),
            Input("drilldown-filter-90",       "n_clicks"),
            State("drilldown-active-brackets", "data"),
            prevent_initial_call=True,
        )
        def _toggle_bracket_filter(n_all, n_ot, n_30, n_60, n_90, active):
            triggered = callback_context.triggered_id
            btn_to_key = {
                "drilldown-filter-on-time": "on_time",
                "drilldown-filter-30":      "30_days",
                "drilldown-filter-60":      "60_days",
                "drilldown-filter-90":      "90_days",
            }
            if triggered == "drilldown-filter-all-btn":
                return [] if set(active) == set(_ALL_BRACKET_KEYS) else list(_ALL_BRACKET_KEYS)
            key = btn_to_key.get(triggered)
            if key:
                new = list(active)
                if key in new:
                    new.remove(key)
                else:
                    new.append(key)
                return new
            return active

        # ── Chip visual state (className) ────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-filter-all-btn", "className"),
            Output("drilldown-filter-on-time", "className"),
            Output("drilldown-filter-30",      "className"),
            Output("drilldown-filter-60",      "className"),
            Output("drilldown-filter-90",      "className"),
            Input("drilldown-active-brackets", "data"),
        )
        def _update_chip_classes(active):
            active_set = set(active or [])
            all_active = (
                "filter-chip filter-chip-all active"
                if active_set == set(_ALL_BRACKET_KEYS)
                else "filter-chip filter-chip-all"
            )

            def chip(key):
                base = f"filter-chip filter-chip-{key.replace('_', '-')}"
                return f"{base} active" if key in active_set else base

            return all_active, chip("on_time"), chip("30_days"), chip("60_days"), chip("90_days")

        # ── Chip visual state (inline style) ─────────────────────────────────
        @dash_app.callback(
            Output("drilldown-filter-all-btn", "style"),
            Output("drilldown-filter-on-time", "style"),
            Output("drilldown-filter-30",      "style"),
            Output("drilldown-filter-60",      "style"),
            Output("drilldown-filter-90",      "style"),
            Input("drilldown-active-brackets", "data"),
        )
        def _update_chip_styles(active):
            active_set = set(active or [])
            all_active = active_set == set(_ALL_BRACKET_KEYS)
            return (
                _chip_style("#1b3a6b", all_active),
                _chip_style("#2d6a4f", "on_time" in active_set),
                _chip_style("#d97706", "30_days" in active_set),
                _chip_style("#c2410c", "60_days" in active_set),
                _chip_style("#b91c1c", "90_days" in active_set),
            )

        # ── Pagination + filter + sort ────────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-table", "data",       allow_duplicate=True),
            Output("drilldown-table", "page_count", allow_duplicate=True),
            Input("drilldown-table",  "page_current"),
            Input("drilldown-table",  "sort_by"),
            Input("drilldown-active-brackets", "data"),
            State("drilldown-predictions-store", "data"),
            prevent_initial_call=True,
        )
        def _page_filter_sort(page_current, sort_by, active_brackets, all_rows):
            if not all_rows:
                raise PreventUpdate
            PAGE_SIZE = 15
            page = (page_current or 0) + 1

            # Filter
            filtered = all_rows
            if active_brackets is not None and len(active_brackets) < 4:
                filtered = [r for r in all_rows if r.get("_pred_key") in active_brackets]

            # Sort
            if sort_by and len(sort_by):
                col = sort_by[0]["column_id"]
                asc = sort_by[0]["direction"] == "asc"

                def sort_key(r):
                    val = r.get(col, "")
                    if col == "invoice_no":
                        try:
                            return (0, float(val))
                        except (TypeError, ValueError):
                            return (1, str(val))
                    if col == "amount":
                        try:
                            return (0, float(str(val).replace("₱", "").replace(",", "")))
                        except (TypeError, ValueError):
                            return (1, str(val))
                    if col == "pred_conf":
                        m = re.search(r"\((\d+\.?\d*)%\)", str(val))
                        if m:
                            try:
                                return (0, float(m.group(1)))
                            except (TypeError, ValueError):
                                return (1, str(val))
                        return (1, str(val))
                    return (0, str(val).lower())

                filtered = sorted(filtered, key=sort_key, reverse=not asc)

            total      = len(filtered)
            start      = (page - 1) * PAGE_SIZE
            visible    = [{k: v for k, v in r.items() if k not in _INTERNAL_KEYS}
                          for r in filtered[start : start + PAGE_SIZE]]
            page_count = (total + PAGE_SIZE - 1) // PAGE_SIZE
            return visible, page_count

        # ── Cash flow chart ───────────────────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-cashflow-wrapper", "children"),
            Input("drilldown-predictions-store", "data"),
            prevent_initial_call=True,
        )
        def _build_cashflow(all_rows):
            if not all_rows:
                return html.Div("No data available.", className="loading-state")

            TODAY = pd.Timestamp.today().normalize()

            BRACKET_DELAY = {
                "on_time": 0,
                "30_days": 30,
                "60_days": 60,
                "90_days": 91,
            }
            BRACKET_COLORS_HEX = {
                "on_time": "#2d6a4f",
                "30_days": "#d97706",
                "60_days": "#c2410c",
                "90_days": "#b91c1c",
            }
            BRACKET_DISPLAY = {
                "on_time": "On Time",
                "30_days": "1–30 Days",
                "60_days": "31–60 Days",
                "90_days": "61+ Days",
            }

            # month_data[month_label][bracket_key] = total_amount
            month_data = defaultdict(lambda: defaultdict(float))
            skipped = 0

            for r in all_rows:
                pred_key   = r.get("_pred_key", "on_time")
                amount_raw = r.get("_amount_raw")
                due_str    = r.get("due_date", "—")

                if amount_raw is None:
                    skipped += 1
                    continue
                try:
                    amount_val = float(amount_raw)
                except (TypeError, ValueError):
                    skipped += 1
                    continue

                try:
                    due_dt = pd.to_datetime(due_str)
                except Exception:
                    skipped += 1
                    continue
                if pd.isna(due_dt):
                    skipped += 1
                    continue

                delay_days    = BRACKET_DELAY.get(pred_key, 0)
                expected_date = due_dt + pd.Timedelta(days=delay_days)
                month_key     = expected_date.strftime("%Y-%m")
                month_data[month_key][pred_key] += amount_val

            if not month_data:
                return html.Div(
                    "No invoices with amount data available for the cash flow chart.",
                    className="loading-state",
                )

            sorted_months = sorted(month_data.keys())
            month_labels  = [pd.to_datetime(m + "-01").strftime("%b %Y") for m in sorted_months]

            today_month   = TODAY.strftime("%Y-%m")
            today_x_label = pd.to_datetime(today_month + "-01").strftime("%b %Y")

            traces = []
            for key in _ALL_BRACKET_KEYS:
                y_vals = [month_data[m].get(key, 0) for m in sorted_months]
                traces.append(go.Bar(
                    name        = BRACKET_DISPLAY[key],
                    x           = month_labels,
                    y           = y_vals,
                    marker_color= BRACKET_COLORS_HEX[key],
                    hovertemplate=(
                        f"<b>{BRACKET_DISPLAY[key]}</b><br>"
                        "%{x}<br>"
                        "Expected: ₱%{y:,.2f}<extra></extra>"
                    ),
                ))

            totals = [sum(month_data[m].values()) for m in sorted_months]
            cumulative = []
            running = 0.0
            for t in totals:
                running += t
                cumulative.append(running)

            traces.append(go.Scatter(
                name        = "Cumulative",
                x           = month_labels,
                y           = cumulative,
                mode        = "lines+markers",
                yaxis       = "y2",
                line        = dict(color="#1b3a6b", width=2, dash="dot"),
                marker      = dict(size=5, color="#1b3a6b"),
                hovertemplate=(
                    "<b>Cumulative</b><br>"
                    "%{x}<br>"
                    "Total: ₱%{y:,.2f}<extra></extra>"
                ),
            ))

            fig = go.Figure(data=traces)

            if today_x_label in month_labels:
                fig.add_vline(
                    x           = today_x_label,
                    line_width  = 2,
                    line_dash   = "dash",
                    line_color  = "#6b7280",
                    annotation_text = "Today",
                    annotation_position = "top",
                    annotation_font_size = 11,
                    annotation_font_color = "#6b7280",
                )

            fig.update_layout(
                barmode       = "stack",
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                margin        = dict(l=60, r=40, t=20, b=60),
                height        = 340,
                legend        = dict(
                    orientation = "h",
                    yanchor     = "bottom",
                    y           = 1.02,
                    xanchor     = "left",
                    x           = 0,
                    font        = dict(size=11),
                ),
                xaxis = dict(
                    title       = "Expected Payment Month",
                    tickfont    = dict(size=11),
                    showgrid    = False,
                    linecolor   = "#e5e1d8",
                ),
                yaxis = dict(
                    title       = "Amount (₱)",
                    tickfont    = dict(size=11),
                    gridcolor   = "#f0ede6",
                    tickformat  = ",.0f",
                    showgrid    = True,
                ),
                yaxis2 = dict(
                    title       = "Cumulative (₱)",
                    overlaying  = "y",
                    side        = "right",
                    tickfont    = dict(size=11),
                    showgrid    = False,
                    tickformat  = ",.0f",
                ),
                font = dict(family="-apple-system, 'Segoe UI', sans-serif", size=12),
            )

            total_amount = sum(
                r.get("_amount_raw", 0) or 0
                for r in all_rows
                if r.get("_amount_raw") is not None
            )
            overdue_keys   = {"30_days", "60_days", "90_days"}
            overdue_amount = sum(
                r.get("_amount_raw", 0) or 0
                for r in all_rows
                if r.get("_pred_key") in overdue_keys and r.get("_amount_raw") is not None
            )
            ontime_amount  = total_amount - overdue_amount
            pct_overdue    = (overdue_amount / total_amount * 100) if total_amount else 0

            kpi_row = html.Div(style={
                "display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap"
            }, children=[
                _kpi_card("Total Receivables", total_amount,  f"{len(all_rows):,} invoices", "#1b3a6b"),
                _kpi_card("Expected On Time",  ontime_amount, f"{100 - pct_overdue:.1f}% of total", "#2d6a4f"),
                _kpi_card("At Risk (Delayed)", overdue_amount, f"{pct_overdue:.1f}% of total", "#b91c1c"),
            ])

            return html.Div([
                kpi_row,
                dcc.Graph(
                    figure = fig,
                    config = {"displayModeBar": False, "responsive": True},
                    style  = {"width": "100%"},
                ),
                html.P(
                    f"* Chart excludes {skipped:,} invoice(s) with missing amount data. "
                    f"Expected dates: On Time = due date; 1–30 Days = due + 30d; "
                    f"31–60 Days = due + 60d; 61+ Days = due + 91d. "
                    f"Today's marker is placed at {TODAY.strftime('%B %Y')}.",
                    style={"fontSize": "11px", "color": "#9ca3af", "marginTop": "8px",
                           "fontStyle": "italic"},
                ),
            ])

        # ── CSV export ────────────────────────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-download", "data"),
            Input("drilldown-export-btn", "n_clicks"),
            State("drilldown-predictions-store", "data"),
            State("drilldown-active-brackets",   "data"),
            prevent_initial_call=True,
        )
        def _export(n_clicks, all_rows, active_brackets):
            if not n_clicks or not all_rows:
                raise PreventUpdate
            rows = all_rows
            if active_brackets is not None and len(active_brackets) < 4:
                rows = [r for r in all_rows if r.get("_pred_key") in active_brackets]
            export_rows = [{k: v for k, v in r.items() if k not in _INTERNAL_KEYS}
                           for r in rows]
            df_export = pd.DataFrame(export_rows)
            return dcc.send_data_frame(
                df_export.to_csv,
                filename="invoice_predictions.csv",
                index=False,
            )


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
