# screens/invoice_drilldown.py
#
# Invoice Prediction Drilldown
# Builds the invoice feature DataFrame via CreditSalesProcessor, runs the
# deployed InferencePipeline, and displays per-invoice predictions with
# confidence scores.
# Supports toggleable bracket filtering, column sorting, CSV export, and
# an expected cash flow projection chart.

from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from src.app import dash_app
from src.utils.data_loaders.read_settings_json import read_settings_json
from src.app.utils.app_cache import load_credit_sales, load_predictions, CashFlowCache, BRACKET_DELAY
from src.modules.machine_learning.utils.inference.inference_pipeline import (
    load_inference_pipeline,
    _BRACKET_LABELS,
    _INTERNAL_KEYS,
)

_logger = logging.getLogger(__name__)

_BRACKET_OPTIONS = [{"label": "All Brackets", "value": "all"}] + [
    {"label": v, "value": k} for k, v in _BRACKET_LABELS.items()
]
_ALL_BRACKET_KEYS = ["on_time", "30_days", "60_days", "90_days"]

_cashflow_cache = CashFlowCache()


# ── Component builders ────────────────────────────────────────────────────────

def _filter_bar() -> html.Div:
    return html.Div(id="drilldown-filter-bar", className="bracket-filter-bar", children=[
        html.Div(className="flex-center gap-1 flex-wrap", children=[
            html.Button("Select All", id="drilldown-filter-all-btn",
                        className="filter-chip filter-chip-all active",
                        n_clicks=0),
            html.Button("On Time",     id="drilldown-filter-on-time",  n_clicks=0,
                        className="filter-chip filter-chip-on-time active"),
            html.Button("1–30 Days",   id="drilldown-filter-30",       n_clicks=0,
                        className="filter-chip filter-chip-30-days active"),
            html.Button("31–60 Days",  id="drilldown-filter-60",       n_clicks=0,
                        className="filter-chip filter-chip-60-days active"),
            html.Button("61+ Days",    id="drilldown-filter-90",       n_clicks=0,
                        className="filter-chip filter-chip-90-days active"),
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
        style_cell_conditional=[
            {
                "if": {"column_id": "invoice_no"},
                "minWidth": "72px",
                "width":    "72px",
                "maxWidth": "90px",
                "paddingRight": "6px",
                "textAlign": "center",
            },
        ],
        style_header_conditional=[
            {
                "if": {"column_id": col},
                "cursor": "pointer",
            }
            for col in ["invoice_no", "due_date", "amount", "actual", "pred_conf"]
        ],
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


_KPI_VARIANT_CLASS = {"primary": "", "success": "mini-kpi-card-success", "danger": "mini-kpi-card-danger"}


def _kpi_card(label: str, value: float, sub: str = "", variant: str = "primary") -> html.Div:
    card_class = " ".join(c for c in ("mini-kpi-card", _KPI_VARIANT_CLASS[variant]) if c)
    return html.Div(className=card_class, children=[
        html.P(label, className="mini-kpi-label"),
        html.P(f"₱{value:,.2f}", className="mini-kpi-value"),
        html.P(sub, className="mini-kpi-sub") if sub else None,
    ])


def _build_load_outputs(all_rows: list[dict] | None):
    """
    Shared body for the mount-load and refresh callbacks: derives every
    output those callbacks need from the (already loaded/cached) full
    invoice prediction set — table rows, page count, the store payload,
    and the per-bracket chip count labels.
    """
    empty_labels = ("On Time · 0", "1–30 Days · 0", "31–60 Days · 0", "61+ Days · 0")

    if not all_rows:
        return [], 0, None, *empty_labels

    try:
        PAGE_SIZE = 15
        visible = [{k: v for k, v in r.items() if k not in _INTERNAL_KEYS}
                   for r in all_rows[:PAGE_SIZE]]
        page_count = (len(all_rows) + PAGE_SIZE - 1) // PAGE_SIZE

        pred_counts = Counter(r["_pred_key"] for r in all_rows)
        _total      = len(all_rows) or 1

        def _fmt(key: str, label: str) -> str:
            n = pred_counts.get(key, 0)
            return f"{label} · {n:,} ({n / _total:.0%})"

        label_on_time = _fmt("on_time", "On Time")
        label_30      = _fmt("30_days", "1–30 Days")
        label_60      = _fmt("60_days", "31–60 Days")
        label_90      = _fmt("90_days", "61+ Days")

        return (visible, page_count, all_rows,
                label_on_time, label_30, label_60, label_90)

    except Exception:
        _logger.exception("Invoice drilldown prediction failed")
        return [], 0, None, *empty_labels


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
                dcc.Store(id="drilldown-layer2-meta-store"),
                dcc.Store(id="drilldown-cashflow-months-store"),

                html.Div(className="page-header", children=[
                    html.H2("Invoice Prediction", className="page-title"),
                    html.P(
                        "Per-invoice payment delay predictions using the deployed model.",
                        className="page-subtitle",
                    ),
                ]),

                # Table card
                html.Div(className="card", children=[
                    html.Div(className="section-header", children=[
                        html.H4("Invoice Results", className="section-title"),
                        # Filter chips + export controls
                        html.Div(className="flex-center gap-1 flex-wrap", children=[
                            _filter_bar(),
                            html.Button(
                                "Refresh",
                                id="drilldown-refresh-btn",
                                className="btn btn-outline",
                                n_clicks=0,
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
                        className="p-0",
                        children=[_table()],
                    ),
                ]),

                # Cash flow card
                html.Div(className="card mt-2", children=[
                    html.Div(className="section-header", children=[
                        html.H4("Expected Cash Flow", className="section-title"),
                        html.P(
                            "Projected receivables by expected payment month, based on predicted delay brackets.",
                            className="section-subtext",
                        ),
                    ]),
                    html.Div(
                        id="drilldown-cashflow-wrapper",
                        className="p-2",
                        children=[
                            html.Div(id="drilldown-cashflow-kpi-wrapper"),
                            html.Div(
                                id="drilldown-cashflow-chart-wrapper",
                                children=[html.Div("Loading cash flow…", className="loading-state")],
                            ),
                        ],
                    ),
                ]),
            ],
        )

    def _register_callbacks(self):
        # ── Load on mount ─────────────────────────────────────────────────────
        @dash_app.callback(
            Output("drilldown-table",             "data"),
            Output("drilldown-table",             "page_count"),
            Output("drilldown-predictions-store", "data"),
            Output("drilldown-filter-on-time",    "children"),
            Output("drilldown-filter-30",         "children"),
            Output("drilldown-filter-60",         "children"),
            Output("drilldown-filter-90",         "children"),
            Output("drilldown-layer2-meta-store", "data"),
            Input("drilldown-mount-interval",     "n_intervals"),
            prevent_initial_call=True,
        )
        def _initial_load(_n):
            pipeline = _load_pipeline()
            settings = read_settings_json()
            df, layer1_meta = load_credit_sales(settings)
            all_rows, layer2_meta = load_predictions(pipeline, df, layer1_meta)
            return *_build_load_outputs(all_rows), layer2_meta

        # ── Manual refresh — bypasses the cache to pick up new data/model ───────
        @dash_app.callback(
            Output("drilldown-table",             "data",       allow_duplicate=True),
            Output("drilldown-table",             "page_count", allow_duplicate=True),
            Output("drilldown-predictions-store", "data",       allow_duplicate=True),
            Output("drilldown-filter-on-time",    "children",   allow_duplicate=True),
            Output("drilldown-filter-30",         "children",   allow_duplicate=True),
            Output("drilldown-filter-60",         "children",   allow_duplicate=True),
            Output("drilldown-filter-90",         "children",   allow_duplicate=True),
            Output("drilldown-layer2-meta-store", "data",       allow_duplicate=True),
            Input("drilldown-refresh-btn",        "n_clicks"),
            prevent_initial_call=True,
        )
        def _refresh(n_clicks):
            if not n_clicks:
                raise PreventUpdate
            _logger.info("[AppCache] Manual refresh: bypassing both cache layers and recomputing.")
            pipeline = _load_pipeline()
            settings = read_settings_json()
            df, layer1_meta = load_credit_sales(settings, force_recompute=True)
            all_rows, layer2_meta = load_predictions(pipeline, df, layer1_meta, force_recompute=True)
            return *_build_load_outputs(all_rows), layer2_meta

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
            Output("drilldown-cashflow-kpi-wrapper",   "children"),
            Output("drilldown-cashflow-chart-wrapper", "children"),
            Output("drilldown-cashflow-months-store",  "data"),
            Input("drilldown-predictions-store", "data"),
            Input("drilldown-active-brackets",   "data"),
            State("drilldown-layer2-meta-store", "data"),
            prevent_initial_call=True,
        )
        def _build_cashflow(all_rows, active_brackets, layer2_meta):
            _no_data = html.Div("No data available.", className="loading-state")
            _no_amt  = html.Div(
                "No invoices with amount data available for the cash flow chart.",
                className="loading-state",
            )
            if not all_rows:
                return html.Div(), _no_data, []

            active_set = set(active_brackets if active_brackets is not None else _ALL_BRACKET_KEYS)

            TODAY = pd.Timestamp.today().normalize()

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

            cache_result = _cashflow_cache.load(all_rows, layer2_meta or {})
            if cache_result is None:
                return html.Div(), _no_amt, []

            full_month_data, skipped = cache_result

            # Filter to active brackets only
            month_data = {}
            for month, brackets in full_month_data.items():
                filtered_brackets = {k: v for k, v in brackets.items() if k in active_set}
                if filtered_brackets:
                    month_data[month] = filtered_brackets

            if not month_data:
                return html.Div(), _no_amt, []

            sorted_months = sorted(month_data.keys())
            month_dates   = [pd.to_datetime(m + "-01") for m in sorted_months]

            today_month = TODAY.strftime("%Y-%m")
            today_date  = pd.to_datetime(today_month + "-01")
            today_str   = today_date.strftime("%Y-%m-%d")

            traces = []
            for key in _ALL_BRACKET_KEYS:
                if key not in active_set:
                    continue
                y_vals = [month_data[m].get(key, 0) for m in sorted_months]
                traces.append(go.Bar(
                    name        = BRACKET_DISPLAY[key],
                    x           = month_dates,
                    y           = y_vals,
                    marker_color= BRACKET_COLORS_HEX[key],
                    hovertemplate=(
                        f"<b>{BRACKET_DISPLAY[key]}</b><br>"
                        "%{x|%b %Y}<br>"
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
                x           = month_dates,
                y           = cumulative,
                mode        = "lines+markers",
                yaxis       = "y2",
                line        = dict(color="#1b3a6b", width=2, dash="dot"),
                marker      = dict(size=5, color="#1b3a6b"),
                hovertemplate=(
                    "<b>Cumulative</b><br>"
                    "%{x|%b %Y}<br>"
                    "Total: ₱%{y:,.2f}<extra></extra>"
                ),
            ))

            fig = go.Figure(data=traces)

            if today_date in month_dates:
                fig.add_shape(
                    type  = "line",
                    x0    = today_str, x1 = today_str,
                    y0    = 0, y1 = 1,
                    xref  = "x", yref = "paper",
                    line  = dict(width=2, dash="dash", color="#6b7280"),
                )
                fig.add_annotation(
                    x = today_str, y = 1,
                    xref = "x", yref = "paper",
                    yanchor = "bottom",
                    text = "Today",
                    showarrow = False,
                    font = dict(size=11, color="#6b7280"),
                )

            fig.update_layout(
                barmode       = "stack",
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                margin        = dict(l=60, r=40, t=50, b=60),
                height        = 440,
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
                    type        = "date",
                    tickformat  = "%b %Y",
                    dtick       = "M1",
                    rangeslider = dict(visible=False),
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

            # KPI amounts computed from the active-bracket subset only
            chart_rows = [r for r in all_rows if r.get("_pred_key") in active_set]
            total_amount = sum(
                r.get("_amount_raw", 0) or 0
                for r in chart_rows
                if r.get("_amount_raw") is not None
            )
            overdue_keys   = {"30_days", "60_days", "90_days"}
            overdue_amount = sum(
                r.get("_amount_raw", 0) or 0
                for r in chart_rows
                if r.get("_pred_key") in overdue_keys and r.get("_amount_raw") is not None
            )
            ontime_amount  = total_amount - overdue_amount
            pct_overdue    = (overdue_amount / total_amount * 100) if total_amount else 0

            kpi_row = html.Div(className="mini-kpi-row", children=[
                _kpi_card("Total Receivables", total_amount,  f"{len(chart_rows):,} invoices", "primary"),
                _kpi_card("Expected On Time",  ontime_amount, f"{100 - pct_overdue:.1f}% of total", "success"),
                _kpi_card("At Risk (Delayed)", overdue_amount, f"{pct_overdue:.1f}% of total", "danger"),
            ])

            chart_div = html.Div([
                dcc.Graph(
                    id     = "drilldown-cashflow-graph",
                    figure = fig,
                    config = {"displayModeBar": False, "responsive": True},
                    style  = {"width": "100%"},
                ),
                html.P(
                    f"* Chart excludes {skipped:,} invoice(s) with missing amount data. "
                    f"Expected dates: On Time = due date; 1–30 Days = due + 30d; "
                    f"31–60 Days = due + 60d; 61+ Days = due + 91d. "
                    f"Today's marker is placed at {TODAY.strftime('%B %Y')}.",
                    className="chart-footnote",
                ),
            ])
            return kpi_row, chart_div, sorted_months

        # ── KPI update on chart zoom / pan ───────────────────────────────────
        # Plotly handles axis zoom client-side (relayoutData fires but the
        # server never sees the new axis range unless we hook into it here).
        # This callback re-aggregates KPI amounts for only the months visible
        # in the current zoomed x-axis window without touching the chart itself.
        @dash_app.callback(
            Output("drilldown-cashflow-kpi-wrapper", "children", allow_duplicate=True),
            Input("drilldown-cashflow-graph",        "relayoutData"),
            State("drilldown-predictions-store",     "data"),
            State("drilldown-active-brackets",       "data"),
            State("drilldown-cashflow-months-store", "data"),
            prevent_initial_call=True,
        )
        def _zoom_filter_kpis(relayout_data, all_rows, active_brackets, sorted_months):
            if not relayout_data or not all_rows or not sorted_months:
                raise PreventUpdate

            active_set = set(
                active_brackets if active_brackets is not None else _ALL_BRACKET_KEYS
            )

            # Determine which months are currently in view
            x0 = relayout_data.get("xaxis.range[0]")
            x1 = relayout_data.get("xaxis.range[1]")
            if x0 is None:
                # Plotly sometimes packs the range as an array
                rng = relayout_data.get("xaxis.range")
                if rng and len(rng) == 2:
                    x0, x1 = rng

            if relayout_data.get("xaxis.autorange"):
                # User double-clicked to zoom out — show all months
                selected_month_keys = set(sorted_months)
            elif x0 is not None and x1 is not None:
                # The x-axis is type "date", so Plotly sends ISO date strings
                # (e.g. "2025-01-15 00:00:00") for the range boundaries.
                # Parse them and include every month whose 1st falls inside the window.
                try:
                    dt0 = pd.to_datetime(x0)
                    dt1 = pd.to_datetime(x1)
                except Exception:
                    raise PreventUpdate
                selected_month_keys = {
                    mk for mk in sorted_months
                    if dt0 <= pd.to_datetime(mk + "-01") <= dt1
                }
            else:
                raise PreventUpdate  # resize or unrelated layout event

            # Re-aggregate amounts for the visible months only
            total_amount  = 0.0
            overdue_amount = 0.0
            row_count     = 0
            overdue_keys  = {"30_days", "60_days", "90_days"}

            for r in all_rows:
                pred_key = r.get("_pred_key", "on_time")
                if pred_key not in active_set:
                    continue
                amount_raw = r.get("_amount_raw")
                if amount_raw is None:
                    continue
                try:
                    amount_val = float(amount_raw)
                except (TypeError, ValueError):
                    continue
                due_str = r.get("due_date", "—")
                try:
                    due_dt = pd.to_datetime(due_str)
                except Exception:
                    continue
                if pd.isna(due_dt):
                    continue
                delay_days    = BRACKET_DELAY.get(pred_key, 0)
                expected_date = due_dt + pd.Timedelta(days=delay_days)
                if expected_date.strftime("%Y-%m") not in selected_month_keys:
                    continue
                total_amount  += amount_val
                if pred_key in overdue_keys:
                    overdue_amount += amount_val
                row_count += 1

            ontime_amount = total_amount - overdue_amount
            pct_overdue   = (overdue_amount / total_amount * 100) if total_amount else 0

            return html.Div(className="mini-kpi-row", children=[
                _kpi_card("Total Receivables", total_amount,  f"{row_count:,} invoices",       "primary"),
                _kpi_card("Expected On Time",  ontime_amount, f"{100 - pct_overdue:.1f}% of total", "success"),
                _kpi_card("At Risk (Delayed)", overdue_amount, f"{pct_overdue:.1f}% of total",  "danger"),
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

        # ── Legend click → sync drilldown-active-brackets (clientside) ──────
        # Plotly handles legend clicks client-side (no Dash callback fires).
        # This callback reads restyleData, maps the toggled trace index back to
        # a bracket key, and writes the new active set so the KPI cards and
        # chip styles update in lock-step with what the chart is showing.
        dash_app.clientside_callback(
            """
            function(restyleData, activeBrackets) {
                if (!restyleData) return window.dash_clientside.no_update;
                var props   = restyleData[0];
                var indices = restyleData[1];
                if (!('visible' in props)) return window.dash_clientside.no_update;

                var ALL_KEYS = ["on_time", "30_days", "60_days", "90_days"];
                var currentActive = activeBrackets ? activeBrackets.slice() : ALL_KEYS.slice();

                // Bar traces are added in ALL_KEYS order, skipping inactive ones;
                // Cumulative is always the last trace and has no bracket key.
                var traceKeyMap = {};
                var traceIdx = 0;
                ALL_KEYS.forEach(function(key) {
                    if (currentActive.indexOf(key) !== -1) {
                        traceKeyMap[traceIdx] = key;
                        traceIdx++;
                    }
                });

                var newActive = currentActive.slice();
                indices.forEach(function(ti, i) {
                    var key = traceKeyMap[ti];
                    if (!key) return;  // Cumulative trace — ignore
                    var vis = Array.isArray(props.visible) ? props.visible[i] : props.visible;
                    var nowVisible = (vis === true || vis === undefined || vis === null);
                    var pos = newActive.indexOf(key);
                    if (nowVisible && pos === -1) {
                        newActive.push(key);
                    } else if (!nowVisible && pos !== -1) {
                        newActive.splice(pos, 1);
                    }
                });

                return newActive;
            }
            """,
            Output("drilldown-active-brackets", "data", allow_duplicate=True),
            Input("drilldown-cashflow-graph",   "restyleData"),
            State("drilldown-active-brackets",  "data"),
            prevent_initial_call=True,
        )

        # ── Header click-to-sort (clientside) ────────────────────────────────
        dash_app.clientside_callback(
            """
            function(id) {
                const table = document.getElementById('drilldown-table');
                if (!table) return window.dash_clientside.no_update;
                table.querySelectorAll('th.dash-header').forEach(th => {
                    if (th.dataset.sortBound) return;
                    th.dataset.sortBound = '1';
                    th.style.cursor = 'pointer';
                    th.addEventListener('click', function(e) {
                        // The DataTable's own sort arrow already handles clicks on
                        // the arrow span; this catches clicks on the label text.
                        const arrow = th.querySelector('.column-header--sort');
                        if (arrow && !arrow.contains(e.target)) {
                            arrow.click();
                        }
                    });
                });
                return window.dash_clientside.no_update;
            }
            """,
            Output("drilldown-table", "id"),   # dummy output — side effect only
            Input("drilldown-table",  "data"), # re-bind after each data update
        )


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_pipeline() -> object | None:
    """Load the deployed InferencePipeline artifact."""
    try:
        settings = read_settings_json()
        deployed_dir = settings.get("Training", {}).get("DEPLOYED_MODELS", "")
        return load_inference_pipeline(deployed_dir)
    except Exception:
        return None
