import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, no_update, callback_context, ALL

from app import dash_app

from app.utils.model_comparison_dashboard.chart_builders import (
    delta_badge,
    build_leaderboard_rows,
    build_roc_figure,
    build_pr_figure,
    build_cm_figure,
    build_features_figure,
)

from app.utils.model_comparison_dashboard.constants import (
    MODELS, METRICS, METRIC_LABELS, MODEL_LABELS,
    STRATEGY_LABELS,
    _model_display, _strategy_display,
)

from app.utils.model_comparison_dashboard.data_loaders import (
    latest_results_path,
    load_models_from_results,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML LAYOUT — STEP 4
# ══════════════════════════════════════════════════════════════════════════════

def _metric_header(metric, label):
    return html.Th(
        html.Button(label, id={"type": "sort-btn", "metric": metric}, className="sort-btn"),
        colSpan=2,
        className="metric-header-cell",
    )


html_step_4 = html.Div(
    className="dashboard-root",
    children=[

        # ── Page header ──────────────────────────────────────────────────────
        html.Div(className="dash-page-header", children=[
            html.Div(className="dash-header-left", children=[
                html.Span("STEP 4 · MODEL ANALYSIS", className="dash-title-tag"),
                html.H2("Result Analysis", className="dash-title"),
                html.P(
                    "Comparative performance across baseline and survival-enhanced pipelines.",
                    className="dash-subtitle",
                ),
            ]),
            html.Div(className="dash-header-right", children=[
                html.Div(className="result-toggle-wrap", children=[
                    html.Span("View:", className="toggle-label"),
                    html.Div(className="result-toggle", children=[
                        html.Button("Baseline", id="toggle-baseline", className="toggle-btn"),
                        html.Button("Enhanced", id="toggle-enhanced", className="toggle-btn active-toggle"),
                    ]),
                ]),
            ]),
        ]),

        # ── Global controls ──────────────────────────────────────────────────
        html.Div(className="global-controls", children=[
            html.Button(
                "⊞ Filter",
                id="filter-panel-toggle",
                className="filter-toggle-btn",
            ),
            html.Div(className="controls-right", children=[
                html.Button("↓ Export CSV", id="export-csv-btn", className="export-btn"),
                dcc.Download(id="download-csv"),
            ]),
        ]),

        # ── Filter panel (collapsible) ────────────────────────────────────────
        html.Div(id="filter-panel", className="filter-panel filter-panel-hidden", children=[
            # Model type filter
            html.Div(className="filter-group", children=[
                html.Div(className="filter-group-header", children=[
                    html.Span("Model Type", className="filter-group-title"),
                    html.Button("All",  id="filter-model-all",  className="filter-select-btn"),
                    html.Button("None", id="filter-model-none", className="filter-select-btn"),
                ]),
                html.Div(
                    id="filter-model-checks",
                    className="filter-checks",
                    children=[
                        html.Label(className="filter-check-item", children=[
                            dcc.Checklist(
                                id={"type": "model-filter", "key": k},
                                options=[{"label": "", "value": k}],
                                value=[k],
                                className="filter-checklist",
                            ),
                            html.Span(v, className="filter-check-label"),
                        ])
                        for k, v in MODEL_LABELS.items()
                    ],
                ),
            ]),
            # Strategy filter
            html.Div(className="filter-group", children=[
                html.Div(className="filter-group-header", children=[
                    html.Span("SMOTE Variant", className="filter-group-title"),
                    html.Button("All",  id="filter-strategy-all",  className="filter-select-btn"),
                    html.Button("None", id="filter-strategy-none", className="filter-select-btn"),
                ]),
                html.Div(
                    id="filter-strategy-checks",
                    className="filter-checks",
                    children=[
                        html.Label(className="filter-check-item", children=[
                            dcc.Checklist(
                                id={"type": "strategy-filter", "key": k},
                                options=[{"label": "", "value": k}],
                                value=[k],
                                className="filter-checklist",
                            ),
                            html.Span(v, className="filter-check-label"),
                        ])
                        for k, v in STRATEGY_LABELS.items()
                    ],
                ),
            ]),
        ]),

        # ── Hidden stores ────────────────────────────────────────────────────
        dcc.Store(id="sort-metric-store",       data="f1_macro"),
        dcc.Store(id="sort-dir-store",          data="desc"),
        dcc.Store(id="sort-result-type-store",  data="enhanced"),
        dcc.Store(id="result-type-store",       data="enhanced"),
        dcc.Store(id="selected-model-store",    data=""),
        dcc.Store(id="row-clicks-store",        data={}),
        dcc.Store(id="page-store",              data=0),
        dcc.Store(id="page-size-store",         data=5),
        dcc.Store(id="step4-data-loaded",       data=False),
        dcc.Store(id="features-positive-store", data=True),
        dcc.Store(id="features-scroll-store",   data={"top": 0, "bar_height": 30}),
        # Active filter sets — lists of selected keys; None means "all"
        dcc.Store(id="filter-model-store",    data=list(MODEL_LABELS.keys())),
        dcc.Store(id="filter-strategy-store", data=list(STRATEGY_LABELS.keys())),

        # ── Parameter tooltip (shown on model-row hover) ─────────────────────
        html.Div(id="params-tooltip", className="params-tooltip", children=[
            html.Div(id="params-tooltip-content", className="params-tooltip-content"),
        ]),

        # ── Leaderboard table ────────────────────────────────────────────────
        html.Div(className="leaderboard-wrap", children=[
            html.Div(id="leaderboard-table-container"),
            html.Div(className="pagination-bar", children=[
                html.Span(id="page-indicator", className="page-indicator"),
                html.Div(className="page-btn-group", children=[
                    html.Button("«", id="page-first", className="page-btn", title="First page"),
                    html.Button("‹", id="page-prev",  className="page-btn", title="Previous page"),
                    html.Button("›", id="page-next",  className="page-btn", title="Next page"),
                    html.Button("»", id="page-last",  className="page-btn", title="Last page"),
                ]),
                html.Div(className="page-size-wrap", children=[
                    html.Span("Show", className="page-size-label"),
                    dcc.Dropdown(
                        id="page-size-dropdown",
                        options=[
                            {"label": "5",  "value": 5},
                            {"label": "10", "value": 10},
                            {"label": "20", "value": 20},
                            {"label": "50", "value": 50},
                        ],
                        value=5,
                        clearable=False,
                        searchable=False,
                        className="page-size-dropdown",
                    ),
                    html.Span("per page", className="page-size-label"),
                ]),
            ]),
        ]),

        # ── Chart section ────────────────────────────────────────────────────
        html.Div(className="charts-section", children=[
            html.Div(id="charts-model-label", className="charts-model-label"),
            html.Div(className="charts-grid", children=[
                html.Div(className="chart-card", children=[dcc.Graph(id="chart-roc",      config={"displayModeBar": False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id="chart-pr",       config={"displayModeBar": False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id="chart-cm",       config={"displayModeBar": False})]),
                html.Div(className="chart-card chart-card-features", children=[
                    html.Div(className="features-card-header", children=[
                        html.Span("Selected Features · Importance", id="features-card-title", className="features-card-title"),
                        html.Button(
                            "✦ Top 15",
                            id="features-filter-btn",
                            className="chart-toolbar-btn chart-toolbar-btn-active",
                            title="Toggle: show top 15 features / show all features",
                        ),
                    ]),
                    html.Div(
                        id="features-scroll-wrap",
                        className="features-scroll-wrap",
                        children=[
                            dcc.Graph(id="chart-features", config={"displayModeBar": False}),
                        ],
                    ),
                ]),
            ]),
        ]),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — STEP 4
# ══════════════════════════════════════════════════════════════════════════════

# ── Lazy data loader — fires once when Step 4 first renders ──────────────────
@dash_app.callback(
    Output("step4-data-loaded", "data"),
    Input("step4-data-loaded", "data"),
    prevent_initial_call=False,
)
def load_step4_data(already_loaded):
    """
    Runs when the step4-data-loaded store is first mounted (i.e. when Step 4
    renders). Populates the global MODELS dict from results.db in the latest
    dated folder.  Subsequent renders skip the reload because already_loaded
    is True.
    """
    global MODELS
    if already_loaded:
        return True

    try:
        MODELS.update(load_models_from_results(latest_results_path))
        print(f"[step4] Loaded {len(MODELS)} models from {latest_results_path}")
    except Exception as exc:
        print(f"[step4] WARNING – could not load results.db: {exc}")
        MODELS.clear()

    return True


# ── Set default selection once data is loaded ────────────────────────────────
@dash_app.callback(
    Output("selected-model-store", "data", allow_duplicate=True),
    Input("step4-data-loaded", "data"),
    State("selected-model-store", "data"),
    prevent_initial_call=True,
)
def initialise_selection(loaded, current_selected):
    """Write rank-#1 key to the store exactly once on load, if nothing is
    already selected.  This ensures render_leaderboard, update_charts, and
    the page-indicator all see a real key from the very first render."""
    if not loaded or not MODELS:
        return no_update
    if current_selected:
        return no_update
    rows = build_leaderboard_rows("f1_macro", "enhanced", sort_result_type="enhanced")
    return rows[0]["key"] if rows else no_update


# ── Navigate to selected model's page when result-type toggles ───────────────
@dash_app.callback(
    Output("page-store", "data", allow_duplicate=True),
    Input("result-type-store", "data"),
    State("selected-model-store", "data"),
    State("sort-metric-store", "data"),
    State("sort-dir-store", "data"),
    State("sort-result-type-store", "data"),
    State("filter-model-store", "data"),
    State("filter-strategy-store", "data"),
    State("page-size-store", "data"),
    prevent_initial_call=True,
)
def navigate_to_selected_on_toggle(result_type, selected_key, sort_metric, sort_dir,
                                   sort_result_type, model_filter, strategy_filter, page_size):
    if not selected_key or not MODELS:
        return 0
    page_size        = page_size or 5
    sort_result_type = sort_result_type or "enhanced"
    all_rows = build_leaderboard_rows(
        sort_metric, result_type,
        sort_result_type=sort_result_type,
        model_filter=model_filter,
        strategy_filter=strategy_filter,
    )
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))
    keys = [r["key"] for r in all_rows]
    if selected_key not in keys:
        return 0
    return keys.index(selected_key) // page_size


# ── Result type toggle ────────────────────────────────────────────────────────
@dash_app.callback(
    Output("result-type-store", "data"),
    Output("toggle-baseline", "className"),
    Output("toggle-enhanced", "className"),
    Input("toggle-baseline", "n_clicks"),
    Input("toggle-enhanced", "n_clicks"),
    prevent_initial_call=False,
)
def update_result_type(n_base, n_enh):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".":
        return "enhanced", "toggle-btn", "toggle-btn active-toggle"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "toggle-enhanced":
        return "enhanced", "toggle-btn", "toggle-btn active-toggle"
    return "baseline", "toggle-btn active-toggle", "toggle-btn"


# ── Page size dropdown → store ────────────────────────────────────────────────
@dash_app.callback(
    Output("page-size-store", "data"),
    Output("page-store", "data", allow_duplicate=True),
    Input("page-size-dropdown", "value"),
    State("selected-model-store", "data"),
    State("sort-metric-store", "data"),
    State("sort-dir-store", "data"),
    State("sort-result-type-store", "data"),
    State("result-type-store", "data"),
    State("filter-model-store", "data"),
    State("filter-strategy-store", "data"),
    prevent_initial_call=True,
)
def update_page_size(value, selected_key, sort_metric, sort_dir,
                     sort_result_type, result_type, model_filter, strategy_filter):
    new_size         = value or 5
    sort_result_type = sort_result_type or "enhanced"
    if not selected_key or not MODELS:
        return new_size, 0
    all_rows = build_leaderboard_rows(
        sort_metric, result_type,
        sort_result_type=sort_result_type,
        model_filter=model_filter,
        strategy_filter=strategy_filter,
    )
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))
    keys     = [r["key"] for r in all_rows]
    new_page = keys.index(selected_key) // new_size if selected_key in keys else 0
    return new_size, new_page


# ── Sort metric store — also resets page to 0 ────────────────────────────────
@dash_app.callback(
    Output("sort-metric-store", "data"),
    Output("sort-dir-store", "data"),
    Output("sort-result-type-store", "data"),
    Output("page-store", "data"),
    Output("selected-model-store", "data", allow_duplicate=True),
    Input({"type": "sort-btn", "metric": ALL}, "n_clicks"),
    Input("page-first", "n_clicks"),
    Input("page-prev",  "n_clicks"),
    Input("page-next",  "n_clicks"),
    Input("page-last",  "n_clicks"),
    State("result-type-store", "data"),
    State("sort-metric-store", "data"),
    State("sort-dir-store", "data"),
    State("sort-result-type-store", "data"),
    State("page-store", "data"),
    State("selected-model-store", "data"),
    State("page-size-store", "data"),
    State("filter-model-store", "data"),
    State("filter-strategy-store", "data"),
    prevent_initial_call=True,
)
def update_sort_and_page(sort_btn_clicks, first, prev, nxt, last,
                         result_type,
                         state_metric, state_dir, state_sort_rt,
                         state_page, selected_key, page_size,
                         model_filter, strategy_filter):
    PAGINATION_IDS = {"page-first", "page-prev", "page-next", "page-last"}

    page_size     = page_size or 5
    state_sort_rt = state_sort_rt or "enhanced"
    no_change     = (state_metric, state_dir, state_sort_rt, state_page, no_update)

    ctx = callback_context
    if not ctx.triggered:
        return no_change

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    def _first_row_key(metric, direction, sort_rt):
        rows = build_leaderboard_rows(
            metric, result_type,
            sort_result_type=sort_rt,
            model_filter=model_filter,
            strategy_filter=strategy_filter,
        )
        if direction == "asc":
            rows = list(reversed(rows))
        return rows[0]["key"] if rows else ""

    # ── Pattern-match trigger: sort-btn only, n_clicks > 0 ───────────────────
    try:
        parsed = json.loads(trigger_id)
        ptype  = parsed.get("type")
        if ptype == "sort-btn":
            if not ctx.triggered[0]["value"]:
                return no_change
            metric  = parsed["metric"]
            new_dir = "asc" if (metric == state_metric and state_dir == "desc") else "desc"
            new_srt = result_type
            top_key = _first_row_key(metric, new_dir, new_srt)
            return metric, new_dir, new_srt, 0, top_key
        return no_change
    except (json.JSONDecodeError, KeyError):
        pass

    # ── Pagination buttons ───────────────────────────────────────────────────
    if trigger_id in PAGINATION_IDS:
        total_rows = len(build_leaderboard_rows(
            state_metric, result_type,
            sort_result_type=state_sort_rt,
            model_filter=model_filter,
            strategy_filter=strategy_filter,
        ))
        last_page = max(0, (total_rows - 1) // page_size)
        if trigger_id == "page-first":
            return state_metric, state_dir, state_sort_rt, 0, no_update
        if trigger_id == "page-prev":
            return state_metric, state_dir, state_sort_rt, max(0, state_page - 1), no_update
        if trigger_id == "page-next":
            return state_metric, state_dir, state_sort_rt, min(last_page, state_page + 1), no_update
        if trigger_id == "page-last":
            return state_metric, state_dir, state_sort_rt, last_page, no_update

    return no_change


# ── Leaderboard render ────────────────────────────────────────────────────────
@dash_app.callback(
    Output("leaderboard-table-container", "children"),
    Output("page-indicator", "children"),
    Input("sort-metric-store", "data"),
    Input("sort-dir-store", "data"),
    Input("sort-result-type-store", "data"),
    Input("result-type-store", "data"),
    Input("page-store", "data"),
    Input("step4-data-loaded", "data"),
    Input("page-size-store", "data"),
    Input("selected-model-store", "data"),
    Input("filter-model-store", "data"),
    Input("filter-strategy-store", "data"),
)
def render_leaderboard(sort_metric, sort_dir, sort_result_type, result_type,
                       page, _loaded, page_size, selected_key,
                       model_filter, strategy_filter):
    page_size        = page_size or 5
    sort_result_type = sort_result_type or "enhanced"
    all_rows = build_leaderboard_rows(
        sort_metric, result_type,
        sort_result_type=sort_result_type,
        model_filter=model_filter,
        strategy_filter=strategy_filter,
    )
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))

    total_rows = len(all_rows)
    last_page  = max(0, (total_rows - 1) // page_size) if total_rows else 0
    page       = max(0, min(page, last_page))

    start = page * page_size
    rows  = all_rows[start : start + page_size]

    # Best values computed across ALL rows (not just current page)
    best = {}
    for m in METRICS:
        col  = f"enh_{m}" if result_type == "enhanced" else f"base_{m}"
        vals = [r[col] for r in all_rows]
        best[m] = max(vals) if vals else 0

    is_enhanced = result_type == "enhanced"

    # Primary (selected/highlighted) value is always on the RIGHT:
    #   Baseline → Enh Δ (left)  | Base  (right, highlighted)
    #   Enhanced → Base Δ (left) | Enh   (right, highlighted)
    sub_headers = [
        html.Th("", className="th-rank"),
        html.Th("Model", className="th-model"),
        html.Th("SMOTE Variant", className="th-strategy"),
    ]
    for m in METRICS:
        active = sort_metric == m
        if is_enhanced:
            sub_headers.append(html.Th("Base",   className=f"th-sub {'th-sub-active' if active else ''}"))
            sub_headers.append(html.Th("Enh Δ",  className=f"th-sub enh-col {'th-sub-active' if active else ''}"))
        else:
            sub_headers.append(html.Th("Enh",    className=f"th-sub enh-col {'th-sub-active' if active else ''}"))
            sub_headers.append(html.Th("Base Δ", className=f"th-sub {'th-sub-active' if active else ''}"))

    thead = html.Thead([
        html.Tr([
            html.Th("", className="th-rank"),
            html.Th("", className="th-model"),
            html.Th("", className="th-strategy"),
            *[_metric_header(m, METRIC_LABELS[m]) for m in METRICS],
        ]),
        html.Tr(sub_headers),
    ])

    tbody_rows = []
    for i, row in enumerate(rows):
        global_rank  = all_rows.index(row) + 1
        # When sorting ascending (worst-first), display rank as n, n-1, n-2...
        # so that the worst model shows the highest number and #1 = best.
        display_rank = (total_rows - global_rank + 1) if sort_dir == "asc" else global_rank
        rank_cls     = "gold" if display_rank == 1 else "silver" if display_rank == 2 else "bronze" if display_rank == 3 else "default"
        cells = [
            html.Td(html.Span(f"#{display_rank}", className=f"rank-badge rank-{rank_cls}"), className="td-rank"),
            html.Td(
                html.Div([html.Span(_model_display(row["name"]), className="model-name")]),
                className="td-model",
            ),
            html.Td(
                html.Span(_strategy_display(row["strategy"]), className="strategy-pill"),
                className="td-strategy",
            ),
        ]
        for m in METRICS:
            base_val = row[f"base_{m}"]
            enh_val  = row[f"enh_{m}"]
            primary  = enh_val if is_enhanced else base_val
            is_best  = abs(primary - best[m]) < 0.0001

            if is_enhanced:
                # Left cell: BASE plain value (no delta)
                cells.append(html.Td(
                    html.Span(f"{base_val:.3f}", className="metric-val"),
                    className="td-metric",
                ))
                # Right cell: ENH value + delta vs base (primary, highlighted when best)
                cells.append(html.Td(
                    html.Div([
                        html.Span(f"{enh_val:.3f}", className="metric-val"),
                        delta_badge(enh_val, base_val),
                    ], className="enh-cell-inner"),
                    className=f"td-metric enh-col {'best-cell' if is_best else ''}".strip(),
                ))
            else:
                # Left cell: ENH plain value (no delta)
                cells.append(html.Td(
                    html.Span(f"{enh_val:.3f}", className="metric-val"),
                    className="td-metric enh-col",
                ))
                # Right cell: BASE value + delta vs enh (primary, highlighted when best)
                cells.append(html.Td(
                    html.Div([
                        html.Span(f"{base_val:.3f}", className="metric-val"),
                        delta_badge(base_val, enh_val),
                    ], className="enh-cell-inner"),
                    className=f"td-metric {'best-cell' if is_best else ''}".strip(),
                ))

        params      = row.get("parameters", {})
        params_json = json.dumps(params) if params else ""
        tbody_rows.append(html.Tr(
            cells,
            id={"type": "model-row", "key": row["key"]},
            className=f"model-row {'selected-row' if row['key'] == selected_key else ''}",
            **{"data-params": params_json},
            n_clicks=0,
        ))

    table     = html.Table([thead, html.Tbody(tbody_rows)], className="leaderboard-table")
    indicator = f"Page {page + 1} of {last_page + 1}  ·  {total_rows} model{'s' if total_rows != 1 else ''}"

    return table, indicator


# ── Row click → selected model ────────────────────────────────────────────────
# Uses row-clicks-store to distinguish real user clicks from phantom triggers
# caused by render_leaderboard rebuilding the table and resetting n_clicks=0.
# A click is real only when the current n_clicks for a key is GREATER than the
# last known value for that key in the store.
@dash_app.callback(
    Output("selected-model-store", "data"),
    Output("row-clicks-store", "data"),
    Input({"type": "model-row", "key": ALL}, "n_clicks"),
    State({"type": "model-row", "key": ALL}, "id"),
    State("row-clicks-store", "data"),
    State("selected-model-store", "data"),
    prevent_initial_call=True,
)
def select_model(n_clicks_list, id_list, prev_clicks, current_selected):
    ctx = callback_context
    if not ctx.triggered or not id_list:
        return no_update, no_update

    prev_clicks = prev_clicks or {}

    # Build current snapshot: {key: n_clicks}
    current_clicks = {
        id_obj["key"]: (n or 0)
        for id_obj, n in zip(id_list, n_clicks_list)
    }

    # Find keys where n_clicks strictly increased — these are real clicks
    clicked_key = None
    for key, count in current_clicks.items():
        if count > prev_clicks.get(key, 0):
            clicked_key = key
            break

    # Always update the store snapshot so next call has fresh baseline
    new_store = current_clicks

    if clicked_key is None:
        # No real click — phantom trigger from table re-render; keep selection
        return no_update, new_store

    return clicked_key, new_store


# ── Highlight selected row in DOM directly (no table re-render needed) ────────
# REMOVED — render_leaderboard now has selected-model-store as an Input and
# stamps selected-row in Python on every render. No clientside DOM mutation
# needed, which eliminates the race condition between Python and JS highlighting.


# ── Filter panel toggle ───────────────────────────────────────────────────────
@dash_app.callback(
    Output("filter-panel", "className"),
    Output("filter-panel-toggle", "className"),
    Input("filter-panel-toggle", "n_clicks"),
    State("filter-panel", "className"),
    prevent_initial_call=True,
)
def toggle_filter_panel(n, current_class):
    hidden = "filter-panel-hidden" in current_class
    if hidden:
        return "filter-panel filter-panel-visible", "filter-toggle-btn filter-toggle-active"
    return "filter-panel filter-panel-hidden", "filter-toggle-btn"


# ── Select-All / None for model type ─────────────────────────────────────────
@dash_app.callback(
    Output({"type": "model-filter", "key": ALL}, "value"),
    Input("filter-model-all",  "n_clicks"),
    Input("filter-model-none", "n_clicks"),
    State({"type": "model-filter", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def select_all_none_model(n_all, n_none, id_list):
    ctx = callback_context
    if not ctx.triggered or not id_list:
        return [no_update] * len(id_list)
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "filter-model-all":
        return [[id_obj["key"]] for id_obj in id_list]
    return [[] for _ in id_list]


# ── Select-All / None for strategy ───────────────────────────────────────────
@dash_app.callback(
    Output({"type": "strategy-filter", "key": ALL}, "value"),
    Input("filter-strategy-all",  "n_clicks"),
    Input("filter-strategy-none", "n_clicks"),
    State({"type": "strategy-filter", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def select_all_none_strategy(n_all, n_none, id_list):
    ctx = callback_context
    if not ctx.triggered or not id_list:
        return [no_update] * len(id_list)
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "filter-strategy-all":
        return [[id_obj["key"]] for id_obj in id_list]
    return [[] for _ in id_list]


# ── Aggregate checkbox values → filter stores ─────────────────────────────────
@dash_app.callback(
    Output("filter-model-store", "data"),
    Input({"type": "model-filter", "key": ALL}, "value"),
    State({"type": "model-filter", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def update_model_filter(values, id_list):
    if not id_list:
        return no_update
    selected = [id_obj["key"] for id_obj, val in zip(id_list, values) if val]
    return selected if selected else list(MODEL_LABELS.keys())  # never show 0 rows


@dash_app.callback(
    Output("filter-strategy-store", "data"),
    Input({"type": "strategy-filter", "key": ALL}, "value"),
    State({"type": "strategy-filter", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def update_strategy_filter(values, id_list):
    if not id_list:
        return no_update
    selected = [id_obj["key"] for id_obj, val in zip(id_list, values) if val]
    return selected if selected else list(STRATEGY_LABELS.keys())


# ── Features filter toggle ────────────────────────────────────────────────────
@dash_app.callback(
    Output("features-positive-store", "data"),
    Output("features-filter-btn", "children"),
    Output("features-filter-btn", "className"),
    Input("features-filter-btn", "n_clicks"),
    State("features-positive-store", "data"),
    prevent_initial_call=True,
)
def toggle_features_filter(n, currently_top15):
    new_state = not currently_top15
    if new_state:
        return new_state, "✦ Top 15", "chart-toolbar-btn chart-toolbar-btn-active"
    return new_state, "+ / − Show All", "chart-toolbar-btn"


@dash_app.callback(
    Output("chart-roc",           "figure"),
    Output("chart-pr",            "figure"),
    Output("chart-cm",            "figure"),
    Output("chart-features",      "figure"),
    Output("charts-model-label",  "children"),
    Output("features-card-title", "children"),
    Input("selected-model-store",    "data"),
    Input("result-type-store",       "data"),
    Input("step4-data-loaded",       "data"),
    Input("features-positive-store", "data"),
)
def update_charts(model_key, result_type, _loaded, positives_only):
    if not model_key or model_key not in MODELS:
        empty = go.Figure()
        empty.update_layout(**_base_layout(""))
        return empty, empty, empty, empty, "", "Selected Features · Importance"

    name  = _model_display(MODELS[model_key]["model"])
    label = html.Div([
        html.Span(name, className="charts-model-name"),
        html.Span(f" · {result_type.capitalize()}", className="charts-model-type"),
    ])

    raw_method    = MODELS[model_key][result_type].get("feature_method", "")
    method_label  = raw_method.upper() if raw_method else "Importance"
    features_title = f"Selected Features · {method_label}"

    return (
        build_roc_figure(model_key, result_type),
        build_pr_figure(model_key, result_type),
        build_cm_figure(model_key, result_type),
        build_features_figure(model_key, result_type, top15_only=bool(positives_only)),
        label,
        features_title,
    )


# ── Selected Features: reset scroll to top on figure change ─────────────────
# The chart now auto-sizes its height to fit all bars (no scrollbar needed),
# and the x-axis uses autorange so no dynamic relayout is required.
# This callback only resets the wrapper scroll position when a new model is
# selected so the user always sees the top-ranked feature first.
dash_app.clientside_callback(
    """
    function(figure) {
        if (!figure || !figure.data) return window.dash_clientside.no_update;
        var wrap = document.getElementById('features-scroll-wrap');
        if (wrap) wrap.scrollTop = 0;
        return window.dash_clientside.no_update;
    }
    """,
    Output("features-scroll-store", "data"),
    Input("chart-features", "figure"),
    prevent_initial_call=True,
)


dash_app.clientside_callback(
    """
    function(tableChildren) {
        const tooltip = document.getElementById('params-tooltip');
        const content = document.getElementById('params-tooltip-content');
        if (!tooltip || !content) return window.dash_clientside.no_update;

        // Remove previous listeners by replacing with cloned nodes (avoids stacking)
        const newTooltip = tooltip.cloneNode(true);
        tooltip.parentNode.replaceChild(newTooltip, tooltip);
        const newContent = newTooltip.querySelector('#params-tooltip-content');

        let mouseX = 0, mouseY = 0;

        function positionTooltip() {
            const scrollY = window.scrollY || document.documentElement.scrollTop;
            const scrollX = window.scrollX || document.documentElement.scrollLeft;
            const tipW = newTooltip.offsetWidth  || 220;
            const tipH = newTooltip.offsetHeight || 100;
            const vw   = window.innerWidth;

            let left = mouseX + scrollX + 14;
            let top  = mouseY + scrollY - tipH - 10;

            if (mouseX + 14 + tipW > vw) left = mouseX + scrollX - tipW - 14;
            if (mouseY - tipH - 10 < 0)  top  = mouseY + scrollY + 18;

            newTooltip.style.left = left + 'px';
            newTooltip.style.top  = top  + 'px';
        }

        function formatKey(k) {
            return String(k).replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
        }

        function parseParams(raw) {
            if (!raw) return null;
            try {
                const obj = JSON.parse(raw);
                if (obj && typeof obj === 'object' && !Array.isArray(obj)) return obj;
            } catch(_) {}
            try {
                const jsonified = raw
                    .replace(/'/g, '"')
                    .replace(/True/g, 'true')
                    .replace(/False/g, 'false')
                    .replace(/None/g, 'null');
                const obj = JSON.parse(jsonified);
                if (obj && typeof obj === 'object') return obj;
            } catch(_) {}
            // Python list-of-tuples repr: [('key', val), ...]
            const result = {};
            const pairRe = /[\\(\\[]\\s*['"]([^'"]+)['"]\\s*,\\s*([^,\\)\\]]+?)\\s*[\\)\\]]/g;
            let m;
            while ((m = pairRe.exec(raw)) !== null) {
                const num = parseFloat(m[2].trim());
                result[m[1].trim()] = isNaN(num) ? m[2].trim() : num;
            }
            return Object.keys(result).length ? result : null;
        }

        document.addEventListener('mousemove', function(e) {
            mouseX = e.clientX;
            mouseY = e.clientY;
            if (newTooltip.classList.contains('params-tooltip-visible')) positionTooltip();
        });

        document.addEventListener('mouseover', function(e) {
            const row = e.target.closest('tr.model-row');
            if (!row) { newTooltip.classList.remove('params-tooltip-visible'); return; }

            const raw = row.getAttribute('data-params');
            if (!raw || raw === '{}' || raw === '') {
                newTooltip.classList.remove('params-tooltip-visible');
                return;
            }

            const params = parseParams(raw);
            if (!params || !Object.keys(params).length) {
                newTooltip.classList.remove('params-tooltip-visible');
                return;
            }

            newContent.innerHTML = Object.keys(params).map(k =>
                '<div class="ptt-row">' +
                  '<span class="ptt-key">' + formatKey(k) + '</span>' +
                  '<span class="ptt-val">' + params[k] + '</span>' +
                '</div>'
            ).join('');

            positionTooltip();
            newTooltip.classList.add('params-tooltip-visible');
        });

        document.addEventListener('mouseout', function(e) {
            const row = e.target.closest('tr.model-row');
            if (row && !row.contains(e.relatedTarget)) {
                newTooltip.classList.remove('params-tooltip-visible');
            }
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output("params-tooltip", "className"),
    Input("leaderboard-table-container", "children"),
    prevent_initial_call=True,
)


# ── CSV export ────────────────────────────────────────────────────────────────
@dash_app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("sort-metric-store", "data"),
    State("result-type-store", "data"),
    State("sort-result-type-store", "data"),
    State("filter-model-store", "data"),
    State("filter-strategy-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n, sort_metric, result_type, sort_result_type, model_filter, strategy_filter):
    if not n:
        return no_update
    rows = build_leaderboard_rows(
        sort_metric, result_type,
        sort_result_type=sort_result_type or "enhanced",
        model_filter=model_filter,
        strategy_filter=strategy_filter,
    )
    records = [{
        "Model":    r["name"],
        "Strategy": r["strategy"],
        **{f"Base {METRIC_LABELS[m]}": round(r[f"base_{m}"], 4) for m in METRICS},
        **{f"Enh {METRIC_LABELS[m]}":  round(r[f"enh_{m}"],  4) for m in METRICS},
    } for r in rows]
    return dcc.send_data_frame(pd.DataFrame(records).to_csv, "model_leaderboard.csv", index=False)