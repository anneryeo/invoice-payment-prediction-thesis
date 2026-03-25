import json

import pandas as pd
import plotly.graph_objects as go

from dash import Input, Output, State, html, dcc, no_update, callback_context, ALL

from app import dash_app

from app.screens.comparative_model_dashboard_template.utils.chart_builders import (
    delta_badge,
    build_leaderboard_rows,
    build_roc_figure,
    build_pr_figure,
    build_cm_figure,
    build_features_figure,
    _base_layout,
)
from app.screens.comparative_model_dashboard_template.constants import (
    MODELS, METRICS, METRIC_LABELS, MODEL_LABELS,
    _model_display, _strategy_display, set_class_labels, get_active_repo,
)


# ── Comparison-mode config ─────────────────────────────────────────────────────
# id  →  (button label,  left sub-header,  right sub-header)
COMP_MODES = {
    "base_vs_enh":          ("Base vs Enhanced",       "Base",     "Enh Δ"),
    "regular_vs_ordinal":   ("Regular vs Ordinal",      "Regular",  "Ordinal Δ"),
    "regular_vs_twostage":  ("Regular vs Two-Stage",    "Regular",  "2-Stage Δ"),
    "nosmote_vs_smote":     ("No SMOTE vs Best SMOTE",  "No SMOTE", "SMOTE Δ"),
}
COMP_MODE_DEFAULT = "base_vs_enh"

# Maps each comp mode → (left toggle label, right toggle label).
# Used to relabel the "View:" toggle and the charts area header.
VIEW_LABELS = {
    "base_vs_enh":          ("Baseline",  "Enhanced"),
    "regular_vs_ordinal":   ("Regular",   "Ordinal"),
    "regular_vs_twostage":  ("Regular",   "Two-Stage"),
    "nosmote_vs_smote":     ("No SMOTE",  "Best SMOTE"),
}


def _resolve_chart_key(selected_key: str, result_type: str, comp_mode: str) -> tuple:
    """
    Determine which (model_key, data_side) to pass to the chart builders
    based on the active comparison mode and which View toggle is active.

    result_type is always "baseline" (left/reference side of the comparison)
    or "enhanced" (right/comparison side).  The mapping to actual data:

    base_vs_enh
        "baseline" → selected model, baseline pipeline data
        "enhanced" → selected model, enhanced pipeline data

    regular_vs_ordinal
        "baseline" → selected model (regular), enhanced pipeline data
        "enhanced" → best ordinal counterpart of same algorithm, enhanced data
                     Falls back to selected model if no ordinal counterpart.

    regular_vs_twostage
        "baseline" → selected model (regular), enhanced pipeline data
        "enhanced" → best two-stage model (highest enhanced F1 globally)

    nosmote_vs_smote
        "baseline" → selected model at balance=none, enhanced pipeline data
        "enhanced" → best SMOTE variant of same (model, params), enhanced data
    """
    comp_mode = comp_mode or COMP_MODE_DEFAULT

    # base_vs_enh is a straight pass-through
    if comp_mode == "base_vs_enh":
        return selected_key, result_type

    # Left side of all other modes = selected model's enhanced data
    if result_type == "baseline":
        return selected_key, "enhanced"

    # ── Right side: find the counterpart model key ────────────────────────────
    m_data = MODELS.get(selected_key, {})

    if comp_mode == "regular_vs_ordinal":
        base_algo   = m_data.get("model", "")
        target_name = f"ordinal_{base_algo}"
        best_key, best_f1 = None, -1.0
        for key, md in MODELS.items():
            if md["model"] == target_name:
                f1 = md["enhanced"]["evaluation"]["metrics"].get("f1_macro") or 0.0
                if f1 > best_f1:
                    best_f1, best_key = f1, key
        return (best_key, "enhanced") if best_key else (selected_key, "enhanced")

    elif comp_mode == "regular_vs_twostage":
        best_key, best_f1 = None, -1.0
        for key, md in MODELS.items():
            if md["model"].startswith("two_stage_"):
                f1 = md["enhanced"]["evaluation"]["metrics"].get("f1_macro") or 0.0
                if f1 > best_f1:
                    best_f1, best_key = f1, key
        return (best_key, "enhanced") if best_key else (selected_key, "enhanced")

    elif comp_mode == "nosmote_vs_smote":
        model_name = m_data.get("model", "")
        params_sig = json.dumps(m_data.get("parameters", {}), sort_keys=True)
        SMOTE_STRATS = {"smote", "borderline_smote", "smote_enn", "smote_tomek"}
        best_key, best_f1 = None, -1.0
        for key, md in MODELS.items():
            if (md["model"] == model_name
                    and md["balance_strategy"] in SMOTE_STRATS
                    and json.dumps(md.get("parameters", {}), sort_keys=True) == params_sig):
                f1 = md["enhanced"]["evaluation"]["metrics"].get("f1_macro") or 0.0
                if f1 > best_f1:
                    best_f1, best_key = f1, key
        return (best_key, "enhanced") if best_key else (selected_key, "enhanced")

    return selected_key, "enhanced"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _metric_header(metric, label):
    return html.Th(
        html.Button(label, id={"type": "sort-btn", "metric": metric}, className="sort-btn"),
        colSpan=2,
        className="metric-header-cell",
    )


def _filter_rows_for_comp_mode(rows, comp_mode):
    """
    Remove rows that shouldn't appear on the left side of a given comparison.

    nosmote_vs_smote        → keep only balance_strategy == 'none' rows.
                              SMOTE rows are lookup targets, not displayed rows.
    regular_vs_ordinal /
    regular_vs_twostage     → keep only regular (non-ordinal, non-two_stage) rows.
                              Ordinal / two_stage rows appear only as the right-
                              side reference value, never as a displayed row.
    base_vs_enh             → no filtering; all rows visible.
    """
    if comp_mode == "nosmote_vs_smote":
        return [r for r in rows
                if MODELS.get(r["key"], {}).get("balance_strategy") == "none"]

    if comp_mode in ("regular_vs_ordinal", "regular_vs_twostage"):
        def _is_regular(r):
            mn = MODELS.get(r["key"], {}).get("model", "")
            return not (mn.startswith("ordinal_") or mn.startswith("two_stage_"))
        return [r for r in rows if _is_regular(r)]

    return rows


def _enrich_rows_with_comp(rows, comp_mode, result_type):
    """
    Add left_{m} / right_{m} / left_label / right_label / right_name to every
    row dict so that render_leaderboard is fully mode-agnostic.

    Modes
    -----
    base_vs_enh
        left  = baseline metric  (no survival features)
        right = enhanced metric  (with survival features)

    regular_vs_ordinal
        left  = the row's own *enhanced* metric
        right = best *enhanced* metric across all ordinal_ variants that share
                the same base algorithm (ada_boost / random_forest / xgboost).
                Models without an ordinal counterpart get right_{m} = None → "N/A".

    regular_vs_twostage
        left  = the row's own *enhanced* metric
        right = global best *enhanced* metric across ALL two_stage_* models
                (same benchmark value on every row).

    nosmote_vs_smote
        left  = the row's own *enhanced* metric  (row is already filtered to
                balance_strategy == 'none' rows only)
        right = best *enhanced* metric for the same (model, params) combo
                across any of the four SMOTE balance strategies.
    """
    if not rows:
        return rows

    _, left_lbl, right_lbl = COMP_MODES[comp_mode]

    # ── pre-build lookup caches (one pass over MODELS) ─────────────────────────
    if comp_mode == "regular_vs_ordinal":
        # base_algorithm → {metric: best_enhanced_float}
        ordinal_best: dict[str, dict] = {}
        for m_data in MODELS.values():
            mname = m_data["model"]
            if not mname.startswith("ordinal_"):
                continue
            base = mname[len("ordinal_"):]
            enh  = m_data["enhanced"]["evaluation"]["metrics"]
            if base not in ordinal_best:
                ordinal_best[base] = {met: 0.0 for met in METRICS}
            for met in METRICS:
                v = enh.get(met) or 0.0
                if v > ordinal_best[base][met]:
                    ordinal_best[base][met] = v

    elif comp_mode == "regular_vs_twostage":
        # single global best per metric across all two_stage_* models
        twostage_best: dict = {met: 0.0 for met in METRICS}
        for m_data in MODELS.values():
            if not m_data["model"].startswith("two_stage_"):
                continue
            enh = m_data["enhanced"]["evaluation"]["metrics"]
            for met in METRICS:
                v = enh.get(met) or 0.0
                if v > twostage_best[met]:
                    twostage_best[met] = v

    elif comp_mode == "nosmote_vs_smote":
        SMOTE_STRATS = {"smote", "borderline_smote", "smote_enn", "smote_tomek"}
        # (model_name, params_sig) → {metric: best_enhanced_float}
        smote_best: dict = {}
        for m_data in MODELS.values():
            if m_data["balance_strategy"] not in SMOTE_STRATS:
                continue
            sig = (m_data["model"],
                   json.dumps(m_data.get("parameters", {}), sort_keys=True))
            enh = m_data["enhanced"]["evaluation"]["metrics"]
            if sig not in smote_best:
                smote_best[sig] = {met: 0.0 for met in METRICS}
            for met in METRICS:
                v = enh.get(met) or 0.0
                if v > smote_best[sig][met]:
                    smote_best[sig][met] = v

    # ── enrich ─────────────────────────────────────────────────────────────────
    enriched = []
    for row in rows:
        row = dict(row)          # shallow copy — never mutate shared cache
        row["left_label"]  = left_lbl
        row["right_label"] = right_lbl
        row["right_name"]  = ""

        m_data = MODELS.get(row["key"], {})
        base_m = m_data.get("baseline", {}).get("evaluation", {}).get("metrics", {})
        enh_m  = m_data.get("enhanced",  {}).get("evaluation", {}).get("metrics", {})

        if comp_mode == "base_vs_enh":
            for met in METRICS:
                row[f"left_{met}"]  = base_m.get(met) or 0.0
                row[f"right_{met}"] = enh_m.get(met)  or 0.0

        elif comp_mode == "regular_vs_ordinal":
            base_algo = m_data.get("model", "")
            ob = ordinal_best.get(base_algo)   # None when no ordinal counterpart
            row["right_name"] = f"ordinal_{base_algo}" if ob else ""
            for met in METRICS:
                row[f"left_{met}"]  = enh_m.get(met) or 0.0
                row[f"right_{met}"] = ob[met] if ob else None

        elif comp_mode == "regular_vs_twostage":
            row["right_name"] = "Best Two-Stage"
            for met in METRICS:
                row[f"left_{met}"]  = enh_m.get(met) or 0.0
                row[f"right_{met}"] = twostage_best[met]

        elif comp_mode == "nosmote_vs_smote":
            sig = (m_data.get("model", ""),
                   json.dumps(m_data.get("parameters", {}), sort_keys=True))
            sb  = smote_best.get(sig)
            for met in METRICS:
                row[f"left_{met}"]  = enh_m.get(met) or 0.0
                row[f"right_{met}"] = sb[met] if sb else None

        enriched.append(row)
    return enriched


def _get_all_rows(sort_metric, result_type, sort_result_type,
                  model_filter, strategy_filter, comp_mode):
    """Build → mode-filter → enrich.  Single source of truth for every callback."""
    comp_mode = comp_mode or COMP_MODE_DEFAULT
    rows = build_leaderboard_rows(
        sort_metric, result_type,
        sort_result_type=sort_result_type,
        model_filter=model_filter,
        strategy_filter=strategy_filter,
    )
    rows = _filter_rows_for_comp_mode(rows, comp_mode)
    rows = _enrich_rows_with_comp(rows, comp_mode, result_type)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON MODE TOGGLE
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("comp-mode-store",                "data"),
    Output("comp-mode-base-vs-enh",          "className"),
    Output("comp-mode-regular-vs-ordinal",   "className"),
    Output("comp-mode-regular-vs-twostage",  "className"),
    Output("comp-mode-nosmote-vs-smote",     "className"),
    Output("toggle-baseline",                "children", allow_duplicate=True),
    Output("toggle-enhanced",                "children", allow_duplicate=True),
    Output("selected-model-store",           "data",     allow_duplicate=True),
    Output("page-store",                     "data",     allow_duplicate=True),
    Input("comp-mode-base-vs-enh",           "n_clicks"),
    Input("comp-mode-regular-vs-ordinal",    "n_clicks"),
    Input("comp-mode-regular-vs-twostage",   "n_clicks"),
    Input("comp-mode-nosmote-vs-smote",      "n_clicks"),
    State("sort-metric-store",               "data"),
    State("sort-dir-store",                  "data"),
    State("sort-result-type-store",          "data"),
    State("result-type-store",               "data"),
    State("filter-model-store",              "data"),
    State("filter-strategy-store",           "data"),
    prevent_initial_call=True,
)
def update_comp_mode(n1, n2, n3, n4,
                     sort_metric, sort_dir, sort_result_type,
                     result_type, model_filter, strategy_filter):
    ctx = callback_context
    if not ctx.triggered:
        return (no_update,) * 9

    btn_to_mode = {
        "comp-mode-base-vs-enh":          "base_vs_enh",
        "comp-mode-regular-vs-ordinal":   "regular_vs_ordinal",
        "comp-mode-regular-vs-twostage":  "regular_vs_twostage",
        "comp-mode-nosmote-vs-smote":     "nosmote_vs_smote",
    }
    triggered_id    = ctx.triggered[0]["prop_id"].split(".")[0]
    new_mode        = btn_to_mode.get(triggered_id, COMP_MODE_DEFAULT)
    left_lbl, right_lbl = VIEW_LABELS[new_mode]

    # Auto-select the #1 row under the new comparison mode
    sort_result_type = sort_result_type or "enhanced"
    rows = _get_all_rows(sort_metric, result_type, sort_result_type,
                         model_filter, strategy_filter, new_mode)
    if sort_dir == "asc":
        rows = list(reversed(rows))
    first_key = rows[0]["key"] if rows else no_update

    def _cls(mode):
        return "comp-mode-btn comp-mode-btn-active" if mode == new_mode else "comp-mode-btn"

    return (
        new_mode,
        _cls("base_vs_enh"),
        _cls("regular_vs_ordinal"),
        _cls("regular_vs_twostage"),
        _cls("nosmote_vs_smote"),
        left_lbl,
        right_lbl,
        first_key,
        0,          # reset to page 1
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("result-type-store", "data"),
    Output("toggle-baseline",   "className"),
    Output("toggle-enhanced",   "className"),
    Output("toggle-baseline",   "children"),
    Output("toggle-enhanced",   "children"),
    Input("toggle-baseline",    "n_clicks"),
    Input("toggle-enhanced",    "n_clicks"),
    State("comp-mode-store",    "data"),
    prevent_initial_call=True,
)
def update_result_type(n_base, n_enh, comp_mode):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    left_lbl, right_lbl = VIEW_LABELS.get(comp_mode or COMP_MODE_DEFAULT,
                                           VIEW_LABELS[COMP_MODE_DEFAULT])
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "toggle-enhanced":
        return "enhanced", "toggle-btn", "toggle-btn active-toggle", left_lbl, right_lbl
    return "baseline", "toggle-btn active-toggle", "toggle-btn", left_lbl, right_lbl


@dash_app.callback(
    Output("page-store", "data", allow_duplicate=True),
    Input("result-type-store", "data"),
    Input("comp-mode-store",   "data"),
    State("selected-model-store",   "data"),
    State("sort-metric-store",      "data"),
    State("sort-dir-store",         "data"),
    State("sort-result-type-store", "data"),
    State("filter-model-store",     "data"),
    State("filter-strategy-store",  "data"),
    State("page-size-store",        "data"),
    prevent_initial_call=True,
)
def navigate_to_selected_on_toggle(result_type, comp_mode, selected_key,
                                   sort_metric, sort_dir, sort_result_type,
                                   model_filter, strategy_filter, page_size):
    if not selected_key or not MODELS:
        return 0
    page_size        = page_size or 5
    sort_result_type = sort_result_type or "enhanced"
    all_rows = _get_all_rows(sort_metric, result_type, sort_result_type,
                             model_filter, strategy_filter, comp_mode)
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))
    keys = [r["key"] for r in all_rows]
    if selected_key not in keys:
        return 0
    return keys.index(selected_key) // page_size


@dash_app.callback(
    Output("page-size-store", "data"),
    Output("page-store", "data", allow_duplicate=True),
    Input("page-size-dropdown", "value"),
    State("selected-model-store",   "data"),
    State("sort-metric-store",      "data"),
    State("sort-dir-store",         "data"),
    State("sort-result-type-store", "data"),
    State("result-type-store",      "data"),
    State("filter-model-store",     "data"),
    State("filter-strategy-store",  "data"),
    State("comp-mode-store",        "data"),
    prevent_initial_call=True,
)
def update_page_size(value, selected_key, sort_metric, sort_dir,
                     sort_result_type, result_type, model_filter,
                     strategy_filter, comp_mode):
    new_size         = value or 5
    sort_result_type = sort_result_type or "enhanced"
    if not selected_key or not MODELS:
        return new_size, 0
    all_rows = _get_all_rows(sort_metric, result_type, sort_result_type,
                             model_filter, strategy_filter, comp_mode)
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))
    keys     = [r["key"] for r in all_rows]
    new_page = keys.index(selected_key) // new_size if selected_key in keys else 0
    return new_size, new_page


@dash_app.callback(
    Output("sort-metric-store",   "data"),
    Output("sort-dir-store",      "data"),
    Output("sort-result-type-store", "data"),
    Output("page-store",          "data"),
    Output("selected-model-store", "data", allow_duplicate=True),
    Input({"type": "sort-btn", "metric": ALL}, "n_clicks"),
    Input("page-first", "n_clicks"),
    Input("page-prev",  "n_clicks"),
    Input("page-next",  "n_clicks"),
    Input("page-last",  "n_clicks"),
    State("result-type-store",      "data"),
    State("sort-metric-store",      "data"),
    State("sort-dir-store",         "data"),
    State("sort-result-type-store", "data"),
    State("page-store",             "data"),
    State("selected-model-store",   "data"),
    State("page-size-store",        "data"),
    State("filter-model-store",     "data"),
    State("filter-strategy-store",  "data"),
    State("comp-mode-store",        "data"),
    prevent_initial_call=True,
)
def update_sort_and_page(sort_btn_clicks, first, prev, nxt, last,
                         result_type, state_metric, state_dir, state_sort_rt,
                         state_page, selected_key, page_size,
                         model_filter, strategy_filter, comp_mode):
    PAGINATION_IDS = {"page-first", "page-prev", "page-next", "page-last"}
    page_size     = page_size or 5
    state_sort_rt = state_sort_rt or "enhanced"
    no_change     = (state_metric, state_dir, state_sort_rt, state_page, no_update)

    ctx = callback_context
    if not ctx.triggered:
        return no_change

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    def _first_row_key(metric, direction, sort_rt):
        rows = _get_all_rows(metric, result_type, sort_rt,
                             model_filter, strategy_filter, comp_mode)
        if direction == "asc":
            rows = list(reversed(rows))
        return rows[0]["key"] if rows else ""

    try:
        parsed = json.loads(trigger_id)
        if parsed.get("type") == "sort-btn":
            if not ctx.triggered[0]["value"]:
                return no_change
            metric  = parsed["metric"]
            new_dir = "asc" if (metric == state_metric and state_dir == "desc") else "desc"
            new_srt = result_type
            return metric, new_dir, new_srt, 0, _first_row_key(metric, new_dir, new_srt)
        return no_change
    except (json.JSONDecodeError, KeyError):
        pass

    if trigger_id in PAGINATION_IDS:
        total_rows = len(_get_all_rows(state_metric, result_type, state_sort_rt,
                                       model_filter, strategy_filter, comp_mode))
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


@dash_app.callback(
    Output("leaderboard-table-container", "children"),
    Output("page-indicator",              "children"),
    Input("current_step",           "data"),
    Input("sort-metric-store",      "data"),
    Input("sort-dir-store",         "data"),
    Input("sort-result-type-store", "data"),
    Input("result-type-store",      "data"),
    Input("page-store",             "data"),
    Input("step4-data-loaded",      "data"),
    Input("page-size-store",        "data"),
    Input("selected-model-store",   "data"),
    Input("filter-model-store",     "data"),
    Input("filter-strategy-store",  "data"),
    Input("comp-mode-store",        "data"),
    prevent_initial_call=True,
)
def render_leaderboard(current_step, sort_metric, sort_dir, sort_result_type, result_type,
                       page, _loaded, page_size, selected_key,
                       model_filter, strategy_filter, comp_mode):
    if not MODELS:
        return no_update, no_update

    comp_mode        = comp_mode or COMP_MODE_DEFAULT
    page_size        = page_size or 5
    sort_result_type = sort_result_type or "enhanced"

    all_rows = _get_all_rows(sort_metric, result_type, sort_result_type,
                             model_filter, strategy_filter, comp_mode)
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))

    total_rows = len(all_rows)
    last_page  = max(0, (total_rows - 1) // page_size) if total_rows else 0
    page       = max(0, min(page, last_page))
    rows       = all_rows[page * page_size : (page + 1) * page_size]

    # Best right-side value per metric (for best-cell highlight)
    best = {}
    for met in METRICS:
        vals = [r[f"right_{met}"] for r in all_rows if r.get(f"right_{met}") is not None]
        best[met] = max(vals) if vals else 0.0

    left_lbl  = rows[0]["left_label"]  if rows else COMP_MODES[comp_mode][1]
    right_lbl = rows[0]["right_label"] if rows else COMP_MODES[comp_mode][2]

    # ── column sub-headers ────────────────────────────────────────────────────
    sub_headers = [
        html.Th("", className="th-rank"),
        html.Th("Model", className="th-model"),
        html.Th("SMOTE Variant", className="th-strategy"),
    ]
    for met in METRICS:
        active = sort_metric == met
        act_cls = "th-sub-active" if active else ""
        sub_headers.append(html.Th(left_lbl,  className=f"th-sub {act_cls}".strip()))
        sub_headers.append(html.Th(right_lbl, className=f"th-sub enh-col {act_cls}".strip()))

    thead = html.Thead([
        html.Tr([
            html.Th("", className="th-rank"),
            html.Th("", className="th-model"),
            html.Th("", className="th-strategy"),
            *[_metric_header(met, METRIC_LABELS[met]) for met in METRICS],
        ]),
        html.Tr(sub_headers),
    ])

    # ── body rows ─────────────────────────────────────────────────────────────
    tbody_rows = []
    for row in rows:
        global_rank  = all_rows.index(row) + 1
        display_rank = (total_rows - global_rank + 1) if sort_dir == "asc" else global_rank
        rank_cls     = ("gold"   if display_rank == 1 else
                        "silver" if display_rank == 2 else
                        "bronze" if display_rank == 3 else "default")

        right_name = row.get("right_name", "")

        cells = [
            html.Td(html.Span(f"#{display_rank}", className=f"rank-badge rank-{rank_cls}"),
                    className="td-rank"),
            html.Td(html.Div([html.Span(_model_display(row["name"]), className="model-name")]),
                    className="td-model"),
            html.Td(html.Span(_strategy_display(row["strategy"]), className="strategy-pill"),
                    className="td-strategy"),
        ]

        for met in METRICS:
            left_val  = row[f"left_{met}"]
            right_val = row.get(f"right_{met}")

            left_td = html.Td(
                html.Span(f"{left_val:.3f}", className="metric-val"),
                className="td-metric",
            )

            if right_val is None:
                # No counterpart available (e.g. model has no ordinal variant)
                right_td = html.Td(
                    html.Span("N/A", className="metric-val metric-na"),
                    className="td-metric enh-col",
                )
            else:
                is_best   = abs(right_val - best[met]) < 0.0001
                inner     = [
                    html.Span(f"{right_val:.3f}", className="metric-val"),
                    delta_badge(right_val, left_val),
                ]
                # Show right_name hint only in the first metric column to avoid clutter
                if right_name and met == METRICS[0]:
                    inner.append(html.Span(right_name, className="right-name-hint"))
                right_td = html.Td(
                    html.Div(inner, className="enh-cell-inner"),
                    className=f"td-metric enh-col {'best-cell' if is_best else ''}".strip(),
                )

            cells.append(left_td)
            cells.append(right_td)

        params_json = json.dumps(row.get("parameters", {})) or ""
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


@dash_app.callback(
    Output("selected-model-store", "data", allow_duplicate=True),
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
    prev_clicks    = prev_clicks or {}
    current_clicks = {id_obj["key"]: (n or 0) for id_obj, n in zip(id_list, n_clicks_list)}
    clicked_key    = next((k for k, c in current_clicks.items() if c > prev_clicks.get(k, 0)), None)
    if clicked_key is None:
        return no_update, current_clicks
    return clicked_key, current_clicks


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISE SELECTION
#  Combined callback covering Step 4 (screen_1) and /analysis (screen_2).
#  Replaces the separate initialise_selection callbacks in screen_1.py and
#  screen_2.py — those files should no longer define this callback.
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("selected-model-store", "data", allow_duplicate=True),
    Input("step4-data-loaded",     "data"),
    State("selected-model-store",  "data"),
    State("url",                   "pathname"),
    prevent_initial_call=True,
)
def initialise_selection(loaded, current_selected, pathname):
    """
    Auto-selects the top-ranked model whenever step4-data-loaded fires.

    Step 4 (setup flow)  — guards with `if current_selected` so a manual
                           selection is not overwritten on re-entry.
    /analysis (Screen 2) — skips the guard so switching sessions always
                           resets to the new top model.
    """
    if not loaded or not MODELS:
        return no_update
    if pathname != "/analysis" and current_selected:
        return no_update
    rows = build_leaderboard_rows(
        sort_metric="f1_macro",
        result_type="enhanced",
        sort_result_type="enhanced",
    )
    return rows[0]["key"] if rows else no_update


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("features-positive-store", "data"),
    Output("features-filter-btn",     "children"),
    Output("features-filter-btn",     "className"),
    Input("features-filter-btn",      "n_clicks"),
    State("features-positive-store",  "data"),
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
    Input("current_step",             "data"),
    Input("selected-model-store",     "data"),
    Input("result-type-store",        "data"),
    Input("comp-mode-store",          "data"),
    Input("step4-data-loaded",        "data"),
    Input("features-positive-store",  "data"),
    prevent_initial_call=True,
)
def update_charts(current_step, model_key, result_type, comp_mode, _loaded, positives_only):
    if not model_key or model_key not in MODELS:
        empty = go.Figure()
        empty.update_layout(**_base_layout(""))
        return empty, empty, empty, empty, "", "Selected Features · Importance"

    # Resolve which model and which data side to show based on comp mode + toggle
    chart_key, chart_rt = _resolve_chart_key(model_key, result_type,
                                              comp_mode or COMP_MODE_DEFAULT)
    if not chart_key or chart_key not in MODELS:
        chart_key, chart_rt = model_key, result_type

    # ── Lazy chart hydration ──────────────────────────────────────────────────
    # Charts are stored as None placeholders in MODELS until first access.
    # Hydrate only the keys that are actually needed for this render.
    # Subsequent renders of the same key are free — hydration mutates MODELS
    # in-place so the data is already there on the next call.
    repo = get_active_repo()
    if repo is not None:
        for key_to_hydrate in {chart_key, model_key}:
            if key_to_hydrate not in MODELS:
                continue
            entry = MODELS[key_to_hydrate]
            # Check both phases — a None roc_curve on either side means unhydrated.
            needs_hydration = any(
                entry[phase]["evaluation"]["charts"].get("roc_curve") is None
                for phase in ("baseline", "enhanced")
            )
            if needs_hydration:
                repo.hydrate_model_charts(entry)

    # Build a descriptive label: show chart model name + which side is displayed
    comp_mode     = comp_mode or COMP_MODE_DEFAULT
    left_lbl, right_lbl = VIEW_LABELS[comp_mode]
    side_lbl      = right_lbl if result_type == "enhanced" else left_lbl
    chart_name    = _model_display(MODELS[chart_key]["model"])

    label = html.Div([
        html.Span(chart_name, className="charts-model-name"),
        html.Span(f" · {side_lbl}", className="charts-model-type"),
    ])
    raw_method     = MODELS[chart_key][chart_rt].get("feature_method", "")
    features_title = f"Selected Features · {raw_method.upper() if raw_method else 'Importance'}"

    return (
        build_roc_figure(chart_key, chart_rt),
        build_pr_figure(chart_key, chart_rt),
        build_cm_figure(chart_key, chart_rt),
        build_features_figure(chart_key, chart_rt, top15_only=bool(positives_only)),
        label,
        features_title,
    )


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
    r"""
    function(tableChildren) {
        const tooltip = document.getElementById('params-tooltip');
        const content = document.getElementById('params-tooltip-content');
        if (!tooltip || !tooltip.parentNode || !content) return window.dash_clientside.no_update;

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
            return String(k).replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
        }

        function parseParams(raw) {
            if (!raw) return null;
            try {
                const obj = JSON.parse(raw);
                if (obj && typeof obj === 'object' && !Array.isArray(obj)) return obj;
            } catch(e) {}
            try {
                const jsonified = raw.replace(/'/g, '"').replace(/True/g, 'true')
                    .replace(/False/g, 'false').replace(/None/g, 'null');
                const obj = JSON.parse(jsonified);
                if (obj && typeof obj === 'object') return obj;
            } catch(e) {}
            const result = {};
            const pairRe = /[\(\[]\s*['"]([^'"]+)['"]\s*,\s*([^,\)\]]+?)\s*[\)\]]/g;
            let m;
            while ((m = pairRe.exec(raw)) !== null) {
                const num = parseFloat(m[2].trim());
                result[m[1].trim()] = isNaN(num) ? m[2].trim() : num;
            }
            return Object.keys(result).length ? result : null;
        }

        function isTwoStage(params) {
            if (!params || typeof params !== 'object') return false;
            const vals = Object.values(params);
            return vals.length > 0 &&
                   vals.every(function(v) { return v && typeof v === 'object' && !Array.isArray(v); });
        }

        function renderFlatParams(params) {
            return Object.keys(params).map(function(k) {
                return '<div class="ptt-row"><span class="ptt-key">' + formatKey(k) +
                       '</span><span class="ptt-val">' + params[k] + '</span></div>';
            }).join('');
        }

        function renderTwoStageParams(params) {
            const stageKeys = Object.keys(params);
            return stageKeys.map(function(stageName, idx) {
                const label = stageName
                    .replace(/([a-z])(\d)/i, '$1 $2')
                    .replace(/\b\w/g, function(c) { return c.toUpperCase(); });
                const divider = idx < stageKeys.length - 1
                    ? '<div class="ptt-stage-divider"></div>' : '';
                return '<div class="ptt-stage-header">' + label + '</div>' +
                       renderFlatParams(params[stageName]) + divider;
            }).join('');
        }

        document.addEventListener('mousemove', function(e) {
            mouseX = e.clientX; mouseY = e.clientY;
            if (newTooltip.classList.contains('params-tooltip-visible')) positionTooltip();
        });
        document.addEventListener('mouseover', function(e) {
            const row = e.target.closest('tr.model-row');
            if (!row) { newTooltip.classList.remove('params-tooltip-visible'); return; }
            const raw = row.getAttribute('data-params');
            if (!raw || raw === '{}' || raw === '') { newTooltip.classList.remove('params-tooltip-visible'); return; }
            const params = parseParams(raw);
            if (!params || !Object.keys(params).length) { newTooltip.classList.remove('params-tooltip-visible'); return; }
            newContent.innerHTML = isTwoStage(params)
                ? renderTwoStageParams(params)
                : renderFlatParams(params);
            positionTooltip();
            newTooltip.classList.add('params-tooltip-visible');
        });
        document.addEventListener('mouseout', function(e) {
            const row = e.target.closest('tr.model-row');
            if (row && !row.contains(e.relatedTarget))
                newTooltip.classList.remove('params-tooltip-visible');
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("params-tooltip", "className"),
    Input("leaderboard-table-container", "children"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("sort-metric-store",      "data"),
    State("result-type-store",      "data"),
    State("sort-result-type-store", "data"),
    State("filter-model-store",     "data"),
    State("filter-strategy-store",  "data"),
    State("comp-mode-store",        "data"),
    prevent_initial_call=True,
)
def export_csv(n, sort_metric, result_type, sort_result_type,
               model_filter, strategy_filter, comp_mode):
    if not n:
        return no_update

    comp_mode = comp_mode or COMP_MODE_DEFAULT

    # Hydrate features/weights for every model before building the CSV.
    # load_models_dict() always initialises features=[] and weights=None as
    # placeholders; the real data only lands after hydrate_model_charts() is
    # called — which normally only happens when a user opens a model's charts.
    # We do a bulk hydration here so every row in the export is complete.
    repo = get_active_repo()
    if repo is not None:
        for entry in MODELS.values():
            if not entry.get("enhanced", {}).get("features"):
                repo.hydrate_model_charts(entry)

    rows      = _get_all_rows(sort_metric, result_type, sort_result_type or "enhanced",
                              model_filter, strategy_filter, comp_mode)

    _, left_lbl, right_lbl = COMP_MODES[comp_mode]
    records = []
    for r in rows:
        rec = {"Model": r["name"], "Strategy": r["strategy"]}

        # ── metric columns ────────────────────────────────────────────────────
        for met in METRICS:
            lv = r.get(f"left_{met}")
            rv = r.get(f"right_{met}")
            rec[f"{left_lbl} {METRIC_LABELS[met]}"]  = round(lv, 4) if lv is not None else ""
            rec[f"{right_lbl} {METRIC_LABELS[met]}"] = round(rv, 4) if rv is not None else "N/A"

        # ── model parameters ──────────────────────────────────────────────────
        m_data = MODELS.get(r["key"], {})
        params = m_data.get("parameters", {})
        rec["Parameters"] = json.dumps(params, sort_keys=True) if params else ""

        # ── feature weights ───────────────────────────────────────────────────
        # Mirrors build_features_figure: features & weights live inside the
        # pipeline sub-dict at MODELS[key][result_type]["features"] /
        # MODELS[key][result_type]["weights"].
        pipeline    = m_data.get(result_type) or m_data.get("enhanced") or {}
        features    = pipeline.get("features") or []
        raw_weights = pipeline.get("weights")

        if features and raw_weights:
            if isinstance(raw_weights, dict):
                weights = [raw_weights.get(f, 0.0) for f in features]
            else:
                weights = list(raw_weights)
            rec["Feature Weights"] = ", ".join(
                f"{name}:{round(float(w), 6)}"
                for name, w in zip(features, weights)
            )
        elif features:
            # weights not stored — export names only in ranked order
            rec["Feature Weights"] = ", ".join(features)
        else:
            rec["Feature Weights"] = ""

        records.append(rec)

    return dcc.send_data_frame(
        pd.DataFrame(records).to_csv,
        "model_leaderboard.csv",
        index=False,
    )
