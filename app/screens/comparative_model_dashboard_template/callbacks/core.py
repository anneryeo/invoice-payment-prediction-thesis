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
    _model_display, _strategy_display,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _metric_header(metric, label):
    return html.Th(
        html.Button(label, id={"type": "sort-btn", "metric": metric}, className="sort-btn"),
        colSpan=2,
        className="metric-header-cell",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("result-type-store", "data"),
    Output("toggle-baseline",   "className"),
    Output("toggle-enhanced",   "className"),
    Input("toggle-baseline",    "n_clicks"),
    Input("toggle-enhanced",    "n_clicks"),
    prevent_initial_call=True,
)
def update_result_type(n_base, n_enh):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "toggle-enhanced":
        return "enhanced", "toggle-btn", "toggle-btn active-toggle"
    return "baseline", "toggle-btn active-toggle", "toggle-btn"


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
                         result_type, state_metric, state_dir, state_sort_rt,
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


@dash_app.callback(
    Output("leaderboard-table-container", "children"),
    Output("page-indicator",              "children"),
    # current_step as Input fires this callback when step 4 becomes active.
    # Because step4-content is permanently in the DOM, all output nodes always
    # exist — no unmount crash possible, no guard needed.
    Input("current_step",            "data"),
    Input("sort-metric-store",       "data"),
    Input("sort-dir-store",          "data"),
    Input("sort-result-type-store",  "data"),
    Input("result-type-store",       "data"),
    Input("page-store",              "data"),
    Input("step4-data-loaded",       "data"),
    Input("page-size-store",         "data"),
    Input("selected-model-store",    "data"),
    Input("filter-model-store",      "data"),
    Input("filter-strategy-store",   "data"),
    prevent_initial_call=True,
)
def render_leaderboard(current_step, sort_metric, sort_dir, sort_result_type, result_type,
                       page, _loaded, page_size, selected_key,
                       model_filter, strategy_filter):
    if not MODELS:
        return no_update, no_update

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
    rows       = all_rows[page * page_size : (page + 1) * page_size]

    best = {}
    for m in METRICS:
        col  = f"enh_{m}" if result_type == "enhanced" else f"base_{m}"
        vals = [r[col] for r in all_rows]
        best[m] = max(vals) if vals else 0

    is_enhanced = result_type == "enhanced"

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
    for row in rows:
        global_rank  = all_rows.index(row) + 1
        display_rank = (total_rows - global_rank + 1) if sort_dir == "asc" else global_rank
        rank_cls     = "gold" if display_rank == 1 else "silver" if display_rank == 2 else "bronze" if display_rank == 3 else "default"
        cells = [
            html.Td(html.Span(f"#{display_rank}", className=f"rank-badge rank-{rank_cls}"), className="td-rank"),
            html.Td(html.Div([html.Span(_model_display(row["name"]), className="model-name")]), className="td-model"),
            html.Td(html.Span(_strategy_display(row["strategy"]), className="strategy-pill"), className="td-strategy"),
        ]
        for m in METRICS:
            base_val = row[f"base_{m}"]
            enh_val  = row[f"enh_{m}"]
            primary  = enh_val if is_enhanced else base_val
            is_best  = abs(primary - best[m]) < 0.0001
            if is_enhanced:
                cells.append(html.Td(html.Span(f"{base_val:.3f}", className="metric-val"), className="td-metric"))
                cells.append(html.Td(
                    html.Div([html.Span(f"{enh_val:.3f}", className="metric-val"), delta_badge(enh_val, base_val)], className="enh-cell-inner"),
                    className=f"td-metric enh-col {'best-cell' if is_best else ''}".strip(),
                ))
            else:
                cells.append(html.Td(html.Span(f"{enh_val:.3f}", className="metric-val"), className="td-metric enh-col"))
                cells.append(html.Td(
                    html.Div([html.Span(f"{base_val:.3f}", className="metric-val"), delta_badge(base_val, enh_val)], className="enh-cell-inner"),
                    className=f"td-metric {'best-cell' if is_best else ''}".strip(),
                ))

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
    prev_clicks    = prev_clicks or {}
    current_clicks = {id_obj["key"]: (n or 0) for id_obj, n in zip(id_list, n_clicks_list)}
    clicked_key    = next((k for k, c in current_clicks.items() if c > prev_clicks.get(k, 0)), None)
    if clicked_key is None:
        return no_update, current_clicks
    return clicked_key, current_clicks


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
    # Same as render_leaderboard — current_step as Input fires on step-4 mount.
    # step4-content is always in the DOM so all chart nodes always exist.
    Input("current_step",             "data"),
    Input("selected-model-store",     "data"),
    Input("result-type-store",        "data"),
    Input("step4-data-loaded",        "data"),
    Input("features-positive-store",  "data"),
    prevent_initial_call=True,
)
def update_charts(current_step, model_key, result_type, _loaded, positives_only):
    if not model_key or model_key not in MODELS:
        empty = go.Figure()
        empty.update_layout(**_base_layout(""))
        return empty, empty, empty, empty, "", "Selected Features · Importance"

    name  = _model_display(MODELS[model_key]["model"])
    label = html.Div([
        html.Span(name, className="charts-model-name"),
        html.Span(f" · {result_type.capitalize()}", className="charts-model-type"),
    ])
    raw_method     = MODELS[model_key][result_type].get("feature_method", "")
    features_title = f"Selected Features · {raw_method.upper() if raw_method else 'Importance'}"

    return (
        build_roc_figure(model_key, result_type),
        build_pr_figure(model_key, result_type),
        build_cm_figure(model_key, result_type),
        build_features_figure(model_key, result_type, top15_only=bool(positives_only)),
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
    """
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
            return String(k).replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
        }

        function parseParams(raw) {
            if (!raw) return null;
            try {
                const obj = JSON.parse(raw);
                if (obj && typeof obj === 'object' && !Array.isArray(obj)) return obj;
            } catch(_) {}
            try {
                const jsonified = raw.replace(/'/g, '"').replace(/True/g, 'true')
                    .replace(/False/g, 'false').replace(/None/g, 'null');
                const obj = JSON.parse(jsonified);
                if (obj && typeof obj === 'object') return obj;
            } catch(_) {}
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
            newContent.innerHTML = Object.keys(params).map(k =>
                '<div class="ptt-row"><span class="ptt-key">' + formatKey(k) +
                '</span><span class="ptt-val">' + params[k] + '</span></div>'
            ).join('');
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