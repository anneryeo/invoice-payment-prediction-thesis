from dash import Input, Output, State, no_update, callback_context, ALL

from app import dash_app
from app.screens.comparative_model_dashboard_template.constants import MODEL_LABELS, STRATEGY_LABELS


@dash_app.callback(
    Output("filter-panel",        "className"),
    Output("filter-panel-toggle", "className"),
    Input("filter-panel-toggle",  "n_clicks"),
    State("filter-panel",         "className"),
    prevent_initial_call=True,
)
def toggle_filter_panel(n, current_class):
    hidden = "filter-panel-hidden" in current_class
    if hidden:
        return "filter-panel filter-panel-visible", "filter-toggle-btn filter-toggle-active"
    return "filter-panel filter-panel-hidden", "filter-toggle-btn"


@dash_app.callback(
    Output({"type": "model-filter", "key": ALL}, "value"),
    Input("filter-model-all",  "n_clicks"),
    Input("filter-model-none", "n_clicks"),
    State({"type": "model-filter", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def select_all_none_model(n_all, n_none, id_list):
    if not id_list:
        return []
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * len(id_list)
    if ctx.triggered[0]["prop_id"].split(".")[0] == "filter-model-all":
        return [[id_obj["key"]] for id_obj in id_list]
    return [[] for _ in id_list]


@dash_app.callback(
    Output({"type": "strategy-filter", "key": ALL}, "value"),
    Input("filter-strategy-all",  "n_clicks"),
    Input("filter-strategy-none", "n_clicks"),
    State({"type": "strategy-filter", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def select_all_none_strategy(n_all, n_none, id_list):
    if not id_list:
        return []
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * len(id_list)
    if ctx.triggered[0]["prop_id"].split(".")[0] == "filter-strategy-all":
        return [[id_obj["key"]] for id_obj in id_list]
    return [[] for _ in id_list]


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
    return selected if selected else list(MODEL_LABELS.keys())


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