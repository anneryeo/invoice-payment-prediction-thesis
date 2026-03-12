import copy
from dash import Input, Output, State, html, dcc, no_update
from app import dash_app

from app.screens.initial_setup.callbacks.step_1 import html_step_1
from app.screens.initial_setup.callbacks.step_2 import html_step_2
from app.screens.initial_setup.callbacks.step_3 import html_step_3, run_training
from app.screens.initial_setup.callbacks.step_4 import html_step_4
from app.screens.initial_setup.callbacks.step_5 import html_step_5, run_finalization

from app.screens.comparative_model_dashboard_template.constants import MODEL_LABELS, STRATEGY_LABELS


# ── Single source of truth for the entire Initial Setup layout ────────────────
initial_setup_layout = html.Div(
    className="setup-container",
    children=[
        # ── All persistent stores — declared exactly once ─────────────────────
        dcc.Store(id="current_step",        data="progress-1"),
        dcc.Store(id="training_status"),
        dcc.Store(id="stored_revenue"),
        dcc.Store(id="stored_enrollees"),
        dcc.Store(id="stored_models"),
        dcc.Store(id="stored_balancing"),
        dcc.Store(id="stored_credit_sales"),
        dcc.Store(id="selected-model-data"),
        dcc.Store(id="fin-training_status"),

        # ── Step-4 dashboard stores ───────────────────────────────────────────
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
        dcc.Store(id="filter-model-store",      data=list(MODEL_LABELS.keys())),
        dcc.Store(id="filter-strategy-store",   data=list(STRATEGY_LABELS.keys())),

        # Step 3 polling interval
        dcc.Interval(id="progress-interval", interval=1000, n_intervals=0, disabled=True),

        # Step 5 progress polling interval — persistent root so it reliably
        # ticks once enabled.  Writing disabled=False to a dynamically-mounted
        # component (inside step-content) is unreliable in Dash: the prop
        # change is acknowledged but the browser timer may never start.
        # Living here it is always in the DOM; run_finalization enables it by
        # writing to fin-training_status, and the polling callbacks read that
        # store to decide whether to do work.
        dcc.Interval(id="fin-progress-interval", interval=1000, n_intervals=0, disabled=True),

        # ── Sticky top: title + progress bar ─────────────────────────────────
        html.Div(
            className="setup-sticky-top",
            children=[
                html.Div(
                    className="setup-header",
                    children=[
                        html.H2("Initial Setup", className="setup-title"),
                        html.Hr(className="setup-divider"),
                    ],
                ),
                html.Div(
                    className="progress-header",
                    children=[
                        html.Div([
                            html.Div("1", className="step-number"),
                            html.Span("Upload Dataset", className="step-label"),
                        ], id="progress-1", className="progress-step active"),
                        html.Div([
                            html.Div("2", className="step-number"),
                            html.Span("Model Training Selection", className="step-label"),
                        ], id="progress-2", className="progress-step future"),
                        html.Div([
                            html.Div("3", className="step-number"),
                            html.Span("Waiting for Model Results", className="step-label"),
                        ], id="progress-3", className="progress-step future"),
                        html.Div([
                            html.Div("4", className="step-number"),
                            html.Span("Model Result Analysis", className="step-label"),
                        ], id="progress-4", className="progress-step future"),
                        html.Div([
                            html.Div("5", className="step-number"),
                            html.Span("Finalization", className="step-label"),
                        ], id="progress-5", className="progress-step future"),
                    ],
                ),
            ],
        ),

        # ── Step content — steps 1, 2, 3, 5 ──────────────────────────────────
        html.Div(id="step-content", className="setup-step"),

        # ── Step 4 content — permanently in the DOM ───────────────────────────
        html.Div(
            id="step4-content",
            className="setup-step",
            style={"display": "none"},
            children=html_step_4,
        ),
    ],
)


##############################################
# MAIN RENDER CALLBACK
##############################################

@dash_app.callback(
    Output("step-content",       "children"),
    Output("step4-content",      "style"),
    Output("progress-interval",  "disabled"),
    Output("fin-progress-interval", "disabled"),
    Input("current_step",        "data"),
    prevent_initial_call='initial_duplicate',
)
def render_step(step):
    show    = {"display": "block"}
    hide    = {"display": "none"}
    # fin-progress-interval: only enabled when on step 5
    fin_off = True
    fin_on  = False

    if step == "progress-1":
        return copy.deepcopy(html_step_1), hide, True, fin_off
    elif step == "progress-2":
        return copy.deepcopy(html_step_2), hide, True, fin_off
    elif step == "progress-3":
        return copy.deepcopy(html_step_3), hide, False, fin_off
    elif step == "progress-4":
        return html.Div(), show, True, fin_off
    elif step == "progress-5":
        import app.screens.initial_setup.callbacks.step_5 as s5
        s5._finalization_triggered = False
        return copy.deepcopy(html_step_5), hide, True, fin_on
    return no_update, no_update, no_update, no_update


##############################################
# PROGRESS HEADER CLASS UPDATE
##############################################

@dash_app.callback(
    Output("progress-1", "className"),
    Output("progress-2", "className"),
    Output("progress-3", "className"),
    Output("progress-4", "className"),
    Output("progress-5", "className"),
    Input("current_step", "data"),
)
def update_progress_classes(current_step):
    step_num = int(current_step.split("-")[1])
    classes = []
    for i in range(1, 6):
        if i < step_num:
            classes.append("progress-step complete")
        elif i == step_num:
            classes.append("progress-step active")
        else:
            classes.append("progress-step future")
    return classes


##############################################
# STEP ADVANCEMENT CALLBACKS
##############################################

@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("upload_confirm_btn", "n_clicks"),
    prevent_initial_call=True,
)
def go_to_step_2(upload_clicks):
    if upload_clicks:
        return "progress-2"
    return no_update


@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("model_parameters_confirm_btn", "n_clicks"),
    State("current_step", "data"),
    prevent_initial_call=True,
)
def go_to_step_3(confirm_clicks, current_step):
    if confirm_clicks and current_step != "progress-3":
        return "progress-3"
    return no_update


@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("next_btn", "n_clicks"),
    prevent_initial_call=True,
)
def go_to_step_4(next_clicks):
    if next_clicks:
        return "progress-4"
    return no_update


# --- Step 4 → Step 5 ---
# This is no longer needed since step_4.py's modal-proceed-btn
# already implements this