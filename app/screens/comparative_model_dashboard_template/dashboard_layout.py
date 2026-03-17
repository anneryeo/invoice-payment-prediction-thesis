from dash import html, dcc

from app.screens.comparative_model_dashboard_template.constants import MODEL_LABELS, STRATEGY_LABELS


def build_dashboard_layout(
    *,
    show_session_selector: bool = False,
    show_confirm_button: bool = False,
) -> html.Div:
    """
    Returns the full Step 4 dashboard layout.

    Parameters
    ----------
    show_session_selector : bool
        When True, renders a session-picker dropdown above the dashboard
        (Screen 2 — Standalone Analysis).
    show_confirm_button : bool
        When True, renders a "Confirm Selected Model" button in the header
        and a summary modal (Screen 1 — Initial Setup only).

    NOTE — stores
    -------------
    Most dcc.Store components live in the persistent root layout in
    initial_setup_layout_step_renderer.py.  The exception is
    ``comp-mode-store``, which is local to this dashboard and defined below.

    NOTE — no mount interval needed
    --------------------------------
    dashboard-mount-interval is intentionally absent.  In Screen 1 the entire
    layout returned by build_dashboard_layout() is placed once into the
    persistent root layout (step4-content) and never unmounted, so there is no
    mount/unmount lifecycle to manage.  Callbacks in core.py use
    Input("current_step") directly as their render trigger.
    """

    # ── Session selector (Screen 2 only) ─────────────────────────────────────
    session_selector = html.Div(
        id="session-selector-wrap",
        className="session-selector-wrap",
        children=[
            html.Label("Training Session", className="session-selector-label"),
            dcc.Dropdown(
                id="session-selector-dropdown",
                placeholder="Select a training session…",
                clearable=False,
                searchable=False,
                className="session-selector-dropdown",
            ),
        ],
    ) if show_session_selector else html.Div(id="session-selector-wrap")

    # ── Inline confirm bar (Screen 1 only) ───────────────────────────────────
    confirm_bar = html.Div(
        id="confirm-fab-wrap",
        className="confirm-bar-wrap confirm-fab-hidden",
        children=[
            html.Div(className="confirm-bar-inner", children=[
                html.Div(className="confirm-bar-left", children=[
                    html.Span("Selected model", className="confirm-bar-eyebrow"),
                    html.Span(id="confirm-fab-label", className="confirm-bar-label"),
                ]),
                html.Button(
                    "Confirm & Proceed to Step 5 →",
                    id="confirm-model-btn",
                    className="confirm-bar-btn",
                    n_clicks=0,
                ),
            ]),
        ],
    ) if show_confirm_button else html.Div(id="confirm-fab-wrap")

    # ── Model summary modal ───────────────────────────────────────────────────
    model_summary_modal = html.Div(
        id="model-summary-modal",
        className="modal-overlay modal-hidden",
        children=[
            html.Div(className="modal-box", children=[

                html.Div(className="modal-header", children=[
                    html.H3("Confirm Selected Model", className="modal-title"),
                    html.Button(
                        "✕",
                        id="modal-close-btn",
                        className="modal-close-btn",
                        n_clicks=0,
                    ),
                ]),

                html.Div(id="modal-body", className="modal-body", children=[

                    html.Div(className="modal-model-identity", children=[
                        html.Span(id="modal-model-name",    className="modal-model-name"),
                        html.Span(id="modal-strategy-pill", className="modal-strategy-pill"),
                    ]),

                    html.Div(className="modal-params-wrap", children=[
                        html.P("Hyperparameters", className="modal-section-label"),
                        html.Div(id="modal-params-content", className="modal-params-content"),
                    ]),

                    html.Div(className="modal-metrics-wrap", children=[
                        html.P("Performance Summary", className="modal-section-label"),
                        html.Div(className="modal-metrics-grid", children=[
                            html.Div(className="modal-metrics-col", children=[
                                html.P("Baseline", className="modal-metrics-col-label"),
                                html.Div(id="modal-baseline-metrics", className="modal-metrics-list"),
                            ]),
                            html.Div(className="modal-metrics-col", children=[
                                html.P("Enhanced", className="modal-metrics-col-label"),
                                html.Div(id="modal-enhanced-metrics", className="modal-metrics-list"),
                            ]),
                        ]),
                    ]),
                ]),

                html.Div(className="modal-footer", children=[
                    html.Button(
                        "Cancel",
                        id="modal-cancel-btn",
                        className="modal-btn modal-btn-cancel",
                        n_clicks=0,
                    ),
                    html.Button(
                        "Proceed to Step 5",
                        id="modal-proceed-btn",
                        className="modal-btn modal-btn-confirm",
                        n_clicks=0,
                    ),
                ]),
            ]),
        ],
    )

    return html.Div(
        className="dashboard-root",
        children=[

            # ── Comparison-mode store (local to this dashboard) ───────────────
            dcc.Store(id="comp-mode-store", data="base_vs_enh"),

            session_selector,
            model_summary_modal,

            # ── Page header ──────────────────────────────────────────────────
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
                    # ── Baseline / Enhanced toggle ────────────────────────────
                    html.Div(className="result-toggle-wrap", children=[
                        html.Span("View:", className="toggle-label"),
                        html.Div(className="result-toggle", children=[
                            html.Button("Baseline", id="toggle-baseline", className="toggle-btn"),
                            html.Button("Enhanced", id="toggle-enhanced", className="toggle-btn active-toggle"),
                        ]),
                    ]),
                    # ── Comparison mode toggle ────────────────────────────────
                    html.Div(className="comp-mode-bar", children=[
                        html.Span("Compare:", className="comp-mode-label"),
                        html.Div(className="comp-mode-toggle-group", children=[
                            html.Button(
                                "Base vs Enhanced",
                                id="comp-mode-base-vs-enh",
                                className="comp-mode-btn comp-mode-btn-active",
                                n_clicks=0,
                            ),
                            html.Button(
                                "Regular vs Ordinal",
                                id="comp-mode-regular-vs-ordinal",
                                className="comp-mode-btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "Regular vs Two-Stage",
                                id="comp-mode-regular-vs-twostage",
                                className="comp-mode-btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "No SMOTE vs Best SMOTE",
                                id="comp-mode-nosmote-vs-smote",
                                className="comp-mode-btn",
                                n_clicks=0,
                            ),
                        ]),
                    ]),
                ]),
            ]),

            # ── Filter controls wrap (positions the floating filter panel) ────
            html.Div(className="filter-controls-wrap", children=[

                # ── Global controls ──────────────────────────────────────────
                html.Div(className="global-controls", children=[
                    html.Button("⊞ Filter", id="filter-panel-toggle", className="filter-toggle-btn"),
                    html.Div(className="controls-right", children=[
                        html.Button("↓ Export CSV", id="export-csv-btn", className="export-btn"),
                        dcc.Download(id="download-csv"),
                    ]),
                ]),

                # ── Filter panel (floats below the controls bar) ──────────────
                html.Div(id="filter-panel", className="filter-panel filter-panel-hidden", children=[
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

            ]),  # end filter-controls-wrap

            # ── Parameter tooltip ────────────────────────────────────────────
            html.Div(id="params-tooltip", className="params-tooltip", children=[
                html.Div(id="params-tooltip-content", className="params-tooltip-content"),
            ]),

            # ── Leaderboard ──────────────────────────────────────────────────
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

            # ── Confirm bar (Screen 1 only) ───────────────────────────────────
            confirm_bar,

            # ── Charts ───────────────────────────────────────────────────────
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
                            children=[dcc.Graph(id="chart-features", config={"displayModeBar": False})],
                        ),
                    ]),
                ]),
            ]),
        ],
    )