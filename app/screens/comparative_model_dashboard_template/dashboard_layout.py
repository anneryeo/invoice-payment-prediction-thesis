from dash import html, dcc

from .constants import MODEL_LABELS, STRATEGY_LABELS


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
    # Sits between the leaderboard and charts. Hidden until a model is selected.
    # Always rendered so Dash output targets always exist.
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
    # Always rendered so Dash never complains about missing output targets.
    # Only shown (via className) when the user clicks Confirm on Screen 1.
    model_summary_modal = html.Div(
        id="model-summary-modal",
        className="modal-overlay modal-hidden",
        children=[
            html.Div(className="modal-box", children=[

                # Header
                html.Div(className="modal-header", children=[
                    html.H3("Confirm Selected Model", className="modal-title"),
                    html.Button(
                        "✕",
                        id="modal-close-btn",
                        className="modal-close-btn",
                        n_clicks=0,
                    ),
                ]),

                # Body — populated by callback when modal opens
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

                # Footer
                html.Div(className="modal-footer", children=[
                    html.Button(
                        "Cancel",
                        id="modal-cancel-btn",
                        className="modal-btn modal-btn-cancel",
                        n_clicks=0,
                    ),
                    # Does NOT have a Dash callback — clicks the persistent
                    # finalize_btn in initial_setup_layout via clientside JS,
                    # avoiding the removeChild error caused by Dash unmounting
                    # step-content at the same time a callback fires on it.
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
                    html.Div(className="result-toggle-wrap", children=[
                        html.Span("View:", className="toggle-label"),
                        html.Div(className="result-toggle", children=[
                            html.Button("Baseline", id="toggle-baseline", className="toggle-btn"),
                            html.Button("Enhanced", id="toggle-enhanced", className="toggle-btn active-toggle"),
                        ]),
                    ]),
                ]),
            ]),

            # ── Global controls ──────────────────────────────────────────────
            html.Div(className="global-controls", children=[
                html.Button("⊞ Filter", id="filter-panel-toggle", className="filter-toggle-btn"),
                html.Div(className="controls-right", children=[
                    html.Button("↓ Export CSV", id="export-csv-btn", className="export-btn"),
                    dcc.Download(id="download-csv"),
                ]),
            ]),

            # ── Filter panel (collapsible) ────────────────────────────────────
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

            # ── Hidden stores ────────────────────────────────────────────────
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

            # ── Parameter tooltip ────────────────────────────────────────────
            html.Div(id="params-tooltip", className="params-tooltip", children=[
                html.Div(id="params-tooltip-content", className="params-tooltip-content"),
            ]),

            # ── Confirm banner (Screen 1 only, appears when a model is selected) ─
            confirm_bar,

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