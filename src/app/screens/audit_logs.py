# screens/audit_logs.py
#
# Audit Logs screen — displays all events written by audit_logger.log_event().
# Auto-refreshes every 30 seconds; manual refresh button available.

from __future__ import annotations

from dash import html, dcc, dash_table, Input, Output
from dash.exceptions import PreventUpdate

from src.app import dash_app
from src.app.utils.audit_logger import read_events


# ── Component helpers ─────────────────────────────────────────────────────────

def _empty() -> html.Div:
    return html.Div(className="empty-state", children=[
        html.Span("📋", className="empty-state-icon"),
        html.P("No audit events yet.", className="empty-state-title"),
        html.P(
            "Events are recorded when predictions are run, settings are saved, "
            "and models are loaded.",
            className="empty-state-text",
        ),
    ])


def _build_table(events: list[dict]) -> dash_table.DataTable:
    rows = [
        {
            "timestamp": e.get("timestamp", "—"),
            "action":    e.get("action",    "—"),
            "details":   e.get("details",   "—"),
        }
        for e in events
    ]
    cols = [
        {"name": "Timestamp", "id": "timestamp"},
        {"name": "Action",    "id": "action"},
        {"name": "Details",   "id": "details"},
    ]
    return dash_table.DataTable(
        data=rows, columns=cols,
        page_size=20,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1b3a6b", "color": "white",
            "fontWeight": "600", "fontSize": "11px",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "padding": "10px 14px", "border": "none",
        },
        style_cell={
            "textAlign": "left", "padding": "9px 14px",
            "fontSize": "13px", "color": "#2b2b2b",
            "border": "1px solid #e5e1d8",
            "fontFamily": "-apple-system, 'Segoe UI', sans-serif",
        },
        style_cell_conditional=[
            {"if": {"column_id": "timestamp"}, "width": "220px", "minWidth": "180px"},
            {"if": {"column_id": "action"},    "width": "180px", "minWidth": "140px"},
            {"if": {"column_id": "details"},   "whiteSpace": "normal"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafaf8"},
        ],
    )


# ── Screen class ──────────────────────────────────────────────────────────────

class AuditLogsScreen:
    def __init__(self, app):
        self.app = app
        self._register_callbacks()

    def layout(self) -> html.Div:
        return html.Div(
            id="audit-root",
            children=[
                # Auto-refresh every 30 s
                dcc.Interval(
                    id="audit-refresh-interval",
                    interval=30_000,
                    n_intervals=0,
                ),

                html.Div(className="page-header", children=[
                    html.H2("Audit Logs", className="page-title"),
                    html.P(
                        "A record of all system actions — predictions, settings changes, "
                        "and model loads.",
                        className="page-subtitle",
                    ),
                ]),

                html.Div(className="card", children=[
                    html.Div(className="section-header", children=[
                        html.H4("Event Log", className="section-title"),
                        html.Button(
                            "Refresh",
                            id="audit-refresh-btn",
                            className="btn btn-outline",
                            n_clicks=0,
                        ),
                    ]),
                    html.Div(
                        id="audit-table-wrapper",
                        style={"padding": "0"},
                        children=[html.Div("Loading…", className="loading-state")],
                    ),
                ]),
            ],
        )

    def _register_callbacks(self):
        @dash_app.callback(
            Output("audit-table-wrapper", "children"),
            Input("audit-refresh-interval", "n_intervals"),
            Input("audit-refresh-btn",      "n_clicks"),
        )
        def _load(_intervals, _clicks):
            events = read_events(limit=500)
            if not events:
                return _empty()
            return [_build_table(events),
                    html.P(f"{len(events)} event(s) shown.",
                           style={"fontSize": "12px", "color": "#888",
                                  "padding": "8px 14px 12px"})]
