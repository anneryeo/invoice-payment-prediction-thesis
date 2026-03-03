from dash import html, dash_table

class AuditLogsScreen:
    def __init__(self, app):
        self.app = app

    def layout(self):
        return html.Div([
            html.H2("Audit Logs"),
            dash_table.DataTable(
                id="logs_table",
                columns=[
                    {"name": "Timestamp", "id": "timestamp"},
                    {"name": "Action", "id": "action"},
                    {"name": "Details", "id": "details"}
                ],
                page_size=10
            )
        ])