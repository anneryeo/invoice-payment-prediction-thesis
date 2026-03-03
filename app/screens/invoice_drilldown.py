from dash import html, dcc, dash_table

class InvoiceDrilldownScreen:
    def __init__(self, app):
        self.app = app

    def layout(self):
        return html.Div([
            html.H2("Invoice Drill-Down"),
            dash_table.DataTable(
                id="invoice_table",
                columns=[
                    {"name": "Student ID", "id": "student_id"},
                    {"name": "Grade Level", "id": "grade_level"},
                    {"name": "Amount", "id": "amount"},
                    {"name": "Due Date", "id": "due_date"},
                    {"name": "Predicted Delay", "id": "predicted_delay"}
                ],
                page_size=10
            ),
            html.Button("Export CSV", id="export_btn")
        ])