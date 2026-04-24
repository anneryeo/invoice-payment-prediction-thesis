from dash import dash_table

def DataTableComponent(table_id, columns):
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": col, "id": col} for col in columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
        style_header={"backgroundColor": "#0078d7", "color": "white"}
    )