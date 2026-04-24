from dash import dcc

def LineChart(chart_id, title=""):
    return dcc.Graph(
        id=chart_id,
        figure={
            "layout": {
                "title": {"text": title, "x": 0.5},
                "paper_bgcolor": "#f9f9f9",
                "plot_bgcolor": "#f9f9f9"
            }
        }
    )