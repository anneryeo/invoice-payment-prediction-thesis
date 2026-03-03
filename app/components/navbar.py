from dash import html, dcc

def Navbar():
    return html.Div(
        className="navbar",
        children=[
            dcc.Link("Dashboard", href="/dashboard", className="nav-link"),
            dcc.Link("Setup", href="/setup", className="nav-link"),
            dcc.Link("Training", href="/train", className="nav-link"),
            dcc.Link("Comparison", href="/compare", className="nav-link"),
            dcc.Link("Drilldown", href="/drilldown", className="nav-link"),
            dcc.Link("Logs", href="/logs", className="nav-link"),
            dcc.Link("Settings", href="/settings", className="nav-link"),
        ]
    )