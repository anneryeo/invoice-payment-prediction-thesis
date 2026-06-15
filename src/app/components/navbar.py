# components/navbar.py
#
# Sidebar navigation component.
# Rendered as a fixed left panel on all non-setup screens.
# Active link highlight is driven by the current URL pathname.

from dash import html, dcc

_NAV_ITEMS = [
    {"label": "Dashboard",          "href": "/dashboard", "icon": "📊"},
    {"label": "Model Analysis",     "href": "/analysis",  "icon": "🔬"},
    {"label": "Invoice Prediction", "href": "/drilldown", "icon": "🔍"},
    {"label": "Audit Logs",         "href": "/logs",      "icon": "📋"},
    {"label": "Settings",           "href": "/settings",  "icon": "⚙️"},
]


def Sidebar(current_path: str = "/dashboard") -> html.Div:
    """
    Return the sidebar HTML element.

    Parameters
    ----------
    current_path : str
        The current URL pathname.  Used to add the ``active`` CSS class
        to the matching nav item.
    """
    nav_links = []
    for item in _NAV_ITEMS:
        is_active = current_path == item["href"]
        nav_links.append(
            dcc.Link(
                href=item["href"],
                className="sidebar-nav-item" + (" active" if is_active else ""),
                children=[
                    html.Span(item["icon"], className="sidebar-nav-icon"),
                    html.Span(item["label"]),
                ],
            )
        )

    return html.Div(
        className="sidebar",
        id="sidebar",
        children=[
            # Brand
            html.Div(
                className="sidebar-brand",
                children=[
                    html.Span(className="sidebar-accent-bar"),
                    html.P("IPPP System", className="sidebar-brand-title"),
                    html.P("Invoice Payment Prediction", className="sidebar-brand-sub"),
                ],
            ),
            # Navigation
            html.Nav(className="sidebar-nav", children=nav_links),
            # Footer
            html.Div(
                className="sidebar-footer",
                children=[
                    html.Div("v2.0 — MVP"),
                    html.Div("School of IT · Mapúa University"),
                ],
            ),
        ],
    )
