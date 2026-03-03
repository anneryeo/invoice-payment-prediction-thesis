# app/__init__.py
from dash import Dash
import dash_bootstrap_components as dbc

# Create the single Dash instance here
dash_app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)

server = dash_app.server