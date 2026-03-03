import pandas as pd

from dash import Input, Output, State, html, dcc, no_update
from app import dash_app

html_step_5 = (
    html.Div([
        html.Div("5", className="step-number"),
        html.H3("Finalization", className="step-header"),
        html.Button("Finalize Model", id="finalize_btn")
    ])
)


#########################################################
#  STEP 5 
#########################################################
