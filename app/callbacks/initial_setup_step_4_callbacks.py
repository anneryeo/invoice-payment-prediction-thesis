import pandas as pd

from dash import Input, Output, State, html, dcc, no_update
from app import dash_app

html_step_4 = (
    html.Div([
        html.Div("4", className="step-number"),
        html.H3("Model Result Analysis", className="step-header"),
        dcc.Dropdown(id="model_summary_dropdown", placeholder="Select a model"),
        dcc.Graph(id="auc_graph"),
        html.Button("Next", id="next_btn")
    ])
)

#########################################################
#  STEP 4 MODELS - Running the techniques
#########################################################
