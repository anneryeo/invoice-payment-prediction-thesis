# screens/initial_setup.py
#
# InitialSetupScreen no longer owns any IDs or stores.
# The single source of truth is initial_setup_layout in
# initial_setup_layout_step_renderer.py — imported and returned here
# so app.py can mount it via screen.layout() as normal.

from src.app.screens.initial_setup.callbacks.initial_setup_layout_step_renderer import initial_setup_layout


class InitialSetupScreen:
    def __init__(self, dash_app):
        self.dash_app = dash_app

    def layout(self):
        return initial_setup_layout