import os
from dash import Dash
import dash_bootstrap_components as dbc

from src.layout import serve_layout
from src.callbacks import register_callbacks

# Tema Bootstrap (elige otro si quieres: LUX, COSMO, CYBORG, etc.)
external_stylesheets = [dbc.themes.LUX]

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server  # para despliegue (gunicorn, etc.)

app.title = "Telecom KPIs Monitor"
app.layout = serve_layout

register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
