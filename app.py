import os
from dash import Dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

from src.callbacks.export_callback import export_callback
from src.layout import serve_layout
from src.callbacks.callbacks import register_callbacks
from src.callbacks.umbrales_callbacks import umbral_callbacks
from src.callbacks.heatmap_callbacks import heatmap_callbacks
from src.callbacks.topoff_callback import register_topoff_callbacks
# Tema Bootstrap (elige otro si quieres: LUX, COSMO, CYBORG, etc.)
cache = Cache(config={
    "CACHE_TYPE": "SimpleCache",          # para empezar; puedes cambiar a Redis luego
    "CACHE_DEFAULT_TIMEOUT": 60           # TTL por defecto
})
external_stylesheets = [dbc.themes.LUX]

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
cache.init_app(app.server)
server = app.server  # para despliegue (gunicorn, etc.)
app.config.suppress_callback_exceptions = True
app.title = "Telecom KPIs Monitor"
app.layout = serve_layout

register_callbacks(app)
umbral_callbacks(app)
export_callback(app)
heatmap_callbacks(app)
register_topoff_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
