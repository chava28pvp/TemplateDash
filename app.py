# app.py
from dash import Dash
import dash_bootstrap_components as dbc

from extensiones import cache            # importa la INSTANCIA
from src.layout import serve_layout
from src.callbacks.callbacks import register_callbacks
from src.callbacks.umbrales_callbacks import umbral_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX],  # usa el tema que quieras (LUX o BOOTSTRAP, no ambos)
           suppress_callback_exceptions=True)
server = app.server

# MUY IMPORTANTE: ligar el cache al server ANTES de registrar callbacks
cache.init_app(server)

app.title = "Telecom KPIs Monitor"
app.layout = serve_layout  # callable (mejor para no compartir estado)

# Pasa la instancia de cache; evita importarla dentro de callbacks.py
register_callbacks(app, cache)
umbral_callbacks(app)   # si estos usan cache, pásalo igual

if __name__ == "__main__":
    app.run_server(debug=True)   # estándar en Dash
