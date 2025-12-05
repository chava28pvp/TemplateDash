from dash import Dash, Input, Output, ALL
import dash_bootstrap_components as dbc
from flask_caching import Cache
from src.callbacks.main.export_callback import export_callback
from src.callbacks.topoff.export_callback import export_topoff_callback
from src.callbacks.topoff.heatmap_callbacks import topoff_heatmap_callbacks
from src.layout import serve_layout
from src.callbacks.main.callbacks import register_callbacks
from src.callbacks.umbrales_callbacks import umbral_callbacks
from src.callbacks.main.heatmap_callbacks import heatmap_callbacks
from src.callbacks.topoff.topoff_callback import register_topoff_callbacks
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
topoff_heatmap_callbacks(app)
export_topoff_callback(app)

app.clientside_callback(
    """
    function(ts_list) {
        // ts_list = lista de n_clicks_timestamp de todos los botones de cluster
        if (!ts_list || ts_list.length === 0) {
            return window.dash_clientside.no_update;
        }

        // encontramos el timestamp más grande (último click real)
        var maxTs = 0;
        for (var i = 0; i < ts_list.length; i++) {
            var t = ts_list[i] || 0;
            if (t > maxTs) {
                maxTs = t;
            }
        }

        // Si nunca se ha clicado ningún cluster (todo 0/undefined) → no hacemos nada
        if (maxTs === 0) {
            return window.dash_clientside.no_update;
        }

        var el = document.getElementById('topoff-anchor');
        if (el && typeof el.scrollIntoView === 'function') {
            el.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }

        // valor dummy para el Output
        return '';
    }
    """,
    Output("topoff-scroll-dummy", "children"),
    Input(
        {"type": "main-cluster-link", "cluster": ALL, "vendor": ALL, "technology": ALL},
        "n_clicks_timestamp"
    ),
    prevent_initial_call=True,
)

if __name__ == "__main__":
    app.run(debug=True)
