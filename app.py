import os
import logging
import threading
import webbrowser

from dash import Dash, Input, Output, ALL
import dash_bootstrap_components as dbc
from flask_caching import Cache
from src.callbacks.main.export_callback import export_callback
from src.callbacks.main.integrity_callbacks import integrity_callbacks
from src.callbacks.topoff.export_callback import export_topoff_callback
from src.callbacks.topoff.heatmap_callbacks import topoff_heatmap_callbacks
from src.layout import serve_layout
from src.callbacks.main.callbacks import register_callbacks
from src.callbacks.umbrales_callbacks import umbral_callbacks
from src.callbacks.main.heatmap_callbacks import heatmap_callbacks, start_main_prewarm_thread
from src.callbacks.topoff.topoff_callback import register_topoff_callbacks
from src.config import APP_LOG_PATH, ASSETS_DIR
# Tema Bootstrap (elige otro si quieres: LUX, COSMO, CYBORG, etc.)
cache = Cache(config={
    "CACHE_TYPE": "SimpleCache",          # para empezar; puedes cambiar a Redis luego
    "CACHE_DEFAULT_TIMEOUT": 60           # TTL por defecto
})
external_stylesheets = [dbc.themes.LUX]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    assets_folder=str(ASSETS_DIR),
)
cache.init_app(app.server)
server = app.server  # para despliegue (gunicorn, etc.)
app.config.suppress_callback_exceptions = True
app.title = "Telecom KPIs Monitor"
app.layout = serve_layout

file_handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
)

root_logger = logging.getLogger()
if not any(
    isinstance(handler, logging.FileHandler)
    and getattr(handler, "baseFilename", None) == str(APP_LOG_PATH)
    for handler in root_logger.handlers
):
    root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)

app_logger = logging.getLogger(__name__)


@server.errorhandler(Exception)
def log_unhandled_exception(error):
    app_logger.exception("Unhandled server error", exc_info=error)
    raise error

register_callbacks(app)
umbral_callbacks(app)
export_callback(app)
heatmap_callbacks(app)
register_topoff_callbacks(app)
topoff_heatmap_callbacks(app)
export_topoff_callback(app)
integrity_callbacks(app)
start_main_prewarm_thread()

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

        var el = document.getElementById('histo-anchor');
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

def run_local_server(open_browser: bool = True) -> None:
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", os.getenv("DASH_PORT", "8050")))

    if open_browser:
        def _open_browser() -> None:
            webbrowser.open(f"http://{host}:{port}", new=1)

        browser_timer = threading.Timer(1.5, _open_browser)
        browser_timer.daemon = True
        browser_timer.start()

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_local_server()
