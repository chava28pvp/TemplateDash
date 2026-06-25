import os
import logging
import threading
import webbrowser

from dash import Dash, Input, Output, State, ALL, ctx
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
from src.config import APP_LOG_PATH, ASSETS_DIR, DATA_SOURCE
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
app.title = "Dashboard Master"
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
if DATA_SOURCE != "api":
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


@app.callback(
    Output("table-focus-store", "data"),
    Input("main-focus-toggle", "n_clicks"),
    Input("topoff-focus-toggle", "n_clicks"),
    Input("table-focus-overlay", "n_clicks"),
    State("table-focus-store", "data"),
    prevent_initial_call=True,
)
def toggle_table_focus(_main_clicks, _topoff_clicks, _overlay_clicks, focus_state):
    focus_state = focus_state or {"main": False, "topoff": False}
    triggered = ctx.triggered_id

    if triggered == "main-focus-toggle":
        next_main = not bool(focus_state.get("main"))
        return {"main": next_main, "topoff": False}

    if triggered == "topoff-focus-toggle":
        next_topoff = not bool(focus_state.get("topoff"))
        return {"main": False, "topoff": next_topoff}

    if triggered == "table-focus-overlay":
        if not (focus_state.get("main") or focus_state.get("topoff")):
            return focus_state
        return {"main": False, "topoff": False}

    return focus_state


@app.callback(
    Output("main-table-panel", "className"),
    Output("topoff-table-panel", "className"),
    Output("table-focus-overlay", "className"),
    Output("main-focus-toggle", "children"),
    Output("topoff-focus-toggle", "children"),
    Input("table-focus-store", "data"),
)
def sync_table_focus_ui(focus_state):
    focus_state = focus_state or {"main": False, "topoff": False}
    main_on = bool(focus_state.get("main"))
    topoff_on = bool(focus_state.get("topoff"))

    main_cls = "table-focus-panel"
    topoff_cls = "table-focus-panel"
    overlay_cls = "table-focus-overlay"

    if main_on:
        main_cls += " is-maximized"
        overlay_cls += " is-visible"
    if topoff_on:
        topoff_cls += " is-maximized"
        overlay_cls += " is-visible"

    return (
        main_cls,
        topoff_cls,
        overlay_cls,
        "Restaurar" if main_on else "Expandir",
        "Restaurar" if topoff_on else "Expandir",
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
