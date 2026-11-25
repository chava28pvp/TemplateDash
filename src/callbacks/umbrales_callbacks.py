import hashlib
import json
import logging
from datetime import datetime

from dash import Input, Output, State, no_update

from dash.exceptions import PreventUpdate
from src.config import DATA_SOURCE
from src.dataAccess import main_api_client
from src.Utils.umbrales.umbrales_manager import UM_MANAGER

logger = logging.getLogger(__name__)


def _trigger_id():
    """Obtiene el id del componente que activo el callback en Dash."""
    try:
        from dash import ctx  # type: ignore
        return ctx.triggered_id
    except Exception:
        from dash import callback_context  # type: ignore
        if not callback_context.triggered:
            return None
        return callback_context.triggered[0]["prop_id"].split(".")[0]


def _sync_thresholds_if_api(profile="main"):
    # La copia local sigue siendo la fuente de verdad si la API no esta disponible.
    if DATA_SOURCE != "api" or not main_api_client.is_configured():
        return None

    cfg = UM_MANAGER.config()
    raw = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    try:
        return main_api_client.sync_thresholds(
            thresholds=cfg,
            profile=profile or "main",
            config_hash=config_hash,
            updated_at=datetime.utcnow().isoformat(),
        )
    except Exception as exc:
        logger.warning("No se pudo sincronizar umbrales con API: %s", exc)
        return {"error": str(exc)}


def _sync_suffix(sync_result):
    if sync_result is None:
        return ""
    if sync_result.get("error"):
        return " Umbrales guardados localmente; no se pudo sincronizar API."
    return " Umbrales sincronizados con API."


def umbral_callbacks(app):

    @app.callback(
        Output("umbral-config-modal", "is_open"),
        Output("umbral-config-store", "data"),
        Input("open-umbral-config", "n_clicks"),
        Input("umbral-cancel", "n_clicks"),
        Input("umbral-save", "n_clicks"),
        State("umbral-config-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_modal(open_n, cancel_n, save_n, is_open):
        """Alterna el modal y carga la configuracion vigente en el store."""
        trig = _trigger_id()
        if not trig:
            raise PreventUpdate

        if trig in ("open-umbral-config", "umbral-cancel", "umbral-save"):
            return (not is_open), UM_MANAGER.config()

        return is_open, no_update

    # El formulario cambia entre umbrales de severidad y rango de progreso.
    @app.callback(
        Output("severity-panel", "hidden"),
        Output("progress-panel", "hidden"),
        Output("umbral-metric-help", "children"),
        Output("sev-excelente", "value"),
        Output("sev-bueno", "value"),
        Output("sev-regular", "value"),
        Output("sev-critico", "value"),
        Output("progress-min", "value"),
        Output("progress-max", "value"),
        Input("umbral-metric", "value"),
        Input("umbral-network", "value"),
        Input("umbral-table", "value"),
        State("umbral-config-store", "data"),
        prevent_initial_call=True,
    )
    def on_metric_change(metric, net_value, table_profile, store):
        """Selecciona el panel y precarga el valor del perfil y red elegidos."""
        if not metric:
            raise PreventUpdate

        network = net_value or None
        profile = table_profile or "main"

        # La configuracion por red tiene prioridad sobre la configuracion global.
        is_sev = UM_MANAGER.is_severity(metric, profile=profile)
        is_prog = UM_MANAGER.is_progress(metric, profile=profile)

        if is_sev:
            sev_cfg = (
                UM_MANAGER.get_severity(metric, network=network, profile=profile)
                or UM_MANAGER.get_severity(metric, profile=profile)
                or {}
            )
            thr = sev_cfg.get("thresholds", {})

            return (
                False, True, "",
                thr.get("excelente"), thr.get("bueno"), thr.get("regular"), thr.get("critico"),
                no_update, no_update
            )

        if is_prog:
            p_cfg = (
                UM_MANAGER.get_progress(metric, network=network, profile=profile)
                or UM_MANAGER.get_progress(metric, profile=profile)
                or {}
            )

            return (
                True, False, "",
                no_update, no_update, no_update, no_update,
                p_cfg.get("min"), p_cfg.get("max")
            )

        return True, True, "", no_update, no_update, no_update, no_update, no_update, no_update

    # Valida los campos antes de actualizar y sincronizar la configuracion.
    @app.callback(
        Output("umbral-config-store", "data", allow_duplicate=True),
        Output("umbral-toast", "children"),
        Output("umbral-toast", "is_open"),
        Output("umbral-error", "is_open"),
        Output("umbral-error", "children"),
        Input("umbral-save", "n_clicks"),
        State("umbral-metric", "value"),
        State("umbral-network", "value"),
        State("umbral-table", "value"),
        State("sev-excelente", "value"),
        State("sev-bueno", "value"),
        State("sev-regular", "value"),
        State("sev-critico", "value"),
        State("progress-min", "value"),
        State("progress-max", "value"),
        prevent_initial_call=True,
    )
    def save_metric(_, metric, net_value, table_profile, ex, bu, re, cr, pmin, pmax):
        """Guarda los umbrales o el rango de progreso para el perfil seleccionado."""
        if not metric:
            raise PreventUpdate

        network = net_value or None
        profile = table_profile or "main"

        if UM_MANAGER.is_severity(metric, profile=profile):
            vals = [ex, bu, re, cr]
            if any(v is None for v in vals):
                return no_update, "Faltan valores en severidad.", True, True, "Complete los 4 campos."

            # Se respeta la orientacion de la configuracion efectiva para validar el orden.
            cur = (
                UM_MANAGER.get_severity(metric, network=network, profile=profile)
                or UM_MANAGER.get_severity(metric, profile=profile)
                or {"orientation": "lower_is_better"}
            )
            ori = cur.get("orientation", "lower_is_better")

            ex, bu, re, cr = map(float, [ex, bu, re, cr])

            if ori == "higher_is_better":
                if not (ex >= bu >= re >= cr):
                    return (
                        no_update,
                        "Orden invÃ¡lido.",
                        True,
                        True,
                        "Para mayor es mejor: excelente >= bueno >= regular >= critico.",
                    )
            else:
                if not (ex <= bu <= re <= cr):
                    return (
                        no_update,
                        "Orden invÃ¡lido.",
                        True,
                        True,
                        "Para menor es mejor: excelente <= bueno <= regular <= critico.",
                    )

            UM_MANAGER.upsert_severity(
                metric,
                thresholds={"excelente": ex, "bueno": bu, "regular": re, "critico": cr},
                network=network,
                profile=profile,
            )

            scope = "(Global)" if network is None else f"({network})"
            sync_result = _sync_thresholds_if_api(profile=profile)
            return (
                UM_MANAGER.config(),
                f"Severidad guardada para {metric} {scope} - Tabla {profile}." + _sync_suffix(sync_result),
                True,
                False,
                "",
            )

        if UM_MANAGER.is_progress(metric, profile=profile):
            if pmin is None or pmax is None:
                return no_update, "Faltan valores en progress.", True, True, "Complete min y max."

            pmin = float(pmin)
            pmax = float(pmax)

            if pmax <= pmin:
                return no_update, "Rango invÃ¡lido.", True, True, "max debe ser > min."

            UM_MANAGER.upsert_progress(
                metric,
                min_v=pmin,
                max_v=pmax,
                network=network,
                profile=profile,
            )

            scope = "(Global)" if network is None else f"({network})"
            sync_result = _sync_thresholds_if_api(profile=profile)
            return (
                UM_MANAGER.config(),
                f"Rango guardado para {metric} {scope} - Tabla {profile}." + _sync_suffix(sync_result),
                True,
                False,
                "",
            )

        return no_update, "MÃ©trica desconocida.", True, True, "Seleccione una mÃ©trica vÃ¡lida."

    # El modal comparte el catalogo de redes del filtro principal.
    @app.callback(
        Output("umbral-network", "options"),
        Input("f-network", "options"),   # reutiliza el dropdown principal
        prevent_initial_call=False,
    )
    def sync_network_options(filter_net_options):
        """Agrega la opcion global antes del catalogo de redes disponibles."""
        base = [{"label": "(Global)", "value": ""}]
        return base + (filter_net_options or [])
