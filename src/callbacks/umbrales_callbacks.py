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
    """
    Devuelve el id del componente que disparÃ³ el callback.
    - Dash nuevo: usa ctx.triggered_id
    - Dash viejo: usa callback_context.triggered
    """
    try:
        from dash import ctx  # type: ignore
        return ctx.triggered_id
    except Exception:
        from dash import callback_context  # type: ignore
        if not callback_context.triggered:
            return None
        return callback_context.triggered[0]["prop_id"].split(".")[0]


def _sync_thresholds_if_api(profile="main"):
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
        """
        Abre/cierra el modal de configuraciÃ³n.
        - Si se abre o se cierra (cancel/save), tambiÃ©n refresca el store con la config actual.
        """
        trig = _trigger_id()
        if not trig:
            raise PreventUpdate

        if trig in ("open-umbral-config", "umbral-cancel", "umbral-save"):
            # alterna el modal y carga la config actual al store
            return (not is_open), UM_MANAGER.config()

        return is_open, no_update

    # -------------------------------------------------
    # Al cambiar mÃ©trica/network/tabla: mostrar panel correcto
    # -------------------------------------------------
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
        """
        Cuando cambias:
        - MÃ©trica
        - Network (Global o override)
        - Tabla/Perfil (main/topoff)
        decide si mostrar:
        - Panel de severidad (4 colores)
        - Panel de progress (min/max)
        y precarga los valores actuales desde UM_MANAGER.
        """
        if not metric:
            raise PreventUpdate

        network = net_value or None           # "" => Global => None
        profile = table_profile or "main"     # perfil por defecto

        # Detecta tipo de mÃ©trica segÃºn el perfil
        is_sev = UM_MANAGER.is_severity(metric, profile=profile)
        is_prog = UM_MANAGER.is_progress(metric, profile=profile)

        # Texto de ayuda (Ã¡mbito + tabla)
        scope_txt = "(Global)" if network is None else f"(Override Â· {network})"
        table_txt = f"Â· Tabla: {profile}"

        # --- Caso Severidad ---
        if is_sev:
            # 1) intenta override por network, 2) fallback a global
            sev_cfg = (
                UM_MANAGER.get_severity(metric, network=network, profile=profile)
                or UM_MANAGER.get_severity(metric, profile=profile)
                or {}
            )
            thr = sev_cfg.get("thresholds", {})
            ori = sev_cfg.get("orientation", "lower_is_better")

            help_txt = (
                f"Tipo: Severidad (4 colores) Â· Ãmbito: {scope_txt} {table_txt} Â· "
                f"OrientaciÃ³n: {'Mayor es mejor' if ori == 'higher_is_better' else 'Menor es mejor'}"
            )

            return (
                False, True, help_txt,  # muestra severidad / oculta progress
                thr.get("excelente"), thr.get("bueno"), thr.get("regular"), thr.get("critico"),
                no_update, no_update
            )

        # --- Caso Progress ---
        if is_prog:
            # 1) intenta override por network, 2) fallback a global
            p_cfg = (
                UM_MANAGER.get_progress(metric, network=network, profile=profile)
                or UM_MANAGER.get_progress(metric, profile=profile)
                or {}
            )

            help_txt = f"Tipo: Progress (min/max) Â· Ãmbito: {scope_txt} {table_txt}"

            return (
                True, False, help_txt,  # oculta severidad / muestra progress
                no_update, no_update, no_update, no_update,
                p_cfg.get("min"), p_cfg.get("max")
            )

        # Si la mÃ©trica no es severidad ni progress, oculta ambos paneles
        return True, True, "", no_update, no_update, no_update, no_update, no_update, no_update

    # -------------------------------------------------
    # Guardar umbral (severidad o progress)
    # -------------------------------------------------
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
        """
        Guarda la configuraciÃ³n en UM_MANAGER:
        - Si es severidad: valida 4 valores y orden segÃºn orientaciÃ³n.
        - Si es progress: valida min/max (max > min).
        TambiÃ©n muestra Toast en Ã©xito o Alert en error.
        """
        if not metric:
            raise PreventUpdate

        network = net_value or None
        profile = table_profile or "main"

        # === Guardar Severidad ===
        if UM_MANAGER.is_severity(metric, profile=profile):
            vals = [ex, bu, re, cr]
            if any(v is None for v in vals):
                return no_update, "Faltan valores en severidad.", True, True, "Complete los 4 campos."

            # Toma orientaciÃ³n actual (override o global) para validar orden
            cur = (
                UM_MANAGER.get_severity(metric, network=network, profile=profile)
                or UM_MANAGER.get_severity(metric, profile=profile)
                or {"orientation": "lower_is_better"}
            )
            ori = cur.get("orientation", "lower_is_better")

            ex, bu, re, cr = map(float, [ex, bu, re, cr])

            # ValidaciÃ³n de orden segÃºn orientaciÃ³n
            if ori == "higher_is_better":
                if not (ex >= bu >= re >= cr):
                    return (
                        no_update,
                        "Orden invÃ¡lido.",
                        True,
                        True,
                        "Para 'mayor es mejor': excelente â‰¥ bueno â‰¥ regular â‰¥ crÃ­tico.",
                    )
            else:
                if not (ex <= bu <= re <= cr):
                    return (
                        no_update,
                        "Orden invÃ¡lido.",
                        True,
                        True,
                        "Para 'menor es mejor': excelente â‰¤ bueno â‰¤ regular â‰¤ crÃ­tico.",
                    )

            # Upsert: guarda (global u override) en el perfil indicado
            UM_MANAGER.upsert_severity(
                metric,
                thresholds={"excelente": ex, "bueno": bu, "regular": re, "critico": cr},
                network=network,
                profile=profile,
            )

            scope = "(Global)" if network is None else f"({network})"
            sync_result = _sync_thresholds_if_api(profile="main")
            return (
                UM_MANAGER.config(),
                f"Severidad guardada para {metric} {scope} - Tabla {profile}." + _sync_suffix(sync_result),
                True,
                False,
                "",
            )

        # === Guardar Progress ===
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
            sync_result = _sync_thresholds_if_api(profile="main")
            return (
                UM_MANAGER.config(),
                f"Rango guardado para {metric} {scope} - Tabla {profile}." + _sync_suffix(sync_result),
                True,
                False,
                "",
            )

        # MÃ©trica no reconocida (no es severidad ni progress)
        return no_update, "MÃ©trica desconocida.", True, True, "Seleccione una mÃ©trica vÃ¡lida."

    # -------------------------------------------------
    # Sincroniza opciones de Network del modal con el filtro principal
    # -------------------------------------------------
    @app.callback(
        Output("umbral-network", "options"),
        Input("f-network", "options"),   # reutiliza el dropdown principal
        prevent_initial_call=False,
    )
    def sync_network_options(filter_net_options):
        """
        Arma las opciones del dropdown Network en el modal:
        - (Global) siempre primero
        - luego las mismas options del filtro principal de redes
        """
        base = [{"label": "(Global)", "value": ""}]
        return base + (filter_net_options or [])
