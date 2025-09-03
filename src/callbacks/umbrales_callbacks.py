from dash import Input, Output, State, no_update

from src.Utils.umbrales.umbrales_manager import UmbralesManager
from dash.exceptions import PreventUpdate
from src.Utils.umbrales.umbrales_manager import UM_MANAGER

umbrales_manager = UmbralesManager()

def _trigger_id():
    """Dash ctx compatibility across versions.
    - Dash >=2.9: use ctx.triggered_id
    - Older versions: use callback_context.triggered[0]['prop_id']
    """
    try:
        from dash import ctx  # type: ignore
        return ctx.triggered_id
    except Exception:
        from dash import callback_context  # type: ignore
        if not callback_context.triggered:
            return None
        return callback_context.triggered[0]["prop_id"].split(".")[0]

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
        trig = _trigger_id()
        if not trig:
            raise PreventUpdate
        if trig in ("open-umbral-config", "umbral-cancel", "umbral-save"):
            return (not is_open), UM_MANAGER.config()
        return is_open, no_update

    # --- reemplaza la firma y el cuerpo de on_metric_change ---
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
        Input("umbral-network", "value"),  # 👈 NUEVO
        State("umbral-config-store", "data"),
        prevent_initial_call=True,
    )
    def on_metric_change(metric, net_value, store):
        from dash import no_update
        if not metric:
            raise PreventUpdate
        network = net_value or None

        is_sev = UM_MANAGER.is_severity(metric)
        is_prog = UM_MANAGER.is_progress(metric)
        scope_txt = "(Global)" if network is None else f"(Override · {network})"

        if is_sev:
            sev_cfg = UM_MANAGER.get_severity(metric, network=network) or UM_MANAGER.get_severity(metric) or {}
            thr = sev_cfg.get("thresholds", {})
            ori = sev_cfg.get("orientation", "lower_is_better")
            help_txt = f"Tipo: Severidad (4 colores) · Ámbito: {scope_txt} · Orientación: {'Mayor es mejor' if ori == 'higher_is_better' else 'Menor es mejor'}"
            return (
                False, True, help_txt,
                thr.get("excelente"), thr.get("bueno"), thr.get("regular"), thr.get("critico"),
                no_update, no_update
            )

        if is_prog:
            p_cfg = UM_MANAGER.get_progress(metric, network=network) or UM_MANAGER.get_progress(metric) or {}
            help_txt = f"Tipo: Progress (min/max) · Ámbito: {scope_txt}"
            return (
                True, False, help_txt,
                no_update, no_update, no_update, no_update,
                p_cfg.get("min"), p_cfg.get("max")
            )

        return True, True, "", no_update, no_update, no_update, no_update, no_update, no_update

    # --- Save ---
    @app.callback(
        Output("umbral-config-store", "data", allow_duplicate=True),
        Output("umbral-toast", "children"),
        Output("umbral-toast", "is_open"),
        Output("umbral-error", "is_open"),
        Output("umbral-error", "children"),
        Input("umbral-save", "n_clicks"),
        State("umbral-metric", "value"),
        State("umbral-network", "value"),  # 👈 NUEVO
        State("sev-excelente", "value"),
        State("sev-bueno", "value"),
        State("sev-regular", "value"),
        State("sev-critico", "value"),
        State("progress-min", "value"),
        State("progress-max", "value"),
        prevent_initial_call=True,
    )
    def save_metric(_, metric, net_value, ex, bu, re, cr, pmin, pmax):
        if not metric:
            raise PreventUpdate

        network = net_value or None  # "" => Global

        if UM_MANAGER.is_severity(metric):
            vals = [ex, bu, re, cr]
            if any(v is None for v in vals):
                return no_update, "Faltan valores en severidad.", True, True, "Complete los 4 campos."

            cur = UM_MANAGER.get_severity(metric, network=network) or UM_MANAGER.get_severity(metric) or {
                "orientation": "lower_is_better"}
            ori = cur.get("orientation", "lower_is_better")
            ex, bu, re, cr = map(float, [ex, bu, re, cr])

            if ori == "higher_is_better":
                if not (ex >= bu >= re >= cr):
                    return no_update, "Orden inválido.", True, True, "Para 'mayor es mejor': excelente ≥ bueno ≥ regular ≥ crítico."
            else:
                if not (ex <= bu <= re <= cr):
                    return no_update, "Orden inválido.", True, True, "Para 'menor es mejor': excelente ≤ bueno ≤ regular ≤ crítico."

            UM_MANAGER.upsert_severity(
                metric,
                thresholds={"excelente": ex, "bueno": bu, "regular": re, "critico": cr},
                network=network,  # 👈 clave!
                # orientation opcional si permites cambiarla en UI
            )
            scope = "(Global)" if network is None else f"({network})"
            return UM_MANAGER.config(), f"Severidad guardada para {metric} {scope}.", True, False, ""

        if UM_MANAGER.is_progress(metric):
            if pmin is None or pmax is None:
                return no_update, "Faltan valores en progress.", True, True, "Complete min y max."
            pmin = float(pmin)
            pmax = float(pmax)
            if pmax <= pmin:
                return no_update, "Rango inválido.", True, True, "max debe ser > min."

            UM_MANAGER.upsert_progress(
                metric,
                min_v=pmin,
                max_v=pmax,
                network=network,  # 👈 clave!
            )
            scope = "(Global)" if network is None else f"({network})"
            return UM_MANAGER.config(), f"Rango de progress guardado para {metric} {scope}.", True, False, ""

        return no_update, "Métrica desconocida.", True, True, "Seleccione una métrica válida."


    @app.callback(
        Output("umbral-network", "options"),
        Input("f-network", "options"),   # usa el dropdown de filtro de redes existente
        prevent_initial_call=False,
    )
    def sync_network_options(filter_net_options):
        base = [{"label": "(Global)", "value": ""}]
        # si f-network.options viene como [{"label":"ATT","value":"ATT"}, ...] lo reutilizamos
        return base + (filter_net_options or [])



