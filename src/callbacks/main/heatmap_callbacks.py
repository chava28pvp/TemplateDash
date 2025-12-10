import math
import pandas as pd
from dash import Input, Output, State, no_update, ctx
from hashlib import md5
import json
import time
import plotly.graph_objs as go
from datetime import datetime, timedelta
from components.main.heatmap import build_heatmap_figure, render_heatmap_summary_table, build_heatmap_payloads_fast, \
    _hm_height, _build_time_header_children, _build_time_header_children_by_dates
from components.main.histograma import \
    build_histo_payloads_fast, build_overlay_waves_figure
import dash_bootstrap_components as dbc

from src.Utils.umbrales.umbrales_manager import UM_MANAGER
from src.dataAccess.data_access import fetch_kpis,fetch_alarm_meta_for_heatmap


#Cach simple en memoria para df_ts
_DFTS_CACHE = {}
_DFTS_TTL = 300  # segundos

# Última clave renderizada para evitar re-render idéntico
_LAST_HEATMAP_KEY = None
_LAST_HI_KEY = None
# Cache simple en memoria para meta de alarmados
_ALARM_META_CACHE = {}
_ALARM_META_TTL = 300  # seg

# Cache de payloads del heatmap por state_key
_HM_PAYLOAD_CACHE = {}
_HM_PAYLOAD_TTL = 120  # seg

PS_VALORES = ("PS_RRC", "PS_S1", "PS_DROP", "PS_RAB")
CS_VALORES = ("CS_RRC", "CS_DROP", "CS_RAB")

def _hm_key(fecha, networks, technologies, vendors, clusters, offset, limit):
    """Clave estable del estado visible del heatmap (sin hora)."""
    def _norm(x):
        x = x if isinstance(x, (list, tuple)) else ([] if x is None else [x])
        return sorted([str(v) for v in x if v is not None])
    obj = {
        "fecha": fecha,
        "networks": _norm(networks),
        "technologies": _norm(technologies),
        "vendors": _norm(vendors),
        "clusters": _norm(clusters),
        "offset": int(offset),
        "limit": int(limit),
    }
    return md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def _ensure_df(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

def _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters):
    """Obtiene df_ts = df(today)+df(yday) cacheado por filtros (sin hora)."""
    key = ("df_ts", today_str, yday_str,
           tuple(sorted(networks or [])),
           tuple(sorted(technologies or [])),
           tuple(sorted(vendors or [])),
           tuple(sorted(clusters or [])))
    now = time.time()
    hit = _DFTS_CACHE.get(key)
    if hit and (now - hit["ts"] < _DFTS_TTL):
        return hit["df"]

    df_today = fetch_kpis(fecha=today_str, hora=None,
                          vendors=vendors or None, clusters=clusters or None,
                          networks=networks or None, technologies=technologies or None,
                          limit=None)
    df_today = _ensure_df(df_today)

    df_yday = fetch_kpis(fecha=yday_str, hora=None,
                         vendors=vendors or None, clusters=clusters or None,
                         networks=networks or None, technologies=technologies or None,
                         limit=None)
    df_yday = _ensure_df(df_yday)

    df_ts = pd.concat([df_today, df_yday], ignore_index=True, sort=False)
    _DFTS_CACHE[key] = {"df": df_ts, "ts": now}
    return df_ts

def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]
def _cache_get(cache: dict, key, ttl: int):
    now = time.time()
    hit = cache.get(key)
    if hit and (now - hit["ts"] < ttl):
        return hit["data"]
    return None


def _cache_set(cache: dict, key, data):
    cache[key] = {"data": data, "ts": time.time()}


def _fetch_alarm_meta_cached(today_str, vendors, clusters, networks, technologies):
    key = (
        "alarm_meta",
        today_str,
        tuple(sorted(vendors or [])),
        tuple(sorted(clusters or [])),
        tuple(sorted(networks or [])),
        tuple(sorted(technologies or [])),
    )
    hit = _cache_get(_ALARM_META_CACHE, key, _ALARM_META_TTL)
    if hit is not None:
        return hit

    df_meta, keys = fetch_alarm_meta_for_heatmap(
        fecha=today_str,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
    )

    data = (df_meta, keys)
    _cache_set(_ALARM_META_CACHE, key, data)
    return data

def _valores_by_domain(domain: str):
    return CS_VALORES if str(domain).upper() == "CS" else PS_VALORES

def heatmap_callbacks(app):

    # -------------------------------------------------
    # 8) HeatMap render
    # -------------------------------------------------
    @app.callback(
        Output("hm-table-container", "children", allow_duplicate=True),
        Output("hm-pct", "figure"),
        Output("hm-unit", "figure"),
        Output("hm-page-indicator", "children", allow_duplicate=True),
        Output("hm-total-rows-banner", "children", allow_duplicate=True),
        Output("heatmap-page-info", "data"),
        Input("heatmap-trigger", "data"),
        State("f-fecha", "date"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("heatmap-page-state", "data"),
        State("hm-order-by", "value"),
        prevent_initial_call=True,
    )
    def refresh_heatmaps(_trigger, fecha, networks, technologies, vendors, clusters, hm_page_state, hm_order_by):
        """Render heatmaps + tabla alineados fila-a-fila."""
        global _LAST_HEATMAP_KEY

        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        # --- Paginado del HEATMAP ---
        page = int((hm_page_state or {}).get("page", 1))
        page_sz = int((hm_page_state or {}).get("page_size", 50))
        offset = max(0, (page - 1) * page_sz)
        limit = max(1, page_sz)

        # --- Key de estado (incluye orden) ---
        state_key = _hm_key(fecha, networks, technologies, vendors, clusters, offset, limit) + f"|ord={hm_order_by}"
        if _LAST_HEATMAP_KEY == state_key:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update
            )

        # --- Fechas HOY/AYER (sin hora) ---
        try:
            today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
        except Exception:
            today_dt = datetime.utcnow()

        yday_dt = today_dt - timedelta(days=1)
        today_str = today_dt.strftime("%Y-%m-%d")
        yday_str = yday_dt.strftime("%Y-%m-%d")

        # --- df_ts cacheado por filtros (no depende de hora) ---
        df_ts = _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters)

        # --- Redes para heatmap ---
        if networks:
            nets_heat = networks
        else:
            nets_heat = sorted(df_ts["network"].dropna().unique().tolist()) \
                if not df_ts.empty and "network" in df_ts.columns else []

        # --- Meta de alarmados (cache) ---
        # REQUIERE que tengas definido _fetch_alarm_meta_cached(...)
        df_meta_heat, alarm_keys_set = _fetch_alarm_meta_cached(
            today_str,
            vendors,
            clusters,
            nets_heat,
            technologies
        )

        # --- Payloads con cache por state_key ---
        # REQUIERE que tengas definidos _HM_PAYLOAD_CACHE, _HM_PAYLOAD_TTL, _cache_get, _cache_set
        cached = _cache_get(_HM_PAYLOAD_CACHE, state_key, _HM_PAYLOAD_TTL)
        if cached is not None:
            pct_payload, unit_payload, page_info = cached
        else:
            if df_meta_heat is not None and not df_meta_heat.empty and nets_heat:
                pct_payload, unit_payload, page_info = build_heatmap_payloads_fast(
                    df_meta=df_meta_heat,
                    df_ts=df_ts,
                    UMBRAL_CFG=UM_MANAGER.config(),
                    networks=nets_heat,
                    valores_order=("PS_RRC", "CS_RRC", "PS_S1", "PS_DROP", "CS_DROP", "PS_RAB", "CS_RAB"),
                    today=today_str, yday=yday_str,
                    alarm_keys=alarm_keys_set,
                    alarm_only=False,
                    offset=offset,
                    limit=limit,
                    order_by=hm_order_by or "alarm_hours"
                )
            else:
                pct_payload = unit_payload = None
                page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

            _cache_set(_HM_PAYLOAD_CACHE, state_key, (pct_payload, unit_payload, page_info))

        # --- Altura alineada a filas ---
        nrows = len((pct_payload or unit_payload or {}).get("y") or [])
        hm_height = _hm_height(nrows)

        # --- Figuras ---
        fig_pct = build_heatmap_figure(pct_payload, height=hm_height, decimals=2) if pct_payload else go.Figure()
        fig_unit = build_heatmap_figure(unit_payload, height=hm_height, decimals=0) if unit_payload else go.Figure()

        # --- Tabla ---
        if pct_payload or unit_payload:
            table_component = render_heatmap_summary_table(
                pct_payload, unit_payload, pct_decimals=2, unit_decimals=0
            )
        else:
            table_component = dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

        # --- Indicadores ---
        total = int(page_info.get("total_rows", 0))
        showing = int(page_info.get("showing", 0))
        start_i = int(page_info.get("offset", 0)) + 1 if showing else 0
        end_i = start_i + showing - 1 if showing else 0
        total_pg = max(1, math.ceil(total / max(1, int((hm_page_state or {}).get("page_size", 5)))))
        hm_indicator = f"Página {int((hm_page_state or {}).get('page', 1))} de {total_pg}"
        hm_banner = "Sin filas." if total == 0 else f"Mostrando {start_i}–{end_i} de {total} filas"

        _LAST_HEATMAP_KEY = state_key

        return (
            table_component,
            fig_pct,
            fig_unit,
            hm_indicator,
            hm_banner,
            page_info
        )

    @app.callback(
        Output("heatmap-trigger", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("heatmap-page-state", "data"),  # dispara por paginado del heatmap
        Input("hm-order-by", "value"),
        prevent_initial_call=False,  # permite “bootstrap” al cargar
    )
    def heatmap_trigger_controller(_fecha, _net, _tech, _vend, _clus, _page_state, _ord):
        return {"ts": time.time()}

    @app.callback(
        Output("heatmap-page-state", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("hm-page-size", "value"),
        Input("hm-order-by", "value"),
        prevent_initial_call=False,  # bootstrap
    )
    def hm_reset_page_on_filters(_fecha, _net, _tech, _ven, _clu, hm_page_size, _ord):
        ps = max(1, int(hm_page_size or 50))
        return {"page": 1, "page_size": ps}

    @app.callback(
        Output("heatmap-page-state", "data", allow_duplicate=True),
        Input("hm-page-prev", "n_clicks"),
        Input("hm-page-next", "n_clicks"),
        State("heatmap-page-state", "data"),
        prevent_initial_call=True,
    )
    def hm_paginate(n_prev, n_next, state):
        state = state or {"page": 1, "page_size": 50}
        page = int(state.get("page", 1))
        ps = int(state.get("page_size", 50))

        trig = ctx.triggered_id
        if trig == "hm-page-prev":
            page = max(1, page - 1)
        elif trig == "hm-page-next":
            page = page + 1

        return {"page": page, "page_size": ps}

    @app.callback(
        Output("hm-pct-dates", "children"),
        Output("hm-pct-hours", "children"),
        Output("hm-unit-dates", "children"),
        Output("hm-unit-hours", "children"),
        Input("f-fecha", "date"),
        prevent_initial_call=False,  # puebla al inicio
    )
    def update_time_headers(selected_date):
        dates_children, hours_children = _build_time_header_children_by_dates(selected_date)
        # % y UNIT comparten la misma línea temporal, por eso duplicamos
        return dates_children, hours_children, dates_children, hours_children
    # -------------------------------------------------
    # 8) Histograma render
    # -------------------------------------------------
    @app.callback(
        Output("hi-pct", "figure"),
        Output("hi-unit", "figure"),
        Output("histo-page-info", "data"),
        Input("histo-trigger", "data"),
        Input("histo-selected-wave", "data"),  # 👈 sólo wave
        State("f-fecha", "date"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("histo-page-state", "data"),
        State("kpi-domain", "value"),
        prevent_initial_call=True,
    )
    def refresh_histograma(_trigger, sel_wave, fecha, networks, technologies, vendors, clusters, hm_page_state, domain):
        global _LAST_HI_KEY

        selected_wave = (sel_wave or {}).get("series_key")

        # Normaliza filtros
        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        # Paginado
        page = int((hm_page_state or {}).get("page", 1))
        page_sz = int((hm_page_state or {}).get("page_size", 50))
        offset = max(0, (page - 1) * page_sz)
        limit = max(1, page_sz)

        # Clave de estado (sin selected_x)
        state_key = (
                _hm_key(fecha, networks, technologies, vendors, clusters, offset, limit)
                + f"|selw={selected_wave}|dom={domain}"
        )
        if _LAST_HI_KEY == state_key and ctx.triggered_id != "histo-selected-wave":
            return no_update, no_update, no_update

        # Fechas
        try:
            today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
        except Exception:
            today_dt = datetime.utcnow()
        yday_dt = today_dt - timedelta(days=1)
        today_str = today_dt.strftime("%Y-%m-%d")
        yday_str = yday_dt.strftime("%Y-%m-%d")

        # Datos
        df_ts = _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters)
        if networks:
            nets_heat = networks
        else:
            nets_heat = sorted(df_ts["network"].dropna().unique().tolist()) if (
                        df_ts is not None and not df_ts.empty and "network" in df_ts.columns) else []

        df_meta_heat, alarm_keys_set = fetch_alarm_meta_for_heatmap(
            fecha=today_str,
            vendors=vendors or None, clusters=clusters or None,
            networks=nets_heat or None, technologies=technologies or None,
        )

        if df_meta_heat is not None and not df_meta_heat.empty and nets_heat:
            traffic_metric = "ps_traff_gb" if str(domain).upper() != "CS" else "cs_traff_erl"
            pct_payload, unit_payload, page_info = build_histo_payloads_fast(
                df_meta=df_meta_heat,
                df_ts=df_ts,
                UMBRAL_CFG=UM_MANAGER.config(),
                networks=nets_heat,
                valores_order=_valores_by_domain(domain),
                today=today_str, yday=yday_str,
                alarm_keys=alarm_keys_set,
                alarm_only=False,
                offset=offset, limit=limit,
                traffic_metric=traffic_metric,
            )
        else:
            pct_payload = unit_payload = None
            page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

        # Figuras (sin selected_x)
        fig_pct = build_overlay_waves_figure(
            pct_payload, UMBRAL_CFG=UM_MANAGER.config(), mode="severity",
            height=420, smooth_win=3, opacity=0.28, line_width=1.2, decimals=2,
            show_yaxis_ticks=True, selected_wave=selected_wave, show_traffic_bars=True,
            traffic_agg="mean", traffic_decimals=1
        ) if pct_payload else go.Figure()

        fig_unit = build_overlay_waves_figure(
            unit_payload, UMBRAL_CFG=UM_MANAGER.config(), mode="progress",
            height=420, smooth_win=3, opacity=0.25, line_width=1.2, decimals=0,
            show_yaxis_ticks=True, selected_wave=selected_wave, show_traffic_bars=False,
        ) if unit_payload else go.Figure()

        _LAST_HI_KEY = state_key
        return fig_pct, fig_unit, page_info


    @app.callback(
        Output("histo-trigger", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("histo-page-state", "data"),  # dispara por paginado del heatmap
        Input("kpi-domain", "value"),
        prevent_initial_call=False,  # permite “bootstrap” al cargar
    )
    def histo_trigger_controller(_fecha, _net, _tech, _vend, _clus, _page_state, _dom):
        return {"ts": time.time()}


    @app.callback(
        Output("histo-page-state", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("hm-page-size", "value"),
        Input("kpi-domain", "value"),
        prevent_initial_call=False,  # bootstrap
    )
    def hi_reset_page_on_filters(_fecha, _net, _tech, _ven, _clu, hm_page_size, _dom):
        ps = max(1, int(hm_page_size or 50))
        return {"page": 1, "page_size": ps}

    @app.callback(
        Output("histo-page-state", "data", allow_duplicate=True),
        Input("hm-page-prev", "n_clicks"),
        Input("hm-page-next", "n_clicks"),
        State("histo-page-state", "data"),
        prevent_initial_call=True,
    )
    def hi_paginate(n_prev, n_next, state):
        state = state or {"page": 1, "page_size": 50}
        page = int(state.get("page", 1))
        ps = int(state.get("page_size", 50))

        trig = ctx.triggered_id
        if trig == "hm-page-prev":
            page = max(1, page - 1)
        elif trig == "hm-page-next":
            page = page + 1

        return {"page": page, "page_size": ps}

    # Click en % (hi-pct)
    @app.callback(
        Output("histo-selected-wave", "data"),
        Input("hi-pct", "clickData"),
        State("hi-pct", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_pct(clickData, fig):
        if not clickData or not fig:
            return no_update
        pt = (clickData.get("points") or [{}])[0]
        i = pt.get("curveNumber")
        traces = (fig or {}).get("data") or []
        if i is None or i >= len(traces):
            return no_update
        cd = traces[i].get("customdata")
        if not cd or not cd[0]:
            return no_update
        series_key = cd[0][0]  # 👈 primera columna del customdata
        return {"series_key": series_key}

    # Click en UNIT (hi-unit)
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-unit", "clickData"),
        State("hi-unit", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_unit(clickData, fig):
        if not clickData or not fig:
            return no_update
        pt = (clickData.get("points") or [{}])[0]
        i = pt.get("curveNumber")
        traces = (fig or {}).get("data") or []
        if i is None or i >= len(traces):
            return no_update
        cd = traces[i].get("customdata")
        if not cd or not cd[0]:
            return no_update
        series_key = cd[0][0]
        return {"series_key": series_key}

    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-pct", "relayoutData"),
        Input("hi-unit", "relayoutData"),
        prevent_initial_call=True,
    )
    def clear_wave_on_doubleclick(r1, r2):
        def is_autosize(r):
            # Plotly manda estas claves cuando haces doble cliick
            return bool(r) and (
                    r.get("autosize") is True
                    or r.get("xaxis.autorange") is True
                    or r.get("yaxis.autorange") is True
            )

        if is_autosize(r1) or is_autosize(r2):
            return {}  # ← limpia la selección (deselecciona la wave)
        return no_update

    @app.callback(
        Output("hi-unit", "figure", allow_duplicate=True),
        Input("hi-pct", "restyleData"),
        State("hi-unit", "figure"),
        State("hi-pct", "figure"),
        prevent_initial_call=True,
    )
    def sync_legend_from_pct_to_unit(restyle, unit_fig, pct_fig):
        # Si no hay interacción o figuras, no hacemos nada
        if not restyle or not unit_fig or not pct_fig:
            return no_update

        # restyle = [update_dict, [indices]]
        try:
            update, idxs = restyle
        except Exception:
            return no_update

        if "visible" not in update:
            # Nos interesa solo cuando cambia la visibilidad vía leyenda
            return no_update

        vis_update = update["visible"]
        # normaliza a lista
        if not isinstance(vis_update, (list, tuple)):
            vis_update = [vis_update] * len(idxs)

        # Copia mutable de la figura de UNIT
        new_unit_fig = unit_fig.copy()
        data_unit = new_unit_fig.get("data", [])
        data_pct = pct_fig.get("data", [])

        # Por cada traza afectada en hi-pct…
        for v, idx in zip(vis_update, idxs):
            if idx is None or idx >= len(data_pct):
                continue
            trace_pct = data_pct[idx]
            # cluster viene del legendgroup o del name
            cluster = trace_pct.get("legendgroup") or trace_pct.get("name")
            if not cluster:
                continue

            # …aplicamos la misma visibilidad a TODAS las trazas
            # del mismo cluster en hi-unit
            for t in data_unit:
                if t.get("legendgroup") == cluster or t.get("name") == cluster:
                    t["visible"] = v

        return new_unit_fig