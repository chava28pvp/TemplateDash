import os
import math
import pandas as pd
from dash import Input, Output, State, no_update, ctx
from hashlib import md5
import json
import time
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Helpers de Heatmap (figuras + payloads + UI)
from components.main.heatmap import (
    build_heatmap_figure,
    render_heatmap_summary_table,
    build_heatmap_payloads_fast,
    _hm_height,
    _build_time_header_children_by_dates
)

# Helpers de Histograma (payloads + overlay waves)
from components.main.histograma import (
    build_histo_payloads_fast,
    build_overlay_waves_figure
)

import dash_bootstrap_components as dbc

# Manager de umbrales (config centralizada)
from src.Utils.umbrales.umbrales_manager import UM_MANAGER

# Acceso a datos
from src.dataAccess.data_access import fetch_kpis, fetch_alarm_meta_for_heatmap


# =========================================================
# CACHES EN MEMORIA
# =========================================================

# Cache simple para df_ts (datos de hoy + ayer) por filtros
_DFTS_CACHE = {}
_DFTS_TTL = 300  # segundos

# Última clave renderizada para evitar re-render idéntico (heatmap)
_LAST_HEATMAP_KEY = None

# Última clave renderizada para evitar re-render idéntico (histograma PS/CS)
_LAST_HI_KEY = {"PS": None, "CS": None}

# Cache simple en memoria para meta de alarmados (df_meta + keys)
_ALARM_META_CACHE = {}
_ALARM_META_TTL = 300  # seg

# Cache de payloads del heatmap por state_key (para no recalcular z/y/x cada rato)
_HM_PAYLOAD_CACHE = {}
_HM_PAYLOAD_TTL = 120  # seg


# =========================================================
# MÉTRICAS POR DOMINIO
# =========================================================
PS_VALORES = ("PS_RRC", "PS_S1", "PS_DROP", "PS_RAB")
CS_VALORES = ("CS_RRC", "CS_DROP", "CS_RAB")


# =========================================================
# HELPERS GENERALES
# =========================================================

def _hm_key(fecha, networks, technologies, vendors, clusters, offset, limit):
    """
    Genera una "firma" (hash md5) del estado actual del heatmap:
    - filtros
    - paginación (offset/limit)
    Esto permite cachear y también evitar renders duplicados.
    """
    def _norm(x):
        # Normaliza a lista de strings ordenada (estable)
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
    """Garantiza que lo que regrese sea DataFrame."""
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()


def _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters):
    """
    Trae el dataset de serie de tiempo (TS) del día de HOY y AYER:
    - fetch_kpis(... hoy ...)
    - fetch_kpis(... ayer ...)
    - concatena ambos
    Se cachea por filtros para ahorrar consultas/cálculo.
    """
    key = (
        "df_ts",
        today_str,
        yday_str,
        tuple(sorted(networks or [])),
        tuple(sorted(technologies or [])),
        tuple(sorted(vendors or [])),
        tuple(sorted(clusters or []))
    )
    now = time.time()
    hit = _DFTS_CACHE.get(key)
    if hit and (now - hit["ts"] < _DFTS_TTL):
        return hit["df"]

    # Hoy (día completo: hora=None)
    df_today = fetch_kpis(
        fecha=today_str,
        hora=None,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        limit=None
    )
    df_today = _ensure_df(df_today)

    # Ayer (día completo: hora=None)
    df_yday = fetch_kpis(
        fecha=yday_str,
        hora=None,
        vendors=vendors or None,
        clusters=clusters or None,
        networks=networks or None,
        technologies=technologies or None,
        limit=None
    )
    df_yday = _ensure_df(df_yday)

    # Serie de tiempo final (hoy + ayer)
    df_ts = pd.concat([df_today, df_yday], ignore_index=True, sort=False)

    # Guardar en cache
    _DFTS_CACHE[key] = {"df": df_ts, "ts": now}
    return df_ts


def _as_list(x):
    """Normaliza a lista:
    - None -> None
    - list/tuple -> list
    - valor -> [valor]
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _cache_get(cache: dict, key, ttl: int):
    """Lee un cache con TTL."""
    now = time.time()
    hit = cache.get(key)
    if hit and (now - hit["ts"] < ttl):
        return hit["data"]
    return None


def _cache_set(cache: dict, key, data):
    """Guarda en cache con timestamp."""
    cache[key] = {"data": data, "ts": time.time()}


def _fetch_alarm_meta_cached(today_str, vendors, clusters, networks, technologies):
    """
    Trae (df_meta, keys) para alarmados en el heatmap:
    - df_meta: metadata para filas (sitio/rnc/nodeb/etc)
    - keys: set/keys para saber cuáles están alarmados
    Cacheado por filtros.
    """
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
    """Devuelve qué métricas se usan según dominio PS o CS."""
    return CS_VALORES if str(domain).upper() == "CS" else PS_VALORES


def _build_histograma_for_domain(
    domain: str,
    sel_wave,
    fecha,
    networks,
    technologies,
    vendors,
    clusters,
    hm_page_state,
    link_state,
):
    """
    Construye 2 figuras (pct y unit) del histograma overlay para el dominio:
    - PS: %IA/%DC + Units
    - CS: %IA/%DC + Units
    Aplica:
    - filtros normales
    - paginado (mismo del heatmap/histo)
    - filtro extra de "link_state" (click desde main para fijar cluster/vendor/tech)
    """
    global _LAST_HI_KEY

    # selected_wave viene de click en histograma (series_key)
    selected_wave = (sel_wave or {}).get("series_key")

    # Normaliza filtros
    networks = _as_list(networks)
    technologies = _as_list(technologies)
    vendors = _as_list(vendors)
    clusters = _as_list(clusters)

    # -------- Filtro extra desde MAIN (link_state) --------
    clusters_effective = clusters
    vendors_effective = vendors
    technologies_effective = technologies

    if link_state and link_state.get("selected"):
        sel = link_state["selected"]
        clus = sel.get("cluster")
        ven = sel.get("vendor")
        tech = sel.get("technology")

        # Si viene alguno, “fija” ese filtro
        if clus:
            clusters_effective = [clus]
        if ven:
            vendors_effective = [ven]
        if tech:
            technologies_effective = [tech]

    # -------- Paginado (usa histo/heatmap state) --------
    page = int((hm_page_state or {}).get("page", 1))
    page_sz = int((hm_page_state or {}).get("page_size", 50))
    offset = max(0, (page - 1) * page_sz)
    limit = max(1, page_sz)

    # -------- Clave de estado: cambia si cambia filtro/página/selección/dominio --------
    state_key = (
        _hm_key(
            fecha,
            networks,
            technologies_effective,
            vendors_effective,
            clusters_effective,
            offset,
            limit,
        )
        + f"|selw={selected_wave}|dom={domain}"
    )

    # Evita recomputar si es exactamente lo mismo (y no fue click de wave)
    if _LAST_HI_KEY.get(domain) == state_key and ctx.triggered_id != "histo-selected-wave":
        return None, None, None, True  # cache-hit: el callback que llama usa no_update

    # -------- Fechas HOY/AYER --------
    try:
        today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
    except Exception:
        today_dt = datetime.utcnow()

    yday_dt = today_dt - timedelta(days=1)
    today_str = today_dt.strftime("%Y-%m-%d")
    yday_str = yday_dt.strftime("%Y-%m-%d")

    # -------- Datos TS (cacheados) --------
    df_ts = _fetch_df_ts_cached(
        today_str, yday_str,
        networks,
        technologies_effective,
        vendors_effective,
        clusters_effective,
    )

    # -------- Redes efectivas para el heat/histo --------
    if networks:
        nets_heat = networks
    else:
        nets_heat = sorted(df_ts["network"].dropna().unique().tolist()) if (
            df_ts is not None and not df_ts.empty and "network" in df_ts.columns
        ) else []

    # -------- Meta para heat/histo (alarmados) --------
    # Nota: aquí usas fetch directo (si quieres, puedes cambiarlo por _fetch_alarm_meta_cached)
    df_meta_heat, alarm_keys_set = fetch_alarm_meta_for_heatmap(
        fecha=today_str,
        vendors=vendors_effective or None,
        clusters=clusters_effective or None,
        networks=nets_heat or None,
        technologies=technologies_effective or None,
    )

    # -------- Payloads para overlay --------
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

    # -------- Figuras overlay (sin selected_x) --------
    fig_pct = build_overlay_waves_figure(
        pct_payload,
        UMBRAL_CFG=UM_MANAGER.config(),
        mode="severity",
        height=420,
        smooth_win=3,
        opacity=0.28,
        line_width=1.2,
        decimals=2,
        show_yaxis_ticks=True,
        selected_wave=selected_wave,
        show_traffic_bars=True,
        traffic_agg="mean",
        traffic_decimals=1
    ) if pct_payload else go.Figure()

    fig_unit = build_overlay_waves_figure(
        unit_payload,
        UMBRAL_CFG=UM_MANAGER.config(),
        mode="progress",
        height=420,
        smooth_win=3,
        opacity=0.25,
        line_width=1.2,
        decimals=0,
        show_yaxis_ticks=True,
        selected_wave=selected_wave,
        show_traffic_bars=False,
    ) if unit_payload else go.Figure()

    # Marca el estado como “ya renderizado”
    _LAST_HI_KEY[domain] = state_key
    return fig_pct, fig_unit, page_info, False


def heatmap_callbacks(app):

    # -------------------------------------------------
    # 8) Render HEATMAP (tabla + figuras + paginado)
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
        global _LAST_HEATMAP_KEY

        # Normaliza filtros
        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        # -------- Paginación HEATMAP --------
        page = int((hm_page_state or {}).get("page", 1))
        page_sz = int((hm_page_state or {}).get("page_size", 50))
        offset = max(0, (page - 1) * page_sz)
        limit = max(1, page_sz)

        # Normaliza modo de orden
        hm_order_by_norm = (hm_order_by or "alarm_hours")
        hm_order_by_norm = str(hm_order_by_norm).strip().lower()

        # State key incluye filtros + página + orden
        state_key = _hm_key(fecha, networks, technologies, vendors, clusters, offset, limit) + f"|ord={hm_order_by_norm}"

        # Si no cambió nada, no re-renderiza
        if _LAST_HEATMAP_KEY == state_key:
            return (no_update, no_update, no_update, no_update, no_update, no_update)

        # -------- Fechas HOY/AYER (sin hora) --------
        try:
            today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
        except Exception:
            today_dt = datetime.utcnow()

        yday_dt = today_dt - timedelta(days=1)
        today_str = today_dt.strftime("%Y-%m-%d")
        yday_str = yday_dt.strftime("%Y-%m-%d")

        # -------- Datos TS (cacheados) --------
        df_ts = _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters)

        # -------- Redes efectivas --------
        if networks:
            nets_heat = networks
        else:
            nets_heat = sorted(df_ts["network"].dropna().unique().tolist()) \
                if not df_ts.empty and "network" in df_ts.columns else []

        # -------- Meta alarmados (cacheada) --------
        df_meta_heat, alarm_keys_set = _fetch_alarm_meta_cached(
            today_str,
            vendors,
            clusters,
            nets_heat,
            technologies
        )

        # -------- Payloads (cacheados por state_key) --------
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
                    order_by=hm_order_by_norm
                )
            else:
                pct_payload = unit_payload = None
                page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

            _cache_set(_HM_PAYLOAD_CACHE, state_key, (pct_payload, unit_payload, page_info))

        # -------- Altura del heatmap (para que cuadre con #filas) --------
        nrows = len((pct_payload or unit_payload or {}).get("y") or [])
        hm_height = _hm_height(nrows)

        # -------- Figuras --------
        fig_pct = build_heatmap_figure(pct_payload, height=hm_height, decimals=2) if pct_payload else go.Figure()
        fig_unit = build_heatmap_figure(unit_payload, height=hm_height, decimals=0) if unit_payload else go.Figure()

        # -------- Tabla resumen --------
        if pct_payload or unit_payload:
            table_component = render_heatmap_summary_table(
                pct_payload, unit_payload, pct_decimals=2, unit_decimals=0
            )
        else:
            table_component = dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

        # -------- Indicadores de paginación --------
        total = int(page_info.get("total_rows", 0))
        showing = int(page_info.get("showing", 0))
        start_i = int(page_info.get("offset", 0)) + 1 if showing else 0
        end_i = start_i + showing - 1 if showing else 0
        total_pg = max(1, math.ceil(total / max(1, page_sz)))

        hm_indicator = f"Página {page} de {total_pg}"
        hm_banner = "Sin filas." if total == 0 else f"Mostrando {start_i}–{end_i} de {total} filas"

        # Marca renderizado
        _LAST_HEATMAP_KEY = state_key
        return (
            table_component,
            fig_pct,
            fig_unit,
            hm_indicator,
            hm_banner,
            page_info
        )

    # -------------------------------------------------
    # Controlador: dispara el “trigger” cuando cambian filtros/página/orden
    # (así el callback pesado solo depende de heatmap-trigger)
    # -------------------------------------------------
    @app.callback(
        Output("heatmap-trigger", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("heatmap-page-state", "data"),
        Input("hm-order-by", "value"),
        prevent_initial_call=False,  # bootstrap al cargar
    )
    def heatmap_trigger_controller(_fecha, _net, _tech, _vend, _clus, _page_state, _ord):
        # Un timestamp basta para “forzar” la actualización
        return {"ts": time.time()}

    # -------------------------------------------------
    # Reset de paginación del heatmap cuando cambian filtros/tamaño/orden
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Botones prev/next del heatmap
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Encabezados de tiempo (dates/hours) arriba del heatmap
    # -------------------------------------------------
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
        # % y UNIT comparten la misma línea temporal
        return dates_children, hours_children, dates_children, hours_children

    # -------------------------------------------------
    # 8) Histograma PS (figuras + page_info)
    # -------------------------------------------------
    @app.callback(
        Output("hi-pct-ps", "figure"),
        Output("hi-unit-ps", "figure"),
        Output("histo-page-info", "data"),
        Input("histo-trigger", "data"),
        Input("histo-selected-wave", "data"),
        State("f-fecha", "date"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("histo-page-state", "data"),
        State("topoff-link-state", "data"),
        prevent_initial_call=True,
    )
    def refresh_histograma_ps(_trigger, sel_wave, fecha, networks, technologies, vendors, clusters, hm_page_state, link_state):
        fig_pct, fig_unit, page_info, is_cache_hit = _build_histograma_for_domain(
            "PS", sel_wave, fecha, networks, technologies, vendors, clusters, hm_page_state, link_state
        )
        # Si fue “cache hit” no actualizamos nada
        if is_cache_hit:
            return no_update, no_update, no_update
        return fig_pct, fig_unit, page_info

    # -------------------------------------------------
    # Histograma CS (figuras)
    # -------------------------------------------------
    @app.callback(
        Output("hi-pct-cs", "figure"),
        Output("hi-unit-cs", "figure"),
        Input("histo-trigger", "data"),
        Input("histo-selected-wave", "data"),
        State("f-fecha", "date"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("histo-page-state", "data"),
        State("topoff-link-state", "data"),
        prevent_initial_call=True,
    )
    def refresh_histograma_cs(_trigger, sel_wave, fecha, networks, technologies, vendors, clusters, hm_page_state, link_state):
        fig_pct, fig_unit, _page_info, is_cache_hit = _build_histograma_for_domain(
            "CS", sel_wave, fecha, networks, technologies, vendors, clusters, hm_page_state, link_state
        )
        if is_cache_hit:
            return no_update, no_update
        return fig_pct, fig_unit

    # -------------------------------------------------
    # Trigger del histograma (para re-render cuando cambian filtros/página/link)
    # -------------------------------------------------
    @app.callback(
        Output("histo-trigger", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("histo-page-state", "data"),
        Input("topoff-link-state", "data"),
        prevent_initial_call=False,
    )
    def histo_trigger_controller(_fecha, _net, _tech, _vend, _clus, _page_state, _link_state):
        return {"ts": time.time()}

    # -------------------------------------------------
    # Reset de paginación de histograma cuando cambian filtros/tamaño/link
    # (OJO: aquí usas hm-page-size y botones hm-page-prev/next)
    # -------------------------------------------------
    @app.callback(
        Output("histo-page-state", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("hm-page-size", "value"),
        Input("topoff-link-state", "data"),
        prevent_initial_call=False,
    )
    def hi_reset_page_on_filters(_fecha, _net, _tech, _ven, _clu, hm_page_size, _link_state):
        ps = max(1, int(hm_page_size or 50))
        return {"page": 1, "page_size": ps}

    # -------------------------------------------------
    # Paginación del histograma (usa botones del heatmap)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Click: seleccionar una wave (PS %)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data"),
        Input("hi-pct-ps", "clickData"),
        State("hi-pct-ps", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_pct_ps(clickData, fig):
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

    # -------------------------------------------------
    # Click: seleccionar una wave (PS UNIT)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-unit-ps", "clickData"),
        State("hi-unit-ps", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_unit_ps(clickData, fig):
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

    # -------------------------------------------------
    # Click: seleccionar una wave (CS %)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-pct-cs", "clickData"),
        State("hi-pct-cs", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_pct_cs(clickData, fig):
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

    # -------------------------------------------------
    # Click: seleccionar una wave (CS UNIT)
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-unit-cs", "clickData"),
        State("hi-unit-cs", "figure"),
        prevent_initial_call=True,
    )
    def on_click_wave_unit_cs(clickData, fig):
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

    # -------------------------------------------------
    # Doble click / autoscale: limpiar selección de wave
    # -------------------------------------------------
    @app.callback(
        Output("histo-selected-wave", "data", allow_duplicate=True),
        Input("hi-pct-ps", "relayoutData"),
        Input("hi-unit-ps", "relayoutData"),
        Input("hi-pct-cs", "relayoutData"),
        Input("hi-unit-cs", "relayoutData"),
        prevent_initial_call=True,
    )
    def clear_wave_on_doubleclick(r_ps_pct, r_ps_unit, r_cs_pct, r_cs_unit):
        def is_autosize(r):
            return bool(r) and (
                r.get("autosize") is True
                or r.get("xaxis.autorange") is True
                or r.get("yaxis.autorange") is True
            )

        # Si alguno hizo autoscale -> limpiamos selección
        if any(is_autosize(r) for r in [r_ps_pct, r_ps_unit, r_cs_pct, r_cs_unit]):
            return {}
        return no_update

    # -------------------------------------------------
    # Sync de legend: PS (lo que ocultas en % también se oculta en UNIT)
    # -------------------------------------------------
    @app.callback(
        Output("hi-unit-ps", "figure", allow_duplicate=True),
        Input("hi-pct-ps", "restyleData"),
        State("hi-unit-ps", "figure"),
        State("hi-pct-ps", "figure"),
        prevent_initial_call=True,
    )
    def sync_legend_from_pct_to_unit_ps(restyle, unit_fig, pct_fig):
        if not restyle or not unit_fig or not pct_fig:
            return no_update
        try:
            update, idxs = restyle
        except Exception:
            return no_update
        if "visible" not in update:
            return no_update

        vis_update = update["visible"]
        if not isinstance(vis_update, (list, tuple)):
            vis_update = [vis_update] * len(idxs)

        new_unit_fig = unit_fig.copy()
        data_unit = new_unit_fig.get("data", [])
        data_pct = pct_fig.get("data", [])

        # Copia el "visible" de la traza en % hacia la traza equivalente en UNIT
        for v, idx in zip(vis_update, idxs):
            if idx is None or idx >= len(data_pct):
                continue
            trace_pct = data_pct[idx]
            cluster = trace_pct.get("legendgroup") or trace_pct.get("name")
            if not cluster:
                continue

            for t in data_unit:
                if t.get("legendgroup") == cluster or t.get("name") == cluster:
                    t["visible"] = v

        return new_unit_fig

    # -------------------------------------------------
    # Sync de legend: CS
    # -------------------------------------------------
    @app.callback(
        Output("hi-unit-cs", "figure", allow_duplicate=True),
        Input("hi-pct-cs", "restyleData"),
        State("hi-unit-cs", "figure"),
        State("hi-pct-cs", "figure"),
        prevent_initial_call=True,
    )
    def sync_legend_from_pct_to_unit_cs(restyle, unit_fig, pct_fig):
        if not restyle or not unit_fig or not pct_fig:
            return no_update
        try:
            update, idxs = restyle
        except Exception:
            return no_update
        if "visible" not in update:
            return no_update

        vis_update = update["visible"]
        if not isinstance(vis_update, (list, tuple)):
            vis_update = [vis_update] * len(idxs)

        new_unit_fig = unit_fig.copy()
        data_unit = new_unit_fig.get("data", [])
        data_pct = pct_fig.get("data", [])

        for v, idx in zip(vis_update, idxs):
            if idx is None or idx >= len(data_pct):
                continue
            trace_pct = data_pct[idx]
            cluster = trace_pct.get("legendgroup") or trace_pct.get("name")
            if not cluster:
                continue

            for t in data_unit:
                if t.get("legendgroup") == cluster or t.get("name") == cluster:
                    t["visible"] = v

        return new_unit_fig
