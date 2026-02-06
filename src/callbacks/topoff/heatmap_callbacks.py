import os
import math
import time
import json
from hashlib import md5
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

# === Componentes TopOff heatmap ===
from components.topoff.heatmap import (
    build_heatmap_payloads_topoff,
    build_heatmap_figure_topoff,
    render_heatmap_summary_table_topoff,
    build_time_header_children_by_dates,
)

# === Componentes TopOff histo ===
from components.topoff.histograma import (
    build_histo_payloads_topoff,
    build_overlay_waves_figure_topoff,
)

from src.Utils.umbrales.umbrales_manager import UM_MANAGER

# === Data access TopOff ===
from src.dataAccess.data_acess_topoff import fetch_topoff_paginated
from src.dataAccess.data_acess_topoff import fetch_alarm_meta_for_topoff


# ======================================================
# Cache simple en memoria para df_ts TopOff
# ======================================================
_DFTS_TOPOFF_CACHE = {}
_DFTS_TOPOFF_TTL = 300  # seg (5 min)

# Última llave renderizada (evita re-render idéntico)
_LAST_TOPOFF_HEATMAP_KEY = None
_LAST_TOPOFF_HI_KEY = {"PS": None, "CS": None}

# Orden de “valores” que se muestran en heatmap/histo por dominio
TOPOFF_PS_VALORES = ("PS_RRC", "PS_DROP", "PS_RAB")
TOPOFF_CS_VALORES = ("CS_RRC", "CS_DROP", "CS_RAB")


def _as_list(x):
    """Convierte un valor (o None) a lista para que sea consistente en filtros."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _hm_key_topoff(fecha, technologies, vendors, clusters, sites, rncs, nodebs, offset, limit):
    """
    Crea una llave hash (md5) que representa el “estado” actual del heatmap TopOff.
    Si el estado no cambia, no re-renderizamos (optimización).
    """
    def _norm(x):
        # Normaliza: siempre lista, elimina None/"" y castea a string
        x = x if isinstance(x, (list, tuple)) else ([] if x is None else [x])
        return sorted([str(v) for v in x if v not in (None, "")])

    obj = {
        "fecha": fecha,
        "technologies": _norm(technologies),
        "vendors": _norm(vendors),
        "clusters": _norm(clusters),
        "site_att": _norm(sites),
        "rnc": _norm(rncs),
        "nodeb": _norm(nodebs),
        "offset": int(offset),
        "limit": int(limit),
    }
    return md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _ensure_df(x):
    """Garantiza un DataFrame válido aunque la función de data access regrese otra cosa."""
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()


def _fetch_topoff_all(fecha, technologies, vendors, clusters, sites, rncs, nodebs):
    """
    Trae “muchas” filas para series de tiempo (hoy/ayer) usando page_size grande.
    Luego aplica filtros finos (site/rnc/nodeb) aquí mismo.
    """
    df, _total = fetch_topoff_paginated(
        fecha=fecha,
        technologies=technologies or None,
        vendors=vendors or None,
        clusters=clusters or None,
        regions=None,
        provinces=None,
        municipalities=None,
        page=1,
        page_size=200000,  # ajusta si tu volumen es mayor
    )

    # Filtros finos (si vienen seleccionados)
    if sites and "site_att" in df.columns:
        df = df[df["site_att"].isin(sites)]
    if rncs and "rnc" in df.columns:
        df = df[df["rnc"].isin(rncs)]
    if nodebs and "nodeb" in df.columns:
        df = df[df["nodeb"].isin(nodebs)]

    return _ensure_df(df)


def _fetch_df_ts_topoff_cached(today_str, yday_str, technologies, vendors, clusters, sites, rncs, nodebs):
    """
    Construye df_ts = df(hoy) + df(ayer) (sin hora), cacheado por filtros.
    Esto alimenta heatmap/histo y evita pegarle a BD en cada repaint.
    """
    key = (
        "df_ts_topoff",
        today_str, yday_str,
        tuple(sorted(technologies or [])),
        tuple(sorted(vendors or [])),
        tuple(sorted(clusters or [])),
        tuple(sorted(sites or [])),
        tuple(sorted(rncs or [])),
        tuple(sorted(nodebs or [])),
    )
    now = time.time()
    hit = _DFTS_TOPOFF_CACHE.get(key)
    if hit and (now - hit["ts"] < _DFTS_TOPOFF_TTL):
        return hit["df"]

    df_today = _fetch_topoff_all(today_str, technologies, vendors, clusters, sites, rncs, nodebs)
    df_yday = _fetch_topoff_all(yday_str, technologies, vendors, clusters, sites, rncs, nodebs)

    df_ts = pd.concat([df_today, df_yday], ignore_index=True, sort=False)
    _DFTS_TOPOFF_CACHE[key] = {"df": df_ts, "ts": now}
    return df_ts


def _valores_by_domain_topoff(domain: str):
    """Selecciona el orden de KPIs (valores) según dominio PS/CS."""
    return TOPOFF_CS_VALORES if str(domain).upper() == "CS" else TOPOFF_PS_VALORES


def _run_topoff_histo_for_domain(
    domain: str,
    sel_wave,
    fecha,
    technologies,
    vendors,
    clusters,
    sites,
    rncs,
    nodebs,
    hi_page_state,
):
    """
    Construye y regresa:
      - fig_pct  (modo severidad)
      - fig_unit (modo progress)
      - page_info (paginación)
      - is_cache_hit (si evitamos recomputar)
    """
    global _LAST_TOPOFF_HI_KEY

    selected_wave = (sel_wave or {}).get("series_key")

    # Normaliza filtros
    technologies = _as_list(technologies)
    vendors = _as_list(vendors)
    clusters = _as_list(clusters)
    sites = _as_list(sites)
    rncs = _as_list(rncs)
    nodebs = _as_list(nodebs)

    # Paginado (usa el mismo state que el heatmap en tu código)
    page = int((hi_page_state or {}).get("page", 1))
    page_sz = int((hi_page_state or {}).get("page_size", 50))
    offset = max(0, (page - 1) * page_sz)
    limit = max(1, page_sz)

    dom = (domain or "PS").upper()

    # Llave de estado: filtros + paginado + wave seleccionada + dominio
    state_key = _hm_key_topoff(
        fecha, technologies, vendors, clusters, sites, rncs, nodebs, offset, limit
    ) + f"|selw={selected_wave}|dom={dom}"

    # Evita re-render idéntico (excepto si el trigger fue click a wave)
    if _LAST_TOPOFF_HI_KEY.get(dom) == state_key and ctx.triggered_id != "topoff-histo-selected-wave":
        return None, None, None, True  # cache hit

    # Fechas hoy/ayer
    try:
        today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
    except Exception:
        today_dt = datetime.utcnow()
    yday_dt = today_dt - timedelta(days=1)
    today_str = today_dt.strftime("%Y-%m-%d")
    yday_str = yday_dt.strftime("%Y-%m-%d")

    # Datos TS cacheados (ayer + hoy)
    df_ts = _fetch_df_ts_topoff_cached(
        today_str, yday_str,
        technologies, vendors, clusters,
        sites, rncs, nodebs
    )

    # Meta + set de alarm keys (para marcar alarmados)
    df_meta_topoff, alarm_keys_set = fetch_alarm_meta_for_topoff(
        fecha=today_str,
        technologies=technologies or None,
        vendors=vendors or None,
        clusters=clusters or None,
        site_atts=sites or None,
        rncs=rncs or None,
        nodebs=nodebs or None,
    )

    # Payloads para figuras
    if df_meta_topoff is not None and not df_meta_topoff.empty:
        pct_payload, unit_payload, page_info = build_histo_payloads_topoff(
            df_meta=df_meta_topoff,
            df_ts=df_ts,
            UMBRAL_CFG=UM_MANAGER.config(),
            valores_order=_valores_by_domain_topoff(dom),
            today=today_str, yday=yday_str,
            alarm_keys=alarm_keys_set,
            alarm_only=False,
            offset=offset,
            limit=limit,
            domain=dom,
        )
    else:
        pct_payload = unit_payload = None
        page_info = {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Figuras
    fig_pct = build_overlay_waves_figure_topoff(
        pct_payload,
        UMBRAL_CFG=UM_MANAGER.config(),
        mode="severity",
        height=420,
        smooth_win=3,
        opacity=0.9,
        line_width=1.2,
        decimals=2,
        show_yaxis_ticks=True,
        selected_wave=selected_wave,
        show_traffic_bars=False,
        traffic_agg="mean",
        traffic_decimals=1,
    ) if pct_payload else go.Figure()

    fig_unit = build_overlay_waves_figure_topoff(
        unit_payload,
        UMBRAL_CFG=UM_MANAGER.config(),
        mode="progress",
        height=420,
        smooth_win=3,
        opacity=0.9,
        line_width=1.2,
        decimals=0,
        show_yaxis_ticks=True,
        selected_wave=selected_wave,
        show_traffic_bars=False,
    ) if unit_payload else go.Figure()

    _LAST_TOPOFF_HI_KEY[dom] = state_key
    return fig_pct, fig_unit, page_info, False


# ======================================================
# Callbacks Heatmap + Histograma TopOff
# ======================================================
def topoff_heatmap_callbacks(app):

    # -------------------------------------------------
    # A) Render heatmap TopOff (tabla + 2 figs + banners)
    # -------------------------------------------------
    @app.callback(
        Output("topoff-hm-table-container", "children", allow_duplicate=True),
        Output("topoff-hm-pct", "figure"),
        Output("topoff-hm-unit", "figure"),
        Output("topoff-hm-page-indicator", "children", allow_duplicate=True),
        Output("topoff-hm-total-rows-banner", "children", allow_duplicate=True),
        Output("topoff-heatmap-page-info", "data"),
        Input("topoff-heatmap-trigger", "data"),
        State("f-fecha", "date"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),  # filtro global de cluster
        State("topoff-site-filter", "value"),
        State("topoff-rnc-filter", "value"),
        State("topoff-nodeb-filter", "value"),
        State("topoff-heatmap-page-state", "data"),
        State("topoff-hm-order-by", "value"),  # orden del heatmap
        prevent_initial_call=True,
    )
    def refresh_topoff_heatmaps(
        _trigger,
        fecha,
        technologies,
        vendors,
        clusters,
        sites,
        rncs,
        nodebs,
        hm_page_state,
        hm_order_by,
    ):
        """
        Renderiza:
          - tabla resumen
          - heatmap % (severity)
          - heatmap unit (progress)
          - indicador + banner
          - page_info (para paginado/altura)
        """
        global _LAST_TOPOFF_HEATMAP_KEY

        # Normaliza filtros
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)
        sites = _as_list(sites)
        rncs = _as_list(rncs)
        nodebs = _as_list(nodebs)

        # Paginado
        page = int((hm_page_state or {}).get("page", 1))
        page_sz = int((hm_page_state or {}).get("page_size", 50))
        offset = max(0, (page - 1) * page_sz)
        limit = max(1, page_sz)

        # Llave de estado (incluye order_by) para evitar recomputar si todo es igual
        state_key = _hm_key_topoff(
            fecha, technologies, vendors, clusters, sites, rncs, nodebs, offset, limit
        ) + f"|ord={hm_order_by or 'alarm_bins_pct'}"

        if _LAST_TOPOFF_HEATMAP_KEY == state_key:
            return (no_update, no_update, no_update, no_update, no_update, no_update)

        # Fechas hoy/ayer
        try:
            today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
        except Exception:
            today_dt = datetime.utcnow()

        yday_dt = today_dt - timedelta(days=1)
        today_str = today_dt.strftime("%Y-%m-%d")
        yday_str = yday_dt.strftime("%Y-%m-%d")

        # df_ts cacheado (ayer+hoy)
        df_ts = _fetch_df_ts_topoff_cached(
            today_str, yday_str,
            technologies, vendors, clusters,
            sites, rncs, nodebs
        )

        # Meta de alarmados (para ordenar y marcar alarmas)
        df_meta_topoff, alarm_keys_set = fetch_alarm_meta_for_topoff(
            fecha=today_str,
            technologies=technologies or None,
            vendors=vendors or None,
            clusters=clusters or None,
            site_atts=sites or None,
            rncs=rncs or None,
            nodebs=nodebs or None,
        )

        # Payloads del heatmap
        if df_meta_topoff is not None and not df_meta_topoff.empty:
            pct_payload, unit_payload, page_info = build_heatmap_payloads_topoff(
                df_meta=df_meta_topoff,
                df_ts=df_ts,
                UMBRAL_CFG=UM_MANAGER.config(),
                valores_order=("PS_RRC", "CS_RRC", "PS_DROP", "CS_DROP", "PS_RAB", "CS_RAB"),
                today=today_str, yday=yday_str,
                alarm_keys=alarm_keys_set,
                alarm_only=False,
                offset=offset,
                limit=limit,
                order_by=hm_order_by or "alarm_bins_pct",
            )
        else:
            pct_payload = unit_payload = None
            page_info = {
                "total_rows": 0,
                "offset": 0,
                "limit": limit,
                "showing": 0,
                "height": 300,
            }

        # Altura alineada a la cantidad de filas de la página
        hm_height = int(page_info.get("height") or 300)

        # Figuras
        fig_pct = (
            build_heatmap_figure_topoff(pct_payload, height=hm_height, decimals=2)
            if pct_payload else go.Figure(layout={"height": hm_height})
        )
        fig_unit = (
            build_heatmap_figure_topoff(unit_payload, height=hm_height, decimals=0)
            if unit_payload else go.Figure(layout={"height": hm_height})
        )

        # Tabla resumen
        if pct_payload or unit_payload:
            table_component = render_heatmap_summary_table_topoff(
                pct_payload, unit_payload, pct_decimals=2, unit_decimals=0
            )
        else:
            table_component = dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

        # Indicadores
        total = int(page_info.get("total_rows", 0))
        showing = int(page_info.get("showing", 0))
        start_i = int(page_info.get("offset", 0)) + 1 if showing else 0
        end_i = start_i + showing - 1 if showing else 0
        total_pg = max(1, math.ceil(total / max(1, page_sz)))

        hm_indicator = f"Página {page} de {total_pg}"
        hm_banner = "Sin filas." if total == 0 else f"Mostrando {start_i}–{end_i} de {total} filas"

        _LAST_TOPOFF_HEATMAP_KEY = state_key

        return (table_component, fig_pct, fig_unit, hm_indicator, hm_banner, page_info)

    # -------------------------------------------------
    # B) Trigger controller heatmap
    # -------------------------------------------------
    @app.callback(
        Output("topoff-heatmap-trigger", "data"),
        # Cualquier cambio en filtros/paginado/orden dispara un refresh
        Input("f-fecha", "date"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("topoff-site-filter", "value"),
        Input("topoff-rnc-filter", "value"),
        Input("topoff-nodeb-filter", "value"),
        Input("topoff-heatmap-page-state", "data"),
        Input("topoff-hm-order-by", "value"),
        prevent_initial_call=False,  # bootstrap
    )
    def topoff_heatmap_trigger_controller(*_):
        """Store “dummy” para disparar el callback grande sin loops raros."""
        return {"ts": time.time()}

    # -------------------------------------------------
    # C) Reset page heatmap en filtros / page-size / order
    # -------------------------------------------------
    @app.callback(
        Output("topoff-heatmap-page-state", "data"),
        # Si cambia cualquier filtro o el tamaño de página, regresamos a página 1
        Input("f-fecha", "date"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("topoff-site-filter", "value"),
        Input("topoff-rnc-filter", "value"),
        Input("topoff-nodeb-filter", "value"),
        Input("topoff-hm-page-size", "value"),
        Input("topoff-hm-order-by", "value"),
        prevent_initial_call=False,  # bootstrap
    )
    def topoff_hm_reset_page_on_filters(*args):
        """Resetea a página 1 y guarda page_size actual."""
        hm_page_size = args[-2]  # penúltimo input
        ps = max(1, int(hm_page_size or 50))
        return {"page": 1, "page_size": ps}

    # -------------------------------------------------
    # D) Prev/Next heatmap
    # -------------------------------------------------
    @app.callback(
        Output("topoff-heatmap-page-state", "data", allow_duplicate=True),
        Input("topoff-hm-page-prev", "n_clicks"),
        Input("topoff-hm-page-next", "n_clicks"),
        State("topoff-heatmap-page-state", "data"),
        prevent_initial_call=True,
    )
    def topoff_hm_paginate(n_prev, n_next, state):
        """Botones Prev/Next para paginar el heatmap."""
        state = state or {"page": 1, "page_size": 50}
        page = int(state.get("page", 1))
        ps = int(state.get("page_size", 50))

        trig = ctx.triggered_id
        if trig == "topoff-hm-page-prev":
            page = max(1, page - 1)
        elif trig == "topoff-hm-page-next":
            page = page + 1

        return {"page": page, "page_size": ps}

    # -------------------------------------------------
    # E) Time headers heatmap (línea de fechas/horas)
    # -------------------------------------------------
    @app.callback(
        Output("topoff-hm-pct-dates", "children"),
        Output("topoff-hm-pct-hours", "children"),
        Output("topoff-hm-unit-dates", "children"),
        Output("topoff-hm-unit-hours", "children"),
        Input("f-fecha", "date"),
        prevent_initial_call=False,  # bootstrap
    )
    def topoff_update_time_headers(selected_date):
        """Actualiza los headers de tiempo para los dos heatmaps (% y UNIT)."""
        dates_children, hours_children = build_time_header_children_by_dates(selected_date)
        return dates_children, hours_children, dates_children, hours_children

    # =================================================
    # F) Render HISTOGRAMA TopOff PS (2 figs + page_info)
    # =================================================
    @app.callback(
        Output("topoff-hi-pct-ps", "figure"),
        Output("topoff-hi-unit-ps", "figure"),
        Output("topoff-histo-page-info", "data"),
        Input("topoff-histo-trigger", "data"),
        Input("topoff-histo-selected-wave", "data"),
        State("f-fecha", "date"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("topoff-site-filter", "value"),
        State("topoff-rnc-filter", "value"),
        State("topoff-nodeb-filter", "value"),
        State("topoff-heatmap-page-state", "data"),
        prevent_initial_call=True,
    )
    def refresh_topoff_histograma_ps(
        _trigger,
        sel_wave,
        fecha,
        technologies,
        vendors,
        clusters,
        sites,
        rncs,
        nodebs,
        hi_page_state,
    ):
        """Construye histo PS (pct y unit) usando la misma paginación."""
        fig_pct, fig_unit, page_info, is_cache = _run_topoff_histo_for_domain(
            "PS",
            sel_wave,
            fecha,
            technologies,
            vendors,
            clusters,
            sites,
            rncs,
            nodebs,
            hi_page_state,
        )
        if is_cache:
            return no_update, no_update, no_update
        return fig_pct, fig_unit, page_info

    # =================================================
    # G) Render HISTOGRAMA TopOff CS (2 figs)
    # =================================================
    @app.callback(
        Output("topoff-hi-pct-cs", "figure"),
        Output("topoff-hi-unit-cs", "figure"),
        Input("topoff-histo-trigger", "data"),
        Input("topoff-histo-selected-wave", "data"),
        State("f-fecha", "date"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("topoff-site-filter", "value"),
        State("topoff-rnc-filter", "value"),
        State("topoff-nodeb-filter", "value"),
        State("topoff-heatmap-page-state", "data"),
        prevent_initial_call=True,
    )
    def refresh_topoff_histograma_cs(
        _trigger,
        sel_wave,
        fecha,
        technologies,
        vendors,
        clusters,
        sites,
        rncs,
        nodebs,
        hi_page_state,
    ):
        """Construye histo CS (pct y unit)."""
        fig_pct, fig_unit, _page_info, is_cache = _run_topoff_histo_for_domain(
            "CS",
            sel_wave,
            fecha,
            technologies,
            vendors,
            clusters,
            sites,
            rncs,
            nodebs,
            hi_page_state,
        )
        if is_cache:
            return no_update, no_update
        return fig_pct, fig_unit

    # -------------------------------------------------
    # J) Click en % PS -> selecciona wave
    # -------------------------------------------------
    @app.callback(
        Output("topoff-histo-selected-wave", "data"),
        Input("topoff-hi-pct-ps", "clickData"),
        State("topoff-hi-pct-ps", "figure"),
        prevent_initial_call=True,
    )
    def topoff_on_click_wave_pct_ps(clickData, fig):
        """Al dar click en una curva (% PS), guarda su series_key en el store."""
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
    # K) Click en UNIT PS -> selecciona wave
    # -------------------------------------------------
    @app.callback(
        Output("topoff-histo-selected-wave", "data", allow_duplicate=True),
        Input("topoff-hi-unit-ps", "clickData"),
        State("topoff-hi-unit-ps", "figure"),
        prevent_initial_call=True,
    )
    def topoff_on_click_wave_unit_ps(clickData, fig):
        """Al dar click en una curva (UNIT PS), guarda su series_key."""
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
    # L) Click en % CS -> selecciona wave
    # -------------------------------------------------
    @app.callback(
        Output("topoff-histo-selected-wave", "data", allow_duplicate=True),
        Input("topoff-hi-pct-cs", "clickData"),
        State("topoff-hi-pct-cs", "figure"),
        prevent_initial_call=True,
    )
    def topoff_on_click_wave_pct_cs(clickData, fig):
        """Al dar click en una curva (% CS), guarda su series_key."""
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
    # M) Click en UNIT CS -> selecciona wave
    # -------------------------------------------------
    @app.callback(
        Output("topoff-histo-selected-wave", "data", allow_duplicate=True),
        Input("topoff-hi-unit-cs", "clickData"),
        State("topoff-hi-unit-cs", "figure"),
        prevent_initial_call=True,
    )
    def topoff_on_click_wave_unit_cs(clickData, fig):
        """Al dar click en una curva (UNIT CS), guarda su series_key."""
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
    # L) Double-click (autosize) -> limpia selección
    # -------------------------------------------------
    @app.callback(
        Output("topoff-histo-selected-wave", "data", allow_duplicate=True),
        Input("topoff-hi-pct-ps", "relayoutData"),
        Input("topoff-hi-unit-ps", "relayoutData"),
        Input("topoff-hi-pct-cs", "relayoutData"),
        Input("topoff-hi-unit-cs", "relayoutData"),
        prevent_initial_call=True,
    )
    def topoff_clear_wave_on_doubleclick(r_ps_pct, r_ps_unit, r_cs_pct, r_cs_unit):
        """Si el usuario hace doble click (autosize), limpia la wave seleccionada."""
        def is_autosize(r):
            return bool(r) and (
                r.get("autosize") is True
                or r.get("xaxis.autorange") is True
                or r.get("yaxis.autorange") is True
            )

        if any(is_autosize(r) for r in [r_ps_pct, r_ps_unit, r_cs_pct, r_cs_unit]):
            return {}
        return no_update

    # -------------------------------------------------
    # Sync leyenda PS: si ocultas/mostrar una curva en %,
    # también la ocultamos/mostramos en UNIT.
    # -------------------------------------------------
    @app.callback(
        Output("topoff-hi-unit-ps", "figure", allow_duplicate=True),
        Input("topoff-hi-pct-ps", "restyleData"),
        State("topoff-hi-unit-ps", "figure"),
        State("topoff-hi-pct-ps", "figure"),
        prevent_initial_call=True,
    )
    def sync_legend_from_pct_to_unit_ps(restyle, unit_fig, pct_fig):
        """Copia visibilidad de curvas del chart % al chart UNIT (PS)."""
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

    # -------------------------------------------------
    # Sync leyenda CS: % -> UNIT
    # -------------------------------------------------
    @app.callback(
        Output("topoff-hi-unit-cs", "figure", allow_duplicate=True),
        Input("topoff-hi-pct-cs", "restyleData"),
        State("topoff-hi-unit-cs", "figure"),
        State("topoff-hi-pct-cs", "figure"),
        prevent_initial_call=True,
    )
    def sync_legend_from_pct_to_unit_cs(restyle, unit_fig, pct_fig):
        """Copia visibilidad de curvas del chart % al chart UNIT (CS)."""
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

    # -------------------------------------------------
    # Trigger controller histo (para refrescar PS/CS)
    # -------------------------------------------------
    @app.callback(
        Output("topoff-histo-trigger", "data"),
        Input("f-fecha", "date"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("topoff-site-filter", "value"),
        Input("topoff-rnc-filter", "value"),
        Input("topoff-nodeb-filter", "value"),
        Input("topoff-heatmap-page-state", "data"),  # paginado compartido
        prevent_initial_call=False,  # bootstrap
    )
    def topoff_histo_trigger_controller(*_):
        """Store dummy que dispara el refresh de histogramas."""
        return {"ts": time.time()}
