
import pandas as pd
from dash import Input, Output, State, no_update, ctx
import time
import plotly.graph_objs as go
import math
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from components.main.heatmap import build_heatmap_figure,\
    _hm_height, add_integrity_deg_pct, _build_time_header_children_by_dates
from components.main.integrity_heatmap import build_integrity_heatmap_payloads_fast, render_integrity_summary_table
from src.callbacks.main.heatmap_callbacks import _as_list, _fetch_df_ts_cached
from src.dataAccess.data_access import fetch_integrity_baseline_week
##HELPERS##
def _baseline_map_from_df(df_bl: pd.DataFrame) -> dict:
    if df_bl is None or df_bl.empty:
        return {}
    # columnas esperadas: network, vendor, noc_cluster, technology, integrity_week_avg
    return {
        (str(r.network).strip(), str(r.vendor).strip(), str(r.noc_cluster).strip(), str(r.technology).strip()): r.integrity_week_avg
        for r in df_bl.itertuples(index=False)
    }
##Integrity heatmap##
def integrity_callbacks(app):
    @app.callback(
        Output("hm-int-table-container", "children"),
        Output("hm-int-pct", "figure"),
        Output("hm-int-unit", "figure"),
        Output("hm-int-page-indicator", "children"),
        Output("hm-int-total-rows-banner", "children"),
        Output("heatmap-integrity-page-info", "data"),
        Input("heatmap-integrity-trigger", "data"),
        State("collapse-hm-int", "is_open"),
        State("f-fecha", "date"),
        State("f-network", "value"),
        State("f-technology", "value"),
        State("f-vendor", "value"),
        State("f-cluster", "value"),
        State("heatmap-integrity-page-state", "data"),
        State("main-context-store", "data"),
        prevent_initial_call=True,
    )
    def refresh_integrity_heatmap(_trigger, is_open, fecha, networks, technologies, vendors, clusters, page_state,
                                  main_ctx):
        if not is_open:
            return no_update, no_update, no_update, no_update, no_update, no_update

        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        page = int((page_state or {}).get("page", 1))
        page_sz = int((page_state or {}).get("page_size", 50))
        offset = max(0, (page - 1) * page_sz)
        limit = max(1, page_sz)

        # fechas hoy/ayer
        try:
            today_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
        except Exception:
            today_dt = datetime.utcnow()
        yday_dt = today_dt - timedelta(days=1)
        today_str = today_dt.strftime("%Y-%m-%d")
        yday_str = yday_dt.strftime("%Y-%m-%d")

        # df_ts (ayer+hoy) SIN hora (como tus heatmaps)
        df_ts = _fetch_df_ts_cached(today_str, yday_str, networks, technologies, vendors, clusters)
        df_ts = df_ts if isinstance(df_ts, pd.DataFrame) else pd.DataFrame()
        if df_ts.empty:
            return dbc.Alert("Sin filas para mostrar.", color="secondary",
                             className="mb-0"), go.Figure(), go.Figure(), "Página 1 de 1", "Sin filas.", {
                "total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

        # normaliza llaves
        for c in ["network", "vendor", "noc_cluster", "technology"]:
            if c in df_ts.columns:
                df_ts[c] = df_ts[c].astype(str).str.strip()

        # baseline map desde store (y fallback a BD si viene vacío)
        main_ctx = main_ctx or {}
        integrity_baseline_list = main_ctx.get("integrity_baseline_map") or []
        integrity_baseline_map = {
            (str(d.get("network", "")).strip(),
             str(d.get("vendor", "")).strip(),
             str(d.get("noc_cluster", "")).strip(),
             str(d.get("technology", "")).strip()
             ): d.get("integrity_week_avg")
            for d in integrity_baseline_list
            if d is not None
        }

        if not integrity_baseline_map:
            df_bl = fetch_integrity_baseline_week(
                fecha=today_str,
                vendors=vendors or None,
                clusters=clusters or None,
                networks=networks or None,
                technologies=technologies or None,
            )
            integrity_baseline_map = _baseline_map_from_df(df_bl)  # <-- define este helper como tú lo traías

        # columna calculada
        df_ts = add_integrity_deg_pct(df_ts, integrity_baseline_map)

        # filtro >= 80% (regla de negocio)
        if "integrity_deg_pct" in df_ts.columns:
            s = pd.to_numeric(df_ts["integrity_deg_pct"], errors="coerce")
            df_ts = df_ts.loc[s >= 80.0].copy()

        if df_ts.empty:
            return dbc.Alert("Sin filas (>=80%) para mostrar.", color="secondary",
                             className="mb-0"), go.Figure(), go.Figure(), "Página 1 de 1", "Sin filas.", {
                "total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

        # networks efectivas (si no eligieron, usa las presentes)
        nets_heat = networks or (
            sorted(df_ts["network"].dropna().unique().tolist()) if "network" in df_ts.columns else []
        )

        # meta total (tríos) + paginado PROPIO
        df_meta_all = (
            df_ts[["noc_cluster", "technology", "vendor"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["noc_cluster", "technology", "vendor"], kind="mergesort")
            .reset_index(drop=True)
        )

        total = int(len(df_meta_all))
        df_meta_page = df_meta_all.iloc[offset:offset + limit].reset_index(drop=True)
        showing = int(len(df_meta_page))

        page_info = {"total_rows": total, "offset": offset, "limit": limit, "showing": showing}

        if df_meta_page.empty:
            total_pg = max(1, math.ceil(total / max(1, page_sz)))
            indicator = f"Página {page} de {total_pg}"
            banner = "Sin filas."
            return dbc.Alert("Sin filas para esta página.", color="secondary",
                             className="mb-0"), go.Figure(), go.Figure(), indicator, banner, page_info

        # ===== Payloads (aquí usa tu builder) =====
        page_limit = len(df_meta_page)
        pct_payload, unit_payload, page_info = build_integrity_heatmap_payloads_fast(
            df_meta=df_meta_all,  # <-- TOTAL, no page
            df_ts=df_ts,
            networks=nets_heat,
            today=today_str, yday=yday_str,
            min_pct_ok=80.0,
            offset=offset,  # <-- offset real de paginación
            limit=limit,  # <-- page_size real (50)
        )

        nrows = len((pct_payload or unit_payload or {}).get("y") or [])
        hm_height = _hm_height(nrows)

        # Figuras
        fig_pct = build_heatmap_figure(pct_payload, height=hm_height, decimals=2) if pct_payload else go.Figure()
        fig_unit = build_heatmap_figure(unit_payload, height=hm_height, decimals=0) if unit_payload else go.Figure()

        # Tabla resumen alineada a la misma página
        table_component = render_integrity_summary_table(
            df_ts=df_ts,
            pct_payload=pct_payload,
            nets_heat=nets_heat,
        )

        total_pg = max(1, math.ceil(total / max(1, page_sz)))
        indicator = f"Página {page} de {total_pg}"

        start_i = offset + 1 if showing else 0
        end_i = offset + showing if showing else 0
        banner = "Sin filas." if total == 0 else f"Mostrando {start_i}–{end_i} de {total} filas"

        return table_component, fig_pct, fig_unit, indicator, banner, page_info

    @app.callback(
        Output("heatmap-integrity-trigger", "data"),
        Input("collapse-hm-int", "is_open"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("heatmap-integrity-page-state", "data"),
        Input("main-context-store", "data"),
        prevent_initial_call=False,
    )
    def integrity_trigger_controller(is_open, *_):
        if not is_open:
            return no_update
        return {"ts": time.time()}

    @app.callback(
        Output("collapse-hm-int", "is_open"),
        Input("btn-toggle-hm-int", "n_clicks"),
        State("collapse-hm-int", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_integrity_panel(n, is_open):
            if not n:
                return is_open
            return not is_open

    @app.callback(
        Output("heatmap-integrity-page-state", "data"),
        Input("f-fecha", "date"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("hm-int-page-size", "value"),
        prevent_initial_call=False,
    )
    def hm_int_reset_page_on_filters(_fecha, _net, _tech, _ven, _clu, page_size):
        ps = max(10, int(page_size or 50))
        return {"page": 1, "page_size": ps}

    @app.callback(
        Output("heatmap-integrity-page-state", "data", allow_duplicate=True),
        Input("hm-int-page-prev", "n_clicks"),
        Input("hm-int-page-next", "n_clicks"),
        State("heatmap-integrity-page-state", "data"),
        prevent_initial_call=True,
    )
    def hm_int_paginate(n_prev, n_next, state):
        state = state or {"page": 1, "page_size": 50}
        page = int(state.get("page", 1))
        ps = int(state.get("page_size", 50))

        trig = ctx.triggered_id
        if trig == "hm-int-page-prev":
            page = max(1, page - 1)
        elif trig == "hm-int-page-next":
            page = page + 1

        return {"page": page, "page_size": ps}

    @app.callback(
        Output("hm-int-pct-dates", "children"),
        Output("hm-int-pct-hours", "children"),
        Output("hm-int-unit-dates", "children"),
        Output("hm-int-unit-hours", "children"),
        Input("f-fecha", "date"),
        prevent_initial_call=False,
    )
    def update_time_headers_integrity(selected_date):
        dates_children, hours_children = _build_time_header_children_by_dates(selected_date)
        return dates_children, hours_children, dates_children, hours_children