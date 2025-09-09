# callbacks.py (o donde declares tus callbacks)
from dash import Input, Output, State, ALL, no_update, ctx
from components.Tables.main_table import (
    pivot_by_network,
    render_kpi_table_multinet,
    strip_net,
)
from components.Tables.simple_tables import render_simple_table
from components.charts import line_by_time_multi

from src.data_access import fetch_kpis, fetch_kpis_paginated, COLMAP, fetch_kpis_paginated_global_sort, \
    fetch_kpis_paginated_alarm_sort
from src.config import REFRESH_INTERVAL_MS
from src.Utils.utils_tables import (
    cols_from_order,
    TABLE_VENDOR_SUMMARY_ORDER,
    HEADER_MAP,
    TABLE_TOP_ORDER,
)
from src.Utils.utils_charts import metrics_for_chart_cs, metrics_for_chart_ps
from src.Utils.utils_time import now_local


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def round_down_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def register_callbacks(app):

    # -------------------------------------------------
    # 0) Actualiza opciones de Network y Technology
    # -------------------------------------------------
    @app.callback(
        Output("f-network", "options"),
        Output("f-network", "value"),
        Output("f-technology", "options"),
        Output("f-technology", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
    )
    def update_network_tech(fecha, hora):
        # Para construir opciones, usa consulta no paginada (solo metadata)
        df = fetch_kpis(fecha=fecha, hora=hora, limit=None)

        networks = sorted(df["network"].dropna().unique().tolist()) if "network" in df.columns else []
        techs = sorted(df["technology"].dropna().unique().tolist()) if "technology" in df.columns else []

        net_opts = [{"label": n, "value": n} for n in networks]
        tech_opts = [{"label": t, "value": t} for t in techs]
        return net_opts, [], tech_opts, []

    # -------------------------------------------------
    # Botones de sort en headers
    # -------------------------------------------------
    @app.callback(
        Output("sort-state", "data"),
        Input({"type": "sort-btn", "col": ALL}, "n_clicks"),
        State("sort-state", "data"),
        prevent_initial_call=True,
    )
    def on_click_sort(n_clicks_list, sort_state):
        sort_state = sort_state or {"column": None, "ascending": True}
        trig = ctx.triggered_id
        if not trig or "col" not in trig:
            return sort_state

        clicked_col = trig["col"]
        if sort_state.get("column") in (clicked_col, strip_net(clicked_col)):
            sort_state["ascending"] = not sort_state.get("ascending", True)
        else:
            sort_state["column"] = clicked_col
            sort_state["ascending"] = True
        return sort_state

    # -------------------------------------------------
    # 1) Actualiza opciones de Vendor/Cluster
    # -------------------------------------------------
    @app.callback(
        Output("f-vendor", "options"),
        Output("f-vendor", "value"),
        Output("f-cluster", "options"),
        Output("f-cluster", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
    )
    def update_vendor_cluster(fecha, hora, networks, technologies):
        networks = _as_list(networks)
        technologies = _as_list(technologies)

        df = fetch_kpis(fecha=fecha, hora=hora, limit=None)
        if networks:
            df = df[df["network"].isin(networks)]
        if technologies:
            df = df[df["technology"].isin(technologies)]

        vendors = sorted(df["vendor"].dropna().unique().tolist()) if "vendor" in df.columns else []
        clusters = sorted(df["noc_cluster"].dropna().unique().tolist()) if "noc_cluster" in df.columns else []

        vendor_opts = [{"label": v, "value": v} for v in vendors]
        cluster_opts = [{"label": c, "value": c} for c in clusters]
        return vendor_opts, [], cluster_opts, []

    # -------------------------------------------------
    # 2) Paginación: reset page cuando cambian filtros/tamaño
    # -------------------------------------------------
    @app.callback(
        Output("page-state", "data"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("page-size", "value"),
        prevent_initial_call=True,
    )
    def reset_page_on_filters(_fecha, _hora, _net, _tech, _ven, _clu, page_size):
        ps = max(1, int(page_size or 50))
        return {"page": 1, "page_size": ps}

    # -------------------------------------------------
    # 3) Botones Anterior/Siguiente → actualizan page-state
    # -------------------------------------------------
    @app.callback(
        Output("page-state", "data", allow_duplicate=True),
        Input("page-prev", "n_clicks"),
        Input("page-next", "n_clicks"),
        State("page-state", "data"),
        prevent_initial_call=True,
    )
    def paginate(n_prev, n_next, state):
        state = state or {"page": 1, "page_size": 50}
        page = int(state.get("page", 1))
        ps = int(state.get("page_size", 50))

        trig = ctx.triggered_id
        if trig == "page-prev":
            page = max(1, page - 1)
        elif trig == "page-next":
            page = page + 1
        return {"page": page, "page_size": ps}

    # -------------------------------------------------
    # 4) Tabla + Charts + Indicadores de paginación (UN SOLO callback)
    #    ← Este callback es el ÚNICO que escribe en table-container y charts
    # -------------------------------------------------
    @app.callback(
        Output("table-container", "children"),
        Output("line-chart-a", "children"),
        Output("line-chart-b", "children"),
        Output("page-indicator", "children"),
        Output("total-rows-banner", "children"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("f-sort-mode", "value"),  # 👈 nuevo
        Input("refresh-timer", "n_intervals"),
        Input("sort-state", "data"),
        Input("page-state", "data"),
        prevent_initial_call=False,
    )
    def refresh_outputs(fecha, hora, networks, technologies, vendors, clusters, sort_mode, _n, sort_state, page_state):
        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        page = int((page_state or {}).get("page", 1))
        page_size = int((page_state or {}).get("page_size", 50))

        # ----------------------------
        # Detección de sort explícito
        # ----------------------------
        sort_by = None
        sort_net = None
        ascending = True
        if sort_state and sort_state.get("column"):
            col = sort_state["column"]
            ascending = bool(sort_state.get("ascending", True))
            if "__" in col:
                sort_net, base = col.split("__", 1)
                sort_by = base
            else:
                sort_by = col

        # ----------------------------
        # Fuente de datos según MODO
        # ----------------------------
        if sort_mode == "alarmado":
            safe_sort_state = None
            # Ignora sort explícito y usa orden por alarmas del JSON
            df, total = fetch_kpis_paginated_alarm_sort(
                fecha=fecha, hora=hora,
                vendors=vendors or None, clusters=clusters or None,
                networks=networks or None, technologies=technologies or None,
                page=page, page_size=page_size,
            )
        else:  # "global"
            safe_sort_state = sort_state
            if sort_by in COLMAP:
                df, total = fetch_kpis_paginated_global_sort(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=technologies or None,
                    page=page, page_size=page_size,
                    sort_by_friendly=sort_by,
                    sort_net=sort_net,
                    ascending=ascending,
                )
            else:
                df, total = fetch_kpis_paginated(
                    fecha=fecha, hora=hora,
                    vendors=vendors or None, clusters=clusters or None,
                    networks=networks or None, technologies=technologies or None,
                    page=page, page_size=page_size,
                )

        # ----------------------------
        # Resto igual que antes (nets, pivot, render, charts, banners)
        # ----------------------------
        if networks:
            nets = networks
        else:
            nets = sorted(df["network"].dropna().unique().tolist()) if "network" in df.columns else []

        if df is not None and not df.empty and nets:
            key_cols = ["fecha", "hora", "vendor", "noc_cluster", "technology"]
            tuples_in_order = list(dict.fromkeys(
                map(tuple, df[key_cols].itertuples(index=False, name=None))
            ))
            order_map = {t: i for i, t in enumerate(tuples_in_order)}
            wide = pivot_by_network(df, networks=nets)
            if wide is not None and not wide.empty:
                wide["_ord"] = wide[key_cols].apply(lambda r: order_map.get(tuple(r.values.tolist()), 10 ** 9), axis=1)
                wide = wide.sort_values("_ord").drop(columns=["_ord"])
        else:
            wide = df

        table = render_kpi_table_multinet(
            wide if (wide is not None and not wide.empty) else df,
            networks=nets,
            sort_state=safe_sort_state,  # 👈 aquí
        )

        chart_cs = line_by_time_multi(df, metrics_for_chart_cs)
        chart_ps = line_by_time_multi(df, metrics_for_chart_ps)

        import math
        total_pages = max(1, math.ceil(total / max(1, page_size)))
        page_corrected = min(max(1, page), total_pages)
        indicator = f"Página {page_corrected} de {total_pages}"
        banner = "Sin resultados." if total == 0 else f"Mostrando {(page_corrected - 1) * page_size + 1}–{min(page_corrected * page_size, total)} de {total} registros"

        return table, chart_cs, chart_ps, indicator, banner

    # -------------------------------------------------
    # 5) Intervalo global → sincroniza el del card (si aplica)
    # -------------------------------------------------
    @app.callback(
        Output("refresh-interval", "interval"),
        Input("refresh-interval-global", "n_intervals"),
        prevent_initial_call=False,
    )
    def sync_intervals(_n):
        return REFRESH_INTERVAL_MS

    # -------------------------------------------------
    # 6) Tablas simples inferiores (sin paginación)
    # -------------------------------------------------
    @app.callback(
        Output("table-bottom-a", "children"),
        Output("table-bottom-b", "children"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("refresh-timer", "n_intervals"),
    )
    def refresh_bottom_tables(fecha, hora, networks, technologies, vendors, clusters, _n):
        networks = _as_list(networks)
        technologies = _as_list(technologies)
        vendors = _as_list(vendors)
        clusters = _as_list(clusters)

        # A) Top clusters (ejemplo CS)
        df_top = fetch_kpis(fecha=fecha, hora=hora, vendors=vendors or None, clusters=clusters or None)
        if networks:
            df_top = df_top[df_top["network"].isin(networks)]
        if technologies:
            df_top = df_top[df_top["technology"].isin(technologies)]
        cols_top = cols_from_order(TABLE_TOP_ORDER, HEADER_MAP)
        table_a = render_simple_table(df_top, "Reporte por CS", cols_top)

        # B) Resumen por vendor (ejemplo PS)
        df_vs = fetch_kpis(fecha=fecha, hora=hora, vendors=vendors or None, clusters=clusters or None)
        if networks:
            df_vs = df_vs[df_vs["network"].isin(networks)]
        if technologies:
            df_vs = df_vs[df_vs["technology"].isin(technologies)]
        cols_vs = cols_from_order(TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP)
        table_b = render_simple_table(df_vs, "Reporte por PS", cols_vs)

        return table_a, table_b

    # -------------------------------------------------
    # 7) Tick: actualizar fecha/hora al inicio de cada hora
    # -------------------------------------------------
    @app.callback(
        Output("f-hora", "value"),
        Output("f-fecha", "date"),
        Input("refresh-timer", "n_intervals"),
        State("f-hora", "value"),
        State("f-fecha", "date"),
        State("f-hora", "options"),
        prevent_initial_call=False,
    )
    def tick(_, current_hour, current_date, hour_options):
        now = now_local()
        floored = round_down_to_hour(now)
        hh = floored.strftime("%H:00:00")
        today = floored.strftime("%Y-%m-%d")

        # No sobre-escribas si el usuario fijó manualmente
        if (current_date not in (None, today)) or (current_hour not in (None, hh)):
            return no_update, no_update

        opt_values = {(o["value"] if isinstance(o, dict) else o) for o in (hour_options or [])}
        if opt_values and hh not in opt_values:
            return no_update, no_update

        return hh, today

    @app.callback(
        Output("page-state", "data", allow_duplicate=True),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("page-size", "value"),
        Input("f-sort-mode", "value"),  # 👈 nuevo
        prevent_initial_call=True,
    )
    def reset_page_on_filters(_fecha, _hora, _net, _tech, _ven, _clu, page_size, _mode):
        ps = max(1, int(page_size or 50))
        return {"page": 1, "page_size": ps}

    @app.callback(
        Output("filters-collapse", "is_open"),
        Input("filters-toggle", "n_clicks"),
        State("filters-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_filters(n, is_open):
        return not is_open