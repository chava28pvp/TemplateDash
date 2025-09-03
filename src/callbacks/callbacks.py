from dash import Input, Output, State, no_update
from components.Tables.main_table import (
    render_kpi_table_multinet, strip_net, expand_groups_for_networks,
    pivot_by_network, _resolve_sort_col
)
from components.Tables.simple_tables import render_simple_table
from components.charts import line_by_time_multi
from src.data_access import fetch_kpis
from src.config import REFRESH_INTERVAL_MS
from src.Utils.utils_tables import (
    cols_from_order, TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP, TABLE_TOP_ORDER,
)
from src.Utils.utils_charts import metrics_for_chart_cs, metrics_for_chart_ps
from src.Utils.utils_time import now_local

def _apply_multi(df, col, selected):
    if not selected:
        return df
    return df[df[col].isin(selected)]

def round_down_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

def _ensure_list(x):
    if x is None: return None
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def register_callbacks(app, cache):   # <-- recibe cache

    # importa ctx/ALL aquí para usarlo en callbacks internos
    from dash import ALL, ctx


    @cache.memoize(timeout=600)  # 10 min o lo que definas
    def _read_hour_df(fecha, hora):
        # Trae TODO lo de esa fecha/hora una vez
        return fetch_kpis(
            fecha=fecha,
            hora=hora,
            vendors=None,
            clusters=None,
            limit=None,
        )
    @cache.memoize(timeout=600)
    def _pivot_cached(fecha, hora, nets_tuple):
        df = _read_hour_df(fecha, hora)
        return pivot_by_network(df, networks=list(nets_tuple))


    # 0) Network/Technology
    @app.callback(
        Output("f-network", "options"),
        Output("f-network", "value"),
        Output("f-technology", "options"),
        Output("f-technology", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        prevent_initial_call=False,
    )
    def update_network_tech(fecha, hora):
        df = _read_hour_df(fecha, hora)  # RAM
        nets = sorted(df["network"].dropna().unique().tolist()) if "network" in df.columns else []
        techs = sorted(df["technology"].dropna().unique().tolist()) if "technology" in df.columns else []
        return (
            [{"label": n, "value": n} for n in nets], [],
            [{"label": t, "value": t} for t in techs], []
        )

    # sort-state
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

    # 1) Vendor/Cluster
    @app.callback(
        Output("f-vendor", "options"),
        Output("f-vendor", "value"),
        Output("f-cluster", "options"),
        Output("f-cluster", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        prevent_initial_call=False,
    )
    def update_vendor_cluster(fecha, hora, networks, technologies):
        df = _read_hour_df(fecha, hora)  # RAM
        networks = _ensure_list(networks) or []
        technologies = _ensure_list(technologies) or []

        if networks:
            df = df[df["network"].isin(networks)]
        if technologies:
            df = df[df["technology"].isin(technologies)]

        vendors = sorted(df["vendor"].dropna().unique().tolist()) if "vendor" in df.columns else []
        clusters = sorted(df["noc_cluster"].dropna().unique().tolist()) if "noc_cluster" in df.columns else []

        return (
            [{"label": v, "value": v} for v in vendors], [],
            [{"label": c, "value": c} for c in clusters], []
        )

    # 2) Tabla + gráficas
    @app.callback(
        Output("table-container", "children"),
        Output("line-chart-a", "children"),
        Output("line-chart-b", "children"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("refresh-timer", "n_intervals"),
        Input("sort-state", "data"),
    )
    def refresh_outputs(fecha, hora, networks, technologies, vendors, clusters, _n, sort_state):
        trig = ctx.triggered_id

        # 1) Carga RAM por (fecha, hora)
        df = _read_hour_df(fecha, hora)

        # 2) Filtros en memoria (rápidos)
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)
        vendors = _ensure_list(vendors)
        clusters = _ensure_list(clusters)

        if networks:
            df = df[df["network"].isin(networks)]
        if technologies:
            df = df[df["technology"].isin(technologies)]
        if vendors:
            df = df[df["vendor"].isin(vendors)]
        if clusters:
            df = df[df["noc_cluster"].isin(clusters)]

        nets = networks if networks else sorted(df["network"].dropna().unique().tolist())

        # 3) Ordenamiento / Tabla / Gráficas
        if trig == "sort-state" and sort_state:
            # (Opcional) usa pivot cacheado
            wide = _pivot_cached(fecha, hora, tuple(nets))
            if wide is not None and not wide.empty:
                _, METRIC_ORDER, _ = expand_groups_for_networks(nets)
                sort_col_req = sort_state.get("column")
                resolved = _resolve_sort_col(wide, METRIC_ORDER, sort_col_req)
                if resolved and (resolved in wide.columns):
                    wide = wide.sort_values(
                        by=resolved,
                        ascending=bool(sort_state.get("ascending", True)),
                        na_position="last"
                    )
                table_children = render_kpi_table_multinet(wide, networks=nets)
            else:
                table_children = render_kpi_table_multinet(df, networks=nets)

            chart_cs = no_update
            chart_ps = no_update
        else:
            table_children = render_kpi_table_multinet(df, networks=nets)
            chart_cs = line_by_time_multi(df, metrics_for_chart_cs)
            chart_ps = line_by_time_multi(df, metrics_for_chart_ps)

        return table_children, chart_cs, chart_ps

    # 3) (OJO) refresh-interval ↔ refresh-interval-global
    # Asegúrate de que ESTOS IDs existan en tu layout; si no, borra este callback.
    @app.callback(
        Output("refresh-interval", "interval"),
        Input("refresh-interval-global", "n_intervals"),
        prevent_initial_call=False,
    )
    def sync_intervals(_n):
        return REFRESH_INTERVAL_MS

    # 4) Tablas inferiores
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
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)
        vendors_t = tuple(_ensure_list(vendors) or [])
        clusters_t = tuple(_ensure_list(clusters) or [])

        df = _read_hour_df(fecha, hora)
        if networks:
            df = df[df["network"].isin(networks)]
        if technologies:
            df = df[df["technology"].isin(technologies)]

        cols_top = cols_from_order(TABLE_TOP_ORDER, HEADER_MAP)
        table_a = render_simple_table(df, "Reporte por CS", cols_top)

        cols_vs = cols_from_order(TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP)
        table_b = render_simple_table(df, "Reporte por PS", cols_vs)

        return table_a, table_b

    # 5) Tick de fecha/hora
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
        if (current_date not in (None, today)) or (current_hour not in (None, hh)):
            return no_update, no_update
        opt_values = {(o["value"] if isinstance(o, dict) else o) for o in (hour_options or [])}
        if opt_values and hh not in opt_values:
            return no_update, no_update
        return hh, today
