from dash import Input, Output, State, no_update
from components.Tables.main_table import render_kpi_table_multinet
from components.Tables.simple_tables import render_simple_table
from components.charts import line_by_time_multi
from .data_access import fetch_kpis
from .config import REFRESH_INTERVAL_MS
from src.Utils.utils_tables import (
    cols_from_order,
    TABLE_VENDOR_SUMMARY_ORDER,
    HEADER_MAP,
    TABLE_TOP_ORDER,
)
from src.Utils.utils_charts import metrics_for_chart_cs, metrics_for_chart_ps
from src.Utils.utils_time import now_local


# =========================
# Helpers
# =========================

def _apply_multi(df, col, selected):
    if not selected:  # None, [], ()
        return df
    return df[df[col].isin(selected)]


def _ensure_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def round_down_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


# =========================
# Callbacks
# =========================

def register_callbacks(app):

    # -------------------------------------------------
    # 0) Actualiza opciones de Network y Technology
    #     cuando cambian fecha/hora
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
        df = fetch_kpis(fecha=fecha, hora=hora, limit=None)

        networks = sorted(df["network"].dropna().unique().tolist())
        techs = sorted(df["technology"].dropna().unique().tolist())

        net_opts = [{"label": n, "value": n} for n in networks]
        tech_opts = [{"label": t, "value": t} for t in techs]

        # <- SIN preselección
        return net_opts, [], tech_opts, []

    # -------------------------------------------------
    # 1) Actualiza opciones de Vendor/Cluster
    #     cuando cambian fecha/hora/network/technology
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
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)

        df = fetch_kpis(fecha=fecha, hora=hora, limit=None)
        if networks:
            df = df[df["network"].isin(networks)]
        if technologies:
            df = df[df["technology"].isin(technologies)]

        vendors = sorted(df["vendor"].dropna().unique().tolist())
        clusters = sorted(df["noc_cluster"].dropna().unique().tolist())

        vendor_opts = [{"label": v, "value": v} for v in vendors]
        cluster_opts = [{"label": c, "value": c} for c in clusters]

        # <- SIN preselección
        return vendor_opts, [], cluster_opts, []

    # -------------------------------------------------
    # 2) Refresca tabla y dos gráficas
    #     ante: filtros o intervalos
    # -------------------------------------------------
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
    )
    def refresh_outputs(fecha, hora, networks, technologies, vendors, clusters, _n):
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)
        vendors = _ensure_list(vendors)
        clusters = _ensure_list(clusters)

        df = fetch_kpis(
            fecha=fecha,
            hora=hora,
            vendors=vendors or None,
            clusters=clusters or None,
            limit=None,
        )

        # Aplica filtros solo si hay selección
        df = _apply_multi(df, "network", networks)
        df = _apply_multi(df, "technology", technologies)

        # Redes a mostrar en el header 3 niveles
        if networks and len(networks) > 0:
            nets = tuple(networks)
        else:
            nets = tuple(sorted(df["network"].dropna().unique().tolist()))

        table = render_kpi_table_multinet(df, networks=nets)
        chart_cs = line_by_time_multi(df, metrics_for_chart_cs)
        chart_ps = line_by_time_multi(df, metrics_for_chart_ps)

        return table, chart_cs, chart_ps

    # -------------------------------------------------
    # 3) Configura el intervalo visual del card de filtros
    #     (sincronizado con global)
    # -------------------------------------------------
    @app.callback(
        Output("refresh-interval", "interval"),
        Input("refresh-interval-global", "n_intervals"),
        prevent_initial_call=False,
    )
    def sync_intervals(_n):
        return REFRESH_INTERVAL_MS

    # -------------------------------------------------
    # 4) Tablas simples abajo (Top clusters + Resumen vendor)
    #     con nuevos filtros
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
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)
        vendors = _ensure_list(vendors)
        clusters = _ensure_list(clusters)

        # A) Top clusters (ejemplo: reporte por CS)
        df_top = fetch_kpis(
            fecha=fecha,
            hora=hora,
            vendors=vendors or None,
            clusters=clusters or None,
        )
        if networks:
            df_top = df_top[df_top["network"].isin(networks)]
        if technologies:
            df_top = df_top[df_top["technology"].isin(technologies)]

        cols_top = cols_from_order(TABLE_TOP_ORDER, HEADER_MAP)
        table_a = render_simple_table(df_top, "Reporte por CS", cols_top)

        # B) Resumen por vendor (ejemplo: reporte por PS)
        df_vs = fetch_kpis(
            fecha=fecha,
            hora=hora,
            vendors=vendors or None,
            clusters=clusters or None,
        )
        if networks:
            df_vs = df_vs[df_vs["network"].isin(networks)]
        if technologies:
            df_vs = df_vs[df_vs["technology"].isin(technologies)]

        cols_vs = cols_from_order(TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP)
        table_b = render_simple_table(df_vs, "Reporte por PS", cols_vs)

        return table_a, table_b

    # -------------------------------------------------
    # 5) Actualiza automáticamente fecha/hora
    #     al inicio de cada hora
    # -------------------------------------------------
    @app.callback(
        Output("f-hora", "value"),
        Output("f-fecha", "date"),
        Input("refresh-timer", "n_intervals"),
        State("f-hora", "value"),
        State("f-fecha", "date"),
        State("f-hora", "options"),  # útil para validar que la hora exista
        prevent_initial_call=False,
    )
    def tick(_, current_hour, current_date, hour_options):
        # Hora local “al inicio de la hora”
        now = now_local()
        floored = round_down_to_hour(now)
        hh = floored.strftime("%H:00:00")
        today = floored.strftime("%Y-%m-%d")

        # ⛔️ Si el usuario fijó manualmente fecha/hora distintas, NO sobre-escribas
        if (current_date not in (None, today)) or (current_hour not in (None, hh)):
            return no_update, no_update

        # ✅ Solo si el usuario está “siguiendo en vivo”
        opt_values = {
            (o["value"] if isinstance(o, dict) else o) for o in (hour_options or [])
        }
        if opt_values and hh not in opt_values:
            # Si la nueva hora no existe en el dropdown, no fuerces actualización
            return no_update, no_update

        return hh, today
