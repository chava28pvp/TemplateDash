from dash import Input, Output, no_update
from components.Tables.main_table import render_kpi_table_multinet
from components.Tables.simple_tables import render_simple_table
from components.charts import line_by_time_multi
from .data_access import fetch_kpis
from .config import REFRESH_INTERVAL_MS
from src.Utils.utils_tables import cols_from_order, TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP, TABLE_TOP_ORDER
from src.Utils.utils_charts import metrics_for_chart_cs, metrics_for_chart_ps

def _ensure_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def register_callbacks(app):

    # 0) Actualiza opciones de Network y Technology cuando cambian fecha/hora
    @app.callback(
        Output("f-network", "options"),
        Output("f-network", "value"),
        Output("f-technology", "options"),
        Output("f-technology", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
    )
    def update_network_tech(fecha, hora):
        # Trae mínimo de datos y obtiene únicos
        df = fetch_kpis(fecha=fecha, hora=hora, limit=None)
        networks = sorted([x for x in df["network"].dropna().unique().tolist()])
        techs    = sorted([x for x in df["technology"].dropna().unique().tolist()])

        net_opts = [{"label": n, "value": n} for n in networks]
        tech_opts = [{"label": t, "value": t} for t in techs]

        # Por defecto, primera opción si existe (multi=True)
        net_val = networks[:1]
        tech_val = techs[:1]

        return net_opts, net_val, tech_opts, tech_val

    # 1) Actualiza opciones de Vendor/Cluster cuando cambian fecha/hora/network/technology
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
        # Normaliza multi
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)

        # Filtra por fecha/hora/network/technology para derivar vendors/clusters posibles
        df = fetch_kpis(
            fecha=fecha, hora=hora,
            # El data_access que te di acepta solo vendors/clusters como parámetros explícitos;
            # para filtrar por network/technology los derivamos con pandas aquí:
            limit=None
        )
        if networks:
            df = df[df["network"].isin(networks)]
        if technologies:
            df = df[df["technology"].isin(technologies)]

        vendors = sorted([x for x in df["vendor"].dropna().unique().tolist()])
        clusters = sorted([x for x in df["noc_cluster"].dropna().unique().tolist()])

        vendor_opts = [{"label": v, "value": v} for v in vendors]
        cluster_opts = [{"label": c, "value": c} for c in clusters]

        vendor_val = vendors[:1]
        cluster_val = clusters[:1]

        return vendor_opts, vendor_val, cluster_opts, cluster_val

    # 2) Refresca tabla y dos gráficas ante: filtros o intervalos
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
        Input("refresh-interval-global", "n_intervals"),
    )
    def refresh_outputs(fecha, hora, networks, technologies, vendors, clusters, _n):
        # Normaliza selección múltiple
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)
        vendors = _ensure_list(vendors)
        clusters = _ensure_list(clusters)

        # Trae datos base por fecha/hora y luego limita por network/tech si corresponde
        df = fetch_kpis(
            fecha=fecha,
            hora=hora,
            vendors=vendors or None,
            clusters=clusters or None,
            limit=None
        )
        if networks:
            df = df[df["network"].isin(networks)]
        if technologies:
            df = df[df["technology"].isin(technologies)]

        table = render_kpi_table_multinet(df)
        chart_cs = line_by_time_multi(df, metrics_for_chart_cs)
        chart_ps = line_by_time_multi(df, metrics_for_chart_ps)
        return table, chart_cs, chart_ps

    # 3) Configura el intervalo visual del card de filtros (sincronizado con global)
    @app.callback(
        Output("refresh-interval", "interval"),
        Input("refresh-interval-global", "n_intervals"),
        prevent_initial_call=False
    )
    def sync_intervals(_n):
        return REFRESH_INTERVAL_MS

    # 4) Tablas simples abajo (Top clusters + Resumen vendor) con nuevos filtros
    @app.callback(
        Output("table-bottom-a", "children"),
        Output("table-bottom-b", "children"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-network", "value"),
        Input("f-technology", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("refresh-interval-global", "n_intervals"),
    )
    def refresh_bottom_tables(fecha, hora, networks, technologies, vendors, clusters, _n):
        networks = _ensure_list(networks)
        technologies = _ensure_list(technologies)
        vendors = _ensure_list(vendors)
        clusters = _ensure_list(clusters)

        # A) Top clusters (ejemplo: reporte por CS)
        df_top = fetch_kpis(
            fecha=fecha, hora=hora, vendors=vendors or None, clusters=clusters or None
        )
        if networks:
            df_top = df_top[df_top["network"].isin(networks)]
        if technologies:
            df_top = df_top[df_top["technology"].isin(technologies)]

        cols_top = cols_from_order(TABLE_TOP_ORDER, HEADER_MAP)
        table_a = render_simple_table(df_top, "Reporte por CS", cols_top)

        # B) Resumen por vendor (ejemplo: reporte por PS)
        df_vs = fetch_kpis(
            fecha=fecha, hora=hora, vendors=vendors or None, clusters=clusters or None
        )
        if networks:
            df_vs = df_vs[df_vs["network"].isin(networks)]
        if technologies:
            df_vs = df_vs[df_vs["technology"].isin(technologies)]

        cols_vs = cols_from_order(TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP)
        table_b = render_simple_table(df_vs, "Reporte por PS", cols_vs)

        return table_a, table_b
