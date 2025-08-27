from dash import Input, Output
from components.Tables.main_table import render_kpi_table
from components.Tables.simple_tables import render_simple_table
from components.charts import line_by_time_multi
from .data_access import fetch_kpis, distinct_vendors, distinct_clusters
from .config import REFRESH_INTERVAL_MS
from src.Utils.utils_tables import cols_from_order, TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP, TABLE_TOP_ORDER
from src.Utils.utils_charts import metrics_for_chart_cs, metrics_for_chart_ps
def register_callbacks(app):

    # 1) Actualiza opciones de vendor/cluster cuando cambian fecha/hora
    @app.callback(
        Output("f-vendor", "options"),
        Output("f-vendor", "value"),
        Output("f-cluster", "options"),
        Output("f-cluster", "value"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
    )
    def update_filter_options(fecha, hora):
        vendors = distinct_vendors(fecha=fecha, hora=hora)
        vendor_opts = [{"label": v, "value": v} for v in vendors]
        vendor_val = vendors[:1]

        clusters = distinct_clusters(fecha=fecha, hora=hora, vendors=vendor_val)
        cluster_opts = [{"label": c, "value": c} for c in clusters]
        cluster_val = clusters[:1]

        return vendor_opts, vendor_val, cluster_opts, cluster_val


    # 2) Refresca tabla y dos gráficas ante: filtros o intervalos
    @app.callback(
        Output("table-container", "children"),
        Output("line-chart-a", "children"),
        Output("line-chart-b", "children"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("refresh-interval-global", "n_intervals"),
    )
    def refresh_outputs(fecha, hora, vendors, clusters, _n):
        if isinstance(vendors, str): vendors = [vendors]
        if isinstance(clusters, str): clusters = [clusters]

        df = fetch_kpis(
            fecha=fecha,
            hora=hora,
            vendors=vendors or None,
            clusters=clusters or None,
            limit=None
        )

        table = render_kpi_table(df)
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

# 4) Tablas simples abajo (Top clusters + Resumen vendor)
    @app.callback(
        Output("table-bottom-a", "children"),
        Output("table-bottom-b", "children"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("refresh-interval-global", "n_intervals"),
)
    def refresh_bottom_tables(fecha, hora, vendors, clusters, _n):
        if isinstance(vendors, str): vendors = [vendors]
        if isinstance(clusters, str): clusters = [clusters]

        # A) Top clusters por PS RRC Fail
        df_top = fetch_kpis(
            fecha=fecha, hora=hora, vendors=vendors or None, clusters=clusters or None
        )
        cols_top = cols_from_order(TABLE_TOP_ORDER, HEADER_MAP)
        table_a = render_simple_table(df_top, "Reporte por CS", cols_top)

        # B) Resumen por vendor
        df_vs = fetch_kpis(
            fecha=fecha, hora=hora, vendors=vendors or None, clusters=clusters or None
        )
        cols_vs = cols_from_order(TABLE_VENDOR_SUMMARY_ORDER, HEADER_MAP)
        table_b = render_simple_table(df_vs, "Reporte por PS", cols_vs)

        return table_a, table_b
