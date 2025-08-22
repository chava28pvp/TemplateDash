from dash import Input, Output, State, callback_context, no_update
from components.kpi_table import render_kpi_table
from components.charts import line_by_time
from .data_access import fetch_kpis, distinct_vendors, distinct_clusters
from .config import REFRESH_INTERVAL_MS

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

    # 2) Refresca tabla y gráfica ante: filtros o intervalos
    @app.callback(
        Output("table-container", "children"),
        Output("line-chart", "children"),
        Input("f-fecha", "date"),
        Input("f-hora", "value"),
        Input("f-vendor", "value"),
        Input("f-cluster", "value"),
        Input("chart-metric", "value"),
        Input("refresh-interval-global", "n_intervals"),
    )
    def refresh_outputs(fecha, hora, vendors, clusters, metric, _n):
        # vendors/clusters pueden ser None o string (cuando una sola opción)
        if isinstance(vendors, str): vendors = [vendors]
        if isinstance(clusters, str): clusters = [clusters]

        df = fetch_kpis(
            fecha=fecha,
            hora=hora,
            vendors=vendors if vendors else None,
            clusters=clusters if clusters else None,
            limit=None
        )

        table = render_kpi_table(df)
        chart = line_by_time(df, y_col=metric, color_col="vendor")
        return table, chart

    # 3) Configura el intervalo visual del card de filtros (sincronizado con global)
    @app.callback(
        Output("refresh-interval", "interval"),
        Input("refresh-interval-global", "n_intervals"),
        prevent_initial_call=False
    )
    def sync_intervals(_n):
        return REFRESH_INTERVAL_MS
