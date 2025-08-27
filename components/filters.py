from dash import html, dcc
import dash_bootstrap_components as dbc
from src.Utils.utils_umbrales import default_date_str, default_hour_str

def build_filters(vendor_options=None, cluster_options=None):
    vendor_options = vendor_options or []
    cluster_options = cluster_options or []

    hours = [f"{h:02d}:00:00" for h in range(24)]
    return dbc.Card(
        dbc.CardBody([
            html.H4("Filtros", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Fecha"),
                    dcc.DatePickerSingle(
                        id="f-fecha",
                        date=default_date_str(),
                        display_format="YYYY-MM-DD",
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Hora"),
                    dcc.Dropdown(
                        id="f-hora",
                        options=[{"label": "Todas", "value": "Todas"}] + [{"label": h[:5], "value": h} for h in hours],
                        value=default_hour_str(),
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Vendor"),
                    dcc.Dropdown(
                        id="f-vendor",
                        options=[{"label": v, "value": v} for v in vendor_options],
                        value=vendor_options[:1],
                        multi=True,
                        placeholder="Selecciona vendor"
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Cluster"),
                    dcc.Dropdown(
                        id="f-cluster",
                        options=[{"label": c, "value": c} for c in cluster_options],
                        value=cluster_options[:1],
                        multi=True,
                        placeholder="Selecciona cluster"
                    )
                ], md=3),
            ], className="g-3"),
            dcc.Interval(id="refresh-interval"),
        ]),
        className="shadow-sm"
    )
