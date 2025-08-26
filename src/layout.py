from dash import html, dcc
import dash_bootstrap_components as dbc

from components.filters import build_filters
from src.config import REFRESH_INTERVAL_MS
from src.utils import default_date_str, default_hour_str

def serve_layout():
    return dbc.Container([
        html.H2("Dashboard Master", className="my-3"),
        build_filters(),  # se llenan opciones en callbacks
        html.Div(id="cards-row", children=[
            dbc.Row([
                dbc.Col(html.Div(id="table-container"), md=12, className="my-3")
            ]),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4("Gráfica de línea", className="mb-3"),
                    dcc.Dropdown(
                        id="chart-metric",
                        options=[
                            {"label": "total_mbytes_nocperf", "value": "total_mbytes_nocperf"},
                            {"label": "total_erlangs_nocperf", "value": "total_erlangs_nocperf"},
                            {"label": "traffic_gb_att", "value": "traffic_gb_att"},
                        ],
                        value="total_mbytes_nocperf",
                        clearable=False
                    ),
                    html.Div(id="line-chart")
                ]), className="shadow-sm"), md=12, className="my-3")
            ]),
            # src/layout.py (dentro de serve_layout, debajo de la fila de la gráfica)
            dbc.Row([
                dbc.Col(html.Div(id="table-bottom-a"), md=6, className="my-3"),
                dbc.Col(html.Div(id="table-bottom-b"), md=6, className="my-3"),
            ])

        ]),
        dcc.Store(id="defaults-store", data={"fecha": default_date_str(), "hora": default_hour_str()}),
        dcc.Interval(id="refresh-interval-global", interval=REFRESH_INTERVAL_MS, n_intervals=0),
    ], fluid=True)
