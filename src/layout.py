from dash import html, dcc
import dash_bootstrap_components as dbc

from components.filters import build_filters
from src.config import REFRESH_INTERVAL_MS
from src.Utils.utils_time import default_date_str, default_hour_str
from components.umbral_config_modal import create_umbral_config_modal


def serve_layout():
    return dbc.Container([
        # Header principal
        dbc.Row([
            dbc.Col([
                html.H2("Dashboard Master", className="my-3"),
            ], width=8),

        ], className="align-items-center"),

        # Filtros
        html.Div(
            children=[
                build_filters(),  # tu card/contenedor de filtros
                html.Div(
                    create_umbral_config_modal(),  # botón + modal
                    className="position-absolute top-0 end-0 mt-3 me-3 umbral-fab"
                ),
            ],
            className="position-relative"  # ancla para el posicionamiento absoluto
        ),
        # Contenido principal
        html.Div(id="cards-row", children=[
            dbc.Row([
                dbc.Col(html.Div(id="table-container"), md=12, className="my-3")
            ]),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4("Gráfica A (CS)", className="mb-3"),
                    html.Div(id="line-chart-a")
                ]), className="shadow-sm"), md=6, sm=12, className="my-3"),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H4("Gráfica B (PS)", className="mb-3"),
                    html.Div(id="line-chart-b")
                ]), className="shadow-sm"), md=6, sm=12, className="my-3"),
            ]),

            # Tablas inferiores
            dbc.Row([
                dbc.Col(html.Div(id="table-bottom-a"), md=6, className="my-3"),
                dbc.Col(html.Div(id="table-bottom-b"), md=6, className="my-3"),
            ])
        ]),

        # Stores e Intervalos
        dcc.Store(id="defaults-store", data={"fecha": default_date_str(), "hora": default_hour_str()}),
        dcc.Interval(id="refresh-timer", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dcc.Store(id="sort-state", data={"column": None, "ascending": True}),

    ], fluid=True)