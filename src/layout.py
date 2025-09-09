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
        html.Div(
            create_umbral_config_modal(),
            className="position-absolute top-0 end-0 mt-3 me-3 umbral-fab"
        ),
        # Filtros + botón de configuración de umbrales (colapsables)
        html.Div(
            children=[
                # Barra de acciones (toggler de filtros)

                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            "Mostrar / Ocultar filtros",
                            id="filters-toggle",
                            color="secondary",
                            size="sm",
                            outline=True,
                            className="mb-2"
                        ),
                        width="auto"
                    ),

                ], className="g-2 align-items-center"),

                # Card de filtros colapsable
                dbc.Collapse(
                    id="filters-collapse",
                    is_open=True,  # abiertos por defecto
                    children=build_filters()
                ),

                # Botón + modal de umbrales, flotando arriba a la derecha

            ],
            className="position-relative"  # ancla para el posicionamiento absoluto
        ),


        # Contenido principal
        html.Div(id="cards-row", children=[

            # Tabla principal + paginación
            dbc.Row([
                dbc.Col([
                    # Controles de paginación
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dbc.Button("« Anterior", id="page-prev", n_clicks=0), width="auto"),
                            dbc.Col(html.Div(id="page-indicator", className="mx-2 fw-semibold"), width="auto"),
                            dbc.Col(dbc.Button("Siguiente »", id="page-next", n_clicks=0), width="auto"),
                            dbc.Col(
                                dbc.Input(
                                    id="page-size",
                                    type="number",
                                    min=10,
                                    step=10,
                                    value=50,
                                    placeholder="Tamaño",
                                    style={"width": "110px"}
                                ),
                                width="auto",
                                className="ms-3"
                            ),
                            dbc.Col(html.Div(id="total-rows-banner", className="text-muted"), width=True),
                        ], className="g-2 align-items-center"),
                    ]), className="shadow-sm mb-2"),

                    # Contenedor con altura fija y scroll para la tabla
                    html.Div(
                        id="table-container",
                        className="kpi-table-wrap kpi-table-container"
                    ),
                ], md=12, className="my-3"),
            ]),

            # Gráficas
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.H4("Gráfica (CS)", className="mb-3"),
                        dcc.Loading(html.Div(id="line-chart-a"))
                    ]), className="shadow-sm"),
                    md=6, sm=12, className="my-3"
                ),
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.H4("Gráfica (PS)", className="mb-3"),
                        dcc.Loading(html.Div(id="line-chart-b"))
                    ]), className="shadow-sm"),
                    md=6, sm=12, className="my-3"
                ),
            ]),

            # Tablas inferiores

        ]),

        # Stores
        dcc.Store(id="defaults-store", data={"fecha": default_date_str(), "hora": default_hour_str()}),
        dcc.Store(id="page-state", data={"page": 1, "page_size": 50}),  # ← estado de paginación
        dcc.Store(id="sort-state", data={"column": None, "ascending": True}),
        dcc.Interval(id="refresh-timer", interval=REFRESH_INTERVAL_MS, n_intervals=0),
    ],
        fluid=True,
        style={"backgroundColor": "#121212", "color": "white"}  # 👈 fondo y texto
    )
