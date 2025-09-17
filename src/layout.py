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
            ],
            className="position-relative"
        ),

        # Contenido principal
        html.Div(id="cards-row", children=[

            # Tabla principal + paginación
            dbc.Row([
                dbc.Col([
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
                            dbc.Col(
                                dbc.Button("Exportar Excel", id="export-excel", color="primary", size="sm"),
                                width="auto"
                            ),
                        ], className="g-2 align-items-center"),
                    ]), className="shadow-sm mb-2"),

                    html.Div(
                        id="table-container",
                        className="kpi-table-wrap kpi-table-container"
                    ),
                ], md=12, className="my-3"),
            ]),

            # === Heatmaps agrupados en un solo Card ===
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Row([
                                dbc.Col(html.H4("Degradados", className="m-0"), width="auto"),
                            ], className="g-2 align-items-center justify-content-center"),
                            className="bg-transparent border-0"
                        ),
                        dbc.CardBody([
                            # Controles compactos centrados
                            dbc.Row([
                                dbc.Col(
                                    dbc.ButtonGroup([
                                        dbc.Button("«", id="hm-page-prev", n_clicks=0, size="sm", color="secondary"),
                                        dbc.Button(id="hm-page-indicator", size="sm", disabled=True,
                                                   color="secondary", className="px-2"),
                                        dbc.Button("»", id="hm-page-next", n_clicks=0, size="sm", color="secondary"),
                                    ], size="sm"),
                                    width="auto", className="d-flex justify-content-center"
                                ),
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Tamaño", className="py-0"),
                                        dbc.Input(
                                            id="hm-page-size",
                                            type="number", min=5, step=5, value=5, size="sm",
                                            style={"width": "80px"}
                                        ),
                                    ], size="sm"),
                                    width="auto", className="d-flex justify-content-center"
                                ),
                                dbc.Col(
                                    html.Small(id="hm-total-rows-banner", className="text-muted"),
                                    width="auto", className="d-flex align-items-center"
                                ),
                            ], className="g-3 justify-content-center text-center mb-2"),

                            # Heatmaps lado a lado
                            dbc.Row([
                                dbc.Col(
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="hm-pct",
                                            config={"displayModeBar": False},
                                            style={"height": "760px", "width": "100%"}
                                        ),
                                        type="default"
                                    ),
                                    md=6, sm=12, className="my-2"
                                ),
                                dbc.Col(
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="hm-unit",
                                            config={"displayModeBar": False},
                                            style={"height": "760px", "width": "100%"}
                                        ),
                                        type="default"
                                    ),
                                    md=6, sm=12, className="my-2"
                                ),
                            ]),
                        ]),
                    ], className="bg-dark text-white border-0 shadow-sm"),
                    md=12, className="my-3"
                ),
            ]),
        ]),

        # Stores
        dcc.Store(id="defaults-store", data={"fecha": default_date_str(), "hora": default_hour_str()}),
        dcc.Store(id="page-state", data={"page": 1, "page_size": 5}),
        dcc.Store(id="sort-state", data={"column": None, "ascending": True}),
        dcc.Store(id="table-page-data"),

        # Señales/estado Heatmap
        dcc.Store(id="heatmap-trigger", data=None),
        dcc.Store(id="heatmap-page-state", data={"page": 1, "page_size": 5}),
        dcc.Store(id="heatmap-page-info"),
        dcc.Store(id="heatmap-params"),

        dcc.Interval(id="refresh-timer", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dcc.Download(id="download-excel"),
    ],
        fluid=True,
        style={"backgroundColor": "#121212", "color": "white"}
    )
