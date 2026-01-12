from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from components.filters import build_filters, build_topoff_filters
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
                    is_open=False,
                    children=build_filters()
                ),
            ],
            className="filters-sticky-wrapper"
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

            dbc.Row([
                dbc.Col(
                    dbc.Card([

                        # =========================
                        # HEADER: título + botón integridad
                        # =========================
                        dbc.CardHeader(
                            dbc.Row(
                                [
                                    # spacer izquierdo (equilibra el ancho del botón)
                                    dbc.Col(width="auto"),

                                    # título centrado
                                    dbc.Col(
                                        html.H4("Degradados", className="m-0 text-center"),
                                        className="d-flex justify-content-center",
                                    ),

                                    # botón a la derecha
                                    dbc.Col(
                                        dbc.Button(
                                            "Integridad",
                                            id="btn-toggle-hm-int",
                                            size="sm",
                                            color="secondary",
                                            outline=True,
                                            n_clicks=0,
                                        ),
                                        width="auto",
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="g-2 align-items-center",
                            ),
                            className="bg-transparent border-0",
                        ),

                        dbc.CardBody([

                            # (Opcional) trigger bajo demanda
                            dcc.Store(id="heatmap-integrity-trigger", data=None),

                            # =========================
                            # Controles de paginación (PRINCIPAL) - se comparten
                            # =========================
                            dbc.Row([
                                dbc.Col(
                                    dbc.ButtonGroup([
                                        dbc.Button("«", id="hm-page-prev", n_clicks=0, size="sm", color="secondary"),
                                        dbc.Button(
                                            id="hm-page-indicator",
                                            size="sm",
                                            disabled=True,
                                            color="secondary",
                                            className="px-2"
                                        ),
                                        dbc.Button("»", id="hm-page-next", n_clicks=0, size="sm", color="secondary"),
                                    ], size="sm"),
                                    width="auto",
                                    className="d-flex justify-content-center"
                                ),

                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Tamaño", className="py-0"),
                                        dbc.Input(
                                            id="hm-page-size",
                                            type="number",
                                            min=10,
                                            step=10,
                                            value=50,
                                            size="sm",
                                            style={"width": "80px"}
                                        ),
                                    ], size="sm"),
                                    width="auto",
                                    className="d-flex justify-content-center"
                                ),

                                dbc.Col(
                                    html.Small(id="hm-total-rows-banner", className="text-muted"),
                                    width="auto",
                                    className="d-flex align-items-center"
                                ),

                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Orden", className="py-0"),
                                        dbc.Select(
                                            id="hm-order-by",
                                            options=[
                                                {"label": "Alarm %", "value": "alarm_hours"},
                                                {"label": "Alarm UNIT", "value": "alarm_hours_unit"},
                                            ],
                                            value="alarm_hours",
                                            size="sm",
                                            className="bg-white text-dark",
                                            style={"width": "160px"}
                                        ),
                                    ], size="sm"),
                                    width="auto",
                                    className="d-flex justify-content-center"
                                ),
                            ], className="g-3 justify-content-center text-center mb-2"),

                            # =========================
                            # Header fijo (principal)
                            # =========================
                            dbc.Row([
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dbc.Table([
                                                html.Thead(html.Tr([
                                                    html.Th("Cluster", className="w-cluster"),
                                                    html.Th("Tech", className="w-tech"),
                                                    html.Th("Vendor", className="w-vendor"),
                                                    html.Th("Valor", className="w-valor"),
                                                    html.Th("Última", className="w-ultima"),
                                                    html.Th("% Fail.", className="w-num ta-right"),
                                                    html.Th("Fail Últ.", className="w-num ta-right"),
                                                ])),
                                            ],
                                                bordered=False,
                                                hover=False,
                                                size="sm",
                                                className="mb-0 table-dark kpi-table kpi-table-summary header-only"
                                            ),
                                        ], className="p-1"),
                                    ),
                                    md=4, sm=12
                                ),

                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(id="hm-pct-dates", className="hm-time-row hm-time-dates"),
                                            html.Div(id="hm-pct-hours", className="hm-time-row hm-time-hours"),
                                        ], className="p-2"),
                                        className="hm-time-card"
                                    ),
                                    md=4, sm=12
                                ),

                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(id="hm-unit-dates", className="hm-time-row hm-time-dates"),
                                            html.Div(id="hm-unit-hours", className="hm-time-row hm-time-hours"),
                                        ], className="p-2"),
                                        className="hm-time-card"
                                    ),
                                    md=4, sm=12
                                ),
                            ], className="g-0 align-items-stretch mb-1"),

                            # =========================
                            # ÚNICO SCROLL (principal)
                            # =========================
                            html.Div(
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                html.Div(id="hm-table-container"),
                                            ], className="p-0", style={"minHeight": 0}),
                                            className="bg-dark text-white border-0 h-100"
                                        ),
                                        md=4, sm=12, className="mb-0"
                                    ),

                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id="hm-pct",
                                                        config={"displayModeBar": False},
                                                        style={"width": "100%", "margin": 0}
                                                    ),
                                                    type="default"
                                                ),
                                            ], className="p-0 hm-nudge-left"),
                                            className="bg-dark text-white border-0 h-100"
                                        ),
                                        md=4, sm=12, className="mb-0"
                                    ),

                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id="hm-unit",
                                                        config={"displayModeBar": False},
                                                        style={"width": "100%", "margin": 0}
                                                    ),
                                                    type="default"
                                                ),
                                            ], className="p-0 hm-nudge-left"),
                                            className="bg-dark text-white border-0 h-100"
                                        ),
                                        md=4, sm=12, className="mb-0"
                                    ),
                                ], className="g-0 align-items-stretch"),
                                className="hm-board"
                            ),

                            # =========================
                            # INTEGRIDAD (bajo demanda) - con tabla + headers fijos + scroll interno
                            # =========================
                            dbc.Collapse(
                                id="collapse-hm-int",
                                is_open=False,
                                className="mt-3",
                                children=[
                                    dbc.CardHeader(
                                        dbc.Row(
                                            [
                                                dbc.Col(width="auto"),
                                                dbc.Col(
                                                    html.H5("Integridad", className="m-0 text-center"),
                                                    className="d-flex justify-content-center",
                                                ),
                                                dbc.Col(width="auto"),
                                            ],
                                            className="g-2 align-items-center",
                                        ),
                                        className="bg-transparent border-0 py-1",
                                    ),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.ButtonGroup([
                                                dbc.Button("«", id="hm-int-page-prev", n_clicks=0, size="sm",
                                                           color="secondary"),
                                                dbc.Button(
                                                    id="hm-int-page-indicator",
                                                    size="sm",
                                                    disabled=True,
                                                    color="secondary",
                                                    className="px-2"
                                                ),
                                                dbc.Button("»", id="hm-int-page-next", n_clicks=0, size="sm",
                                                           color="secondary"),
                                            ], size="sm"),
                                            width="auto",
                                            className="d-flex justify-content-center"
                                        ),

                                        dbc.Col(
                                            dbc.InputGroup([
                                                dbc.InputGroupText("Tamaño", className="py-0"),
                                                dbc.Input(
                                                    id="hm-int-page-size",
                                                    type="number",
                                                    min=10,
                                                    step=10,
                                                    value=50,
                                                    size="sm",
                                                    style={"width": "80px"}
                                                ),
                                            ], size="sm"),
                                            width="auto",
                                            className="d-flex justify-content-center"
                                        ),

                                        dbc.Col(
                                            html.Small(id="hm-int-total-rows-banner", className="text-muted"),
                                            width="auto",
                                            className="d-flex align-items-center"
                                        ),
                                    ], className="g-3 justify-content-center text-center mb-2"),
                                    # Header fijo de integridad (tabla + timelines)
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    dbc.Table([
                                                        html.Thead(html.Tr([
                                                            html.Th("Cluster", className="w-cluster"),
                                                            html.Th("Tech", className="w-tech"),
                                                            html.Th("Vendor", className="w-vendor"),
                                                            html.Th("Última", className="w-ultima"),
                                                            html.Th("%", className="w-num ta-right"),
                                                            html.Th("UNIT", className="w-num ta-right"),
                                                        ])),
                                                    ],
                                                        bordered=False,
                                                        hover=False,
                                                        size="sm",
                                                        className="mb-0 table-dark kpi-table kpi-table-summary header-only"
                                                    ),
                                                ], className="p-1"),
                                            ),
                                            md=4, sm=12
                                        ),

                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.Div(id="hm-int-pct-dates",
                                                             className="hm-time-row hm-time-dates"),
                                                    html.Div(id="hm-int-pct-hours",
                                                             className="hm-time-row hm-time-hours"),
                                                ], className="p-2"),
                                                className="hm-time-card"
                                            ),
                                            md=4, sm=12
                                        ),

                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.Div(id="hm-int-unit-dates",
                                                             className="hm-time-row hm-time-dates"),
                                                    html.Div(id="hm-int-unit-hours",
                                                             className="hm-time-row hm-time-hours"),
                                                ], className="p-2"),
                                                className="hm-time-card"
                                            ),
                                            md=4, sm=12
                                        ),
                                    ], className="g-0 align-items-stretch mb-1"),

                                    # Board con scroll interno (tabla + hm% + hm unit)
                                    html.Div(
                                        dbc.Row([
                                            # Tabla integridad
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        html.Div(id="hm-int-table-container"),
                                                    ], className="p-0", style={"minHeight": 0}),
                                                    className="bg-dark text-white border-0 h-100"
                                                ),
                                                md=4, sm=12, className="mb-0 hm-int-table-pane"
                                            ),

                                            # Heatmap % integridad
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        dcc.Loading(
                                                            dcc.Graph(
                                                                id="hm-int-pct",
                                                                figure=go.Figure(),
                                                                config={"displayModeBar": False},
                                                                style={"width": "100%", "margin": 0}
                                                            ),
                                                            type="default"
                                                        ),
                                                    ], className="p-0 hm-nudge-left"),
                                                    className="bg-dark text-white border-0 h-100"
                                                ),
                                                md=4, sm=12, className="mb-0"
                                            ),

                                            # Heatmap UNIT integridad
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        dcc.Loading(
                                                            dcc.Graph(
                                                                id="hm-int-unit",
                                                                figure=go.Figure(),
                                                                config={"displayModeBar": False},
                                                                style={"width": "100%", "margin": 0}
                                                            ),
                                                            type="default"
                                                        ),
                                                    ], className="p-0 hm-nudge-left"),
                                                    className="bg-dark text-white border-0 h-100"
                                                ),
                                                md=4, sm=12, className="mb-0"
                                            ),
                                        ], className="g-0 align-items-stretch"),
                                        className="hm-board",
                                        # si tu .hm-board ya controla scroll, borra esto:
                                        style={"maxHeight": "560px", "overflowY": "auto"},
                                    ),
                                ],
                            ),

                        ], className="p-2"),

                    ], className="bg-dark text-white border-0 shadow-sm mb-2"),
                    md=12
                ),
            ]),
            # === Histogramas en su propio Card (lado a lado, con títulos y scroll) ===
            html.Div(id="histo-anchor"),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.H4("Histogramas", className="m-0 text-center"),
                                        width=12
                                    ),
                                ],
                                className="g-0"
                            ),
                            className="bg-transparent border-0 py-2"
                        ),

                        dbc.CardBody([
                            # ---------- Bloque PS ----------
                            html.H5("PS", className="mb-2 text-center text-info"),

                            dbc.Row([
                                # PS %
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H6("%", className="mb-2 text-center"),
                                            html.Div(
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id="hi-pct-ps",
                                                        config={"displayModeBar": False},
                                                        className="histo-wide",
                                                        style={"height": "420px", "width": "1400px", "margin": "0"}
                                                    ),
                                                    type="default"
                                                ),
                                                className="histo-scroll"
                                            ),
                                        ],
                                        className="hm-wrap",
                                        style={"overflow": "hidden", "marginBottom": "6px"}
                                    ),
                                    md=6, sm=12, className="my-0"
                                ),

                                # PS UNIT
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H6("UNIT", className="mb-2 text-center"),
                                            html.Div(
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id="hi-unit-ps",
                                                        config={"displayModeBar": False},
                                                        className="histo-wide",
                                                        style={"height": "420px", "width": "1400px", "margin": "0"}
                                                    ),
                                                    type="default"
                                                ),
                                                className="histo-scroll"
                                            ),
                                        ],
                                        className="hm-wrap",
                                        style={"overflow": "hidden"}
                                    ),
                                    md=6, sm=12, className="my-0"
                                ),
                            ], className="g-3"),

                            html.Hr(className="my-3"),

                            # ---------- Bloque CS ----------
                            html.H5("CS", className="mb-2 text-center text-warning"),

                            dbc.Row([
                                # CS %
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H6("%", className="mb-2 text-center"),
                                            html.Div(
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id="hi-pct-cs",
                                                        config={"displayModeBar": False},
                                                        className="histo-wide",
                                                        style={"height": "420px", "width": "1400px", "margin": "0"}
                                                    ),
                                                    type="default"
                                                ),
                                                className="histo-scroll"
                                            ),
                                        ],
                                        className="hm-wrap",
                                        style={"overflow": "hidden", "marginBottom": "6px"}
                                    ),
                                    md=6, sm=12, className="my-0"
                                ),

                                # CS UNIT
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H6("UNIT", className="mb-2 text-center"),
                                            html.Div(
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id="hi-unit-cs",
                                                        config={"displayModeBar": False},
                                                        className="histo-wide",
                                                        style={"height": "420px", "width": "1400px", "margin": "0"}
                                                    ),
                                                    type="default"
                                                ),
                                                className="histo-scroll"
                                            ),
                                        ],
                                        className="hm-wrap",
                                        style={"overflow": "hidden"}
                                    ),
                                    md=6, sm=12, className="my-0"
                                ),
                            ], className="g-3"),
                        ], className="p-2"),
                    ], className="bg-dark text-white border-0 shadow-sm mb-2"),
                    md=12
                ),
            ]),

            dbc.Row([
                dbc.Col([

                    # Título + toggler
                    dbc.Row([
                        dbc.Col(html.H4("TopOff - Tabla", className="mb-2"), width=True),
                        dbc.Col(
                            dbc.Button(
                                "Filtros TopOff",
                                id="topoff-filters-toggle",
                                size="sm",
                                outline=True,
                                color="secondary",
                                className="mb-2",
                            ),
                            width="auto"
                        )
                    ], className="align-items-center g-2"),

                    # Collapse con filtros mini
                    dbc.Collapse(
                        id="topoff-filters-collapse",
                        is_open=False,
                        children=build_topoff_filters()
                    ),

                    # Card de paginado + export
                    dbc.Card(
                        dbc.CardBody([

                            dbc.Row([
                                dbc.Col(dbc.Button("« Anterior", id="topoff-page-prev", n_clicks=0), width="auto"),
                                dbc.Col(html.Div(id="topoff-page-indicator", className="mx-2 fw-semibold"),
                                        width="auto"),
                                dbc.Col(dbc.Button("Siguiente »", id="topoff-page-next", n_clicks=0), width="auto"),

                                dbc.Col(
                                    dbc.Input(
                                        id="topoff-page-size",
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

                                dbc.Col(html.Div(id="topoff-total-rows-banner", className="text-muted"), width=True),

                                # 👇 NUEVO: botón export a Excel para TopOff
                                dbc.Col(
                                    dbc.Button(
                                        "Exportar Excel",
                                        id="topoff-export-excel",
                                        color="primary",
                                        size="sm"
                                    ),
                                    width="auto"
                                ),
                            ], className="g-2 align-items-center"),

                            # 👇 NUEVO: target de descarga
                            dcc.Download(id="topoff-download-excel"),

                        ]),
                        className="shadow-sm mb-2"
                    ),

                    html.Div(
                        id="topoff-table-container",
                        className="kpi-table-wrap kpi-table-container",
                        style={"overflowX": "auto"},
                    ),
                ], md=12, className="my-3"),
            ]),

            # ==========================
            #  TopOff Heatmaps
            # ==========================
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Row([
                                dbc.Col(html.H4("TopOff - Degradados", className="m-0"), width="auto"),
                            ], className="g-2 align-items-center justify-content-center"),
                            className="bg-transparent border-0"
                        ),

                        dbc.CardBody([

                            # -- Controles de paginación del heatmap TopOff --
                            dbc.Row([
                                dbc.Col(
                                    dbc.ButtonGroup([
                                        dbc.Button("«", id="topoff-hm-page-prev", n_clicks=0, size="sm",
                                                   color="secondary"),
                                        dbc.Button(id="topoff-hm-page-indicator", size="sm", disabled=True,
                                                   color="secondary", className="px-2"),
                                        dbc.Button("»", id="topoff-hm-page-next", n_clicks=0, size="sm",
                                                   color="secondary"),
                                    ], size="sm"),
                                    width="auto", className="d-flex justify-content-center"
                                ),
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Tamaño", className="py-0"),
                                        dbc.Input(
                                            id="topoff-hm-page-size", type="number", min=10, step=10, value=50, size="sm",
                                            style={"width": "80px"}
                                        ),
                                    ], size="sm"),
                                    width="auto", className="d-flex justify-content-center"
                                ),
                                dbc.Col(
                                    html.Small(id="topoff-hm-total-rows-banner", className="text-muted"),
                                    width="auto", className="d-flex align-items-center"
                                ),
                                dbc.Col(
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Orden", className="py-0"),
                                        dcc.Dropdown(
                                            id="topoff-hm-order-by",
                                            clearable=False,
                                            searchable=False,
                                            value="alarm_bins_pct",
                                            options=[
                                                {"label": "Alarmas %", "value": "alarm_bins_pct"},
                                                {"label": "Alarmas UNIT", "value": "alarm_bins_unit"},
                                            ],
                                            className="bg-white text-dark",
                                            style={"minWidth": "160px"},
                                        ),
                                    ], size="sm"),
                                    width="auto", className="d-flex justify-content-center"
                                ),
                            ], className="g-3 justify-content-center text-center mb-2"),

                            # =========================================================
                            # WRAPPER TOP (namespace TopOff SIN scroll)
                            # =========================================================
                            html.Div(

                                [

                                    # -- HEADER FIJO (no scrollea): Tabla + Timeline % + Timeline UNIT --
                                    dbc.Row([
                                        # Encabezado de la tabla resumen (md=4)
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    dbc.Table([
                                                        html.Thead(html.Tr([
                                                            html.Th("Cluster", className="w-cluster"),
                                                            html.Th("Sitio", className="w-sitio"),
                                                            html.Th("Tech", className="w-tech"),
                                                            html.Th("Vendor", className="w-vendor"),
                                                            html.Th("Valor", className="w-valor"),
                                                            html.Th("Última", className="w-ultima"),
                                                            html.Th("% Últ.", className="w-num ta-right"),
                                                            html.Th("UNIT Últ.", className="w-num ta-right"),
                                                        ])),
                                                    ], bordered=False, hover=False, size="sm",
                                                        className="mb-0 table-dark kpi-table topoff-table header-only"),
                                                ], className="p-1"),
                                                className="bg-dark text-white border-0 h-100"
                                            ),
                                            md=4, sm=12
                                        ),

                                        # Timeline header Heatmap %
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.Div(id="topoff-hm-pct-dates",
                                                             className="hm-time-row hm-time-dates"),
                                                    html.Div(id="topoff-hm-pct-hours",
                                                             className="hm-time-row hm-time-hours"),
                                                ], className="p-2"),
                                                className="hm-time-card bg-dark text-white border-0 h-100"
                                            ),
                                            md=4, sm=12
                                        ),

                                        # Timeline header Heatmap UNIT
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.Div(id="topoff-hm-unit-dates",
                                                             className="hm-time-row hm-time-dates"),
                                                    html.Div(id="topoff-hm-unit-hours",
                                                             className="hm-time-row hm-time-hours"),
                                                ], className="p-2"),
                                                className="hm-time-card bg-dark text-white border-0 h-100"
                                            ),
                                            md=4, sm=12
                                        ),
                                    ], className="g-0 align-items-stretch mb-1"),

                                    # =========================================================
                                    # ÚNICO SCROLL: Tabla body + Heatmap % + Heatmap UNIT
                                    # =========================================================
                                    html.Div(
                                        dbc.Row([

                                            # === Tabla resumen body (sin headers) ===
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        html.Div(id="topoff-hm-table-container"),
                                                    ], className="p-0", style={"minHeight": 0}),
                                                    className="bg-dark text-white border-0 h-100"
                                                ),
                                                md=4, sm=12, className="mb-0"
                                            ),

                                            # === Heatmap % ===
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        dcc.Loading(
                                                            dcc.Graph(
                                                                id="topoff-hm-pct",
                                                                config={"displayModeBar": False},
                                                                style={"width": "100%", "margin": 0}
                                                            ),
                                                            type="default"
                                                        ),
                                                    ], className="p-0 hm-nudge-left"),
                                                    className="bg-dark text-white border-0 h-100"
                                                ),
                                                md=4, sm=12, className="mb-0"
                                            ),

                                            # === Heatmap UNIT ===
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        dcc.Loading(
                                                            dcc.Graph(
                                                                id="topoff-hm-unit",
                                                                config={"displayModeBar": False},
                                                                style={"width": "100%", "margin": 0}
                                                            ),
                                                            type="default"
                                                        ),
                                                    ], className="p-0 hm-nudge-left"),
                                                    className="bg-dark text-white border-0 h-100"
                                                ),
                                                md=4, sm=12, className="mb-0"
                                            ),

                                        ], className="g-0 align-items-stretch"),

                                        className="topoff-pane"  # 👈 solo aquí scrollea el body
                                    ),

                                ],

                                # 👇 Namespace TopOff pero SIN scroll en este wrapper
                                className="topoff-board",
                                style={"height": "auto", "overflow": "visible"}
                            ),



                        ], className="p-2"),
                    ], className="bg-dark text-white border-0 shadow-sm mb-2"),
                    md=12
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Row([
                                dbc.Col(html.H4("Histogramas TopOff", className="m-0"), width="auto"),
                            ], className="g-2 align-items-center justify-content-center"),
                            className="bg-transparent border-0"
                        ),

                        dbc.CardBody([

                            html.H5("PS", className="text-info text-center mb-3"),

                            dbc.Row([
                                # PS %
                                dbc.Col(
                                    html.Div([
                                        html.H6("%", className="mb-2 text-center"),
                                        dcc.Loading(
                                            dcc.Graph(id="topoff-hi-pct-ps", config={"displayModeBar": False}),
                                            type="default"
                                        )
                                    ], className="histo-wrap"),
                                    md=6
                                ),

                                # PS UNIT
                                dbc.Col(
                                    html.Div([
                                        html.H6("UNIT", className="mb-2 text-center"),
                                        dcc.Loading(
                                            dcc.Graph(id="topoff-hi-unit-ps", config={"displayModeBar": False}),
                                            type="default"
                                        )
                                    ], className="histo-wrap"),
                                    md=6
                                ),
                            ], className="g-3"),

                            html.Hr(className="my-4"),

                            html.H5("CS", className="text-warning text-center mb-3"),

                            dbc.Row([
                                # CS %
                                dbc.Col(
                                    html.Div([
                                        html.H6("%", className="mb-2 text-center"),
                                        dcc.Loading(
                                            dcc.Graph(id="topoff-hi-pct-cs", config={"displayModeBar": False}),
                                            type="default"
                                        )
                                    ], className="histo-wrap"),
                                    md=6
                                ),

                                # CS UNIT
                                dbc.Col(
                                    html.Div([
                                        html.H6("UNIT", className="mb-2 text-center"),
                                        dcc.Loading(
                                            dcc.Graph(id="topoff-hi-unit-cs", config={"displayModeBar": False}),
                                            type="default"
                                        )
                                    ], className="histo-wrap"),
                                    md=6
                                ),
                            ], className="g-3"),

                        ], className="p-2"),
                    ], className="bg-dark text-white border-0 shadow-sm mb-2"),
                )
            ]),

        ]),

        # Stores
        dcc.Store(id="defaults-store", data={"fecha": default_date_str(), "hora": default_hour_str()}),
        dcc.Store(id="dt-manual-store", data={"last_manual_ts": 0}),
        dcc.Store(id="page-state", data={"page": 1, "page_size": 5}),
        dcc.Store(id="sort-state", data={"column": None, "ascending": True}),
        dcc.Store(id="table-page-data"),
        dcc.Store(id="main-context-store"),

        # Señales/estado Heatmap
        dcc.Store(id="heatmap-trigger", data=None),
        dcc.Store(id="heatmap-page-state", data={"page": 1, "page_size": 50}),
        dcc.Store(id="heatmap-page-info"),
        dcc.Store(id="heatmap-params"),

        dcc.Store(id="histo-trigger", data=None),
        dcc.Store(id="histo-page-state", data={"page": 1, "page_size": 50}),
        dcc.Store(id="histo-page-info"),
        dcc.Store(id="histo-params"),
        dcc.Store(id="histo-selected-wave"),

        dcc.Store(id="topoff-page-state", data={"page": 1, "page_size": 50}),
        dcc.Store(id="topoff-sort-state", data={"column": None, "ascending": True}),
        dcc.Store(id="topoff-sort-last-ts", data=0),

        # Stores Heatmap TopOff
        dcc.Store(id="topoff-heatmap-trigger", data=None),
        dcc.Store(id="topoff-heatmap-page-state", data={"page": 1, "page_size": 50}),
        dcc.Store(id="topoff-heatmap-page-info"),

        dcc.Store(id="topoff-histo-trigger"),
        dcc.Store(id="topoff-histo-page-info"),
        dcc.Store(id="topoff-histo-selected-wave", data={}),

        dcc.Interval(id="refresh-timer", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dcc.Download(id="download-excel"),
        dcc.Store(id="topoff-link-state", data={"selected": None}),
        dcc.Store(id="topoff-cluster-mode", data={"mode": "full"}),
        html.Div(id="topoff-scroll-dummy", style={"display": "none"}),

        dcc.Store(id="heatmap-integrity-page-state", data={"page": 1, "page_size": 50}),
        dcc.Store(id="heatmap-integrity-page-info"),
    ],
        fluid=True,
        style={"backgroundColor": "#121212", "color": "white"}
    )
