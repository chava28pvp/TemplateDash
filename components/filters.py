import dash_bootstrap_components as dbc
from dash import dcc, html
import datetime
from database.db_connection import get_available_regions, get_date_range


def create_date_range_filter():
    min_date, max_date = get_date_range() or (datetime.datetime.now() - datetime.timedelta(days=30),
                                              datetime.datetime.now())

    return dbc.Card([
        dbc.CardHeader("Rango de Fechas"),
        dbc.CardBody([
            dcc.DatePickerRange(
                id='date-range-filter',
                start_date=min_date,
                end_date=max_date,
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            )
        ])
    ], className="mb-3")


def create_region_filter():
    regions = get_available_regions()
    return dbc.Card([
        dbc.CardHeader("Regiones"),
        dbc.CardBody([
            dcc.Dropdown(
                id='region-filter',
                options=[{'label': region, 'value': region} for region in regions],
                value=regions,
                multi=True,
                clearable=True,
                placeholder="Seleccione regiones"
            )
        ])
    ], className="mb-3")


def create_kpi_threshold_filter():
    return dbc.Card([
        dbc.CardHeader("Umbrales de Calidad"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("PS RRC Failure (%)"),
                    dcc.Slider(
                        id='ps-rrc-threshold',
                        min=0,
                        max=5,
                        step=0.1,
                        value=1.5,
                        marks={i: str(i) for i in range(0, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("CS RRC Failure (%)"),
                    dcc.Slider(
                        id='cs-rrc-threshold',
                        min=0,
                        max=3,
                        step=0.1,
                        value=1.0,
                        marks={i: str(i) for i in range(0, 4)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6)
            ])
        ])
    ], className="mb-3")


def create_filter_panel():
    return html.Div([
        html.H4("Filtros", className="mb-3"),
        create_date_range_filter(),
        create_region_filter(),
        create_kpi_threshold_filter(),
        html.Div([
            dbc.Button("🔄 Actualizar", id="apply-filters", color="primary", className="w-100 mb-2"),
            dbc.Button("🧹 Limpiar", id="clear-filters", color="secondary", className="w-100"),
        ])
    ])