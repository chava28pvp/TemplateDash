from dash import html, dcc
import dash_bootstrap_components as dbc

from src.Utils.umbrales.umbrales_manager import UM_MANAGER


def create_umbral_config_modal():
    metrics = UM_MANAGER.list_metrics()

    return html.Div([
        dbc.Button(
            "Configurar umbrales",
            id="open-umbral-config",
            color="secondary",
            className="ms-2",
        ),
        dcc.Store(id="umbral-config-store"),  # holds full JSON config in-memory
        dbc.Toast(
            id="umbral-toast",
            header="Umbrales",
            is_open=False,
            duration=3000,
            icon="success",
            dismissable=True,
            style={"position": "fixed", "top": 10, "right": 10, "zIndex": 2000},
        ),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Configurar umbrales")),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Métrica a configurar"),
                        dcc.Dropdown(
                            id="umbral-metric",
                            options=[{"label": m, "value": m} for m in metrics],
                            placeholder="Seleccione una métrica…",
                            clearable=False,
                        ),
                        html.Small(
                            id="umbral-metric-help",
                            className="text-muted d-block mt-1",
                        ),
                    ])
                ], className="mb-3"),

                # Severity panel (4 colors)
                html.Div([
                    dbc.Badge("4 niveles", color="primary", className="me-2"),
                    html.Span("Use valores de corte apropiados según la orientación."),
                    dbc.Row([
                        dbc.Col(_severity_card("excelente", "#2ecc71"), md=6),
                        dbc.Col(_severity_card("bueno", "#f1c40f"), md=6),
                        dbc.Col(_severity_card("regular", "#e67e22"), md=6),
                        dbc.Col(_severity_card("critico", "#e74c3c"), md=6),
                    ], className="g-3 mt-1"),
                    html.Small(
                        "Si 'mayor es mejor', estos son mínimos por categoría. Si 'menor es mejor', son máximos.",
                        className="text-muted",
                    ),
                ], id="severity-panel", hidden=True, className="mb-3"),

                # Progress panel (min/max)
                html.Div([
                    dbc.Badge("Progress (min/max)", color="info", className="me-2"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="progress-min", type="number", placeholder="Mínimo", step="any"), md=6),
                        dbc.Col(dbc.Input(id="progress-max", type="number", placeholder="Máximo", step="any"), md=6),
                    ], className="g-3 mt-1"),
                    html.Small("El color del progress es fijo; aquí sólo se define el rango.", className="text-muted"),
                ], id="progress-panel", hidden=True, className="mb-3"),

                dbc.Alert(id="umbral-error", color="danger", is_open=False, className="py-2"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancelar", id="umbral-cancel", color="secondary", className="me-2"),
                dbc.Button("Guardar", id="umbral-save", color="primary"),
            ]),
        ], id="umbral-config-modal", is_open=False, centered=True, size="lg"),
    ])


def _severity_card(label: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(style={"display": "inline-block", "width": 12, "height": 12, "background": color, "borderRadius": 2, "marginRight": 6}),
                html.Strong(label.title()),
            ], className="mb-2 d-flex align-items-center"),
            dbc.Input(id=f"sev-{label}", type="number", placeholder="Valor", step="any"),
        ]), className="h-100"
    )