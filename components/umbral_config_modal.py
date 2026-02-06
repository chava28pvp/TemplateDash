from dash import html, dcc
import dash_bootstrap_components as dbc

from src.Utils.umbrales.umbrales_manager import UM_MANAGER


def create_umbral_config_modal(network_options=None):
    """
    Crea el UI (botón + modal) para configurar umbrales.
    Incluye:
    - Selector de tabla/perfil (main / topoff)
    - Selector de métrica (desde UM_MANAGER)
    - Selector de network (Global o una network específica)
    - Panel de severidad (excelente/bueno/regular/crítico)
    - Panel de progress (min/max)
    - Store + Toast para guardar/feedback (lo controlan callbacks)
    """

    # Opciones de tablas/perfiles disponibles
    table_options = [
        {"label": "Tabla principal (Main)", "value": "main"},
        {"label": "TopOff", "value": "topoff"},
    ]

    # Lista de métricas disponibles (centralizada en UM_MANAGER)
    metrics = UM_MANAGER.list_metrics()

    # Selector de network: "" significa Global (sin scope por red)
    net_options = [{"label": "(Global)", "value": ""}]
    if network_options:
        net_options += [{"label": n, "value": n} for n in network_options]

    return html.Div([
        # Botón que abre el modal (callback lo usa con el id)
        dbc.Button(
            "Configurar umbrales",
            id="open-umbral-config",
            color="secondary",
            className="ms-2",
        ),

        # Store para mantener el JSON de config en memoria (lo actualizan callbacks)
        dcc.Store(id="umbral-config-store"),

        # Toast para mostrar “Guardado” o errores rápidos
        dbc.Toast(
            id="umbral-toast",
            header="Umbrales",
            is_open=False,
            duration=3000,
            icon="success",
            dismissable=True,
            style={"position": "fixed", "top": 10, "right": 10, "zIndex": 2000},
        ),

        # Modal completo de configuración
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Configurar umbrales")),
            dbc.ModalBody([

                # Tabla / perfil al que aplica la config
                dbc.Row([
                    dbc.Col([
                        html.Label("Tabla / Perfil"),
                        dcc.Dropdown(
                            id="umbral-table",
                            options=table_options,
                            value="main",
                            clearable=False,
                        ),
                        html.Small(
                            "Selecciona qué tabla vas a configurar",
                            className="text-muted d-block mt-1",
                        ),
                    ], md=8),
                ], className="mb-3"),

                # Métrica a configurar
                dbc.Row([
                    dbc.Col([
                        html.Label("Métrica a configurar"),
                        dcc.Dropdown(
                            id="umbral-metric",
                            options=[{"label": m, "value": m} for m in metrics],
                            placeholder="Seleccione una métrica…",
                            clearable=False,
                        ),
                        # Texto de ayuda dinámico (lo puede llenar un callback)
                        html.Small(
                            id="umbral-metric-help",
                            className="text-muted d-block mt-1",
                        ),
                    ])
                ], className="mb-3"),

                # Network scope (Global o network específica)
                dbc.Row([
                    dbc.Col([
                        html.Label("Network"),
                        dcc.Dropdown(
                            id="umbral-network",
                            options=net_options,
                            value="",  # "" == Global
                            clearable=False,
                        ),
                        html.Small(
                            "Selecciona una network.",
                            className="text-muted d-block mt-1",
                        ),
                    ], md=8),
                ], className="mb-3"),

                # Panel Severidad (4 niveles). Se muestra/oculta según la métrica/callback.
                html.Div([
                    dbc.Badge("4 niveles", color="primary", className="me-2"),
                    dbc.Row([
                        dbc.Col(_severity_card("excelente", "#2ecc71"), md=6),
                        dbc.Col(_severity_card("bueno", "#f1c40f"), md=6),
                        dbc.Col(_severity_card("regular", "#e67e22"), md=6),
                        dbc.Col(_severity_card("critico", "#e74c3c"), md=6),
                    ], className="g-3 mt-1"),
                ], id="severity-panel", hidden=True, className="mb-3"),

                # Panel Progress (min/max). Se muestra/oculta según la métrica/callback.
                html.Div([
                    dbc.Badge("Progress (min/max)", color="info", className="me-2"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="progress-min", type="number", placeholder="Mínimo", step="any"), md=6),
                        dbc.Col(dbc.Input(id="progress-max", type="number", placeholder="Máximo", step="any"), md=6),
                    ], className="g-3 mt-1"),
                ], id="progress-panel", hidden=True, className="mb-3"),

                # Alert para mostrar errores de validación (lo controla un callback)
                dbc.Alert(id="umbral-error", color="danger", is_open=False, className="py-2"),
            ]),

            # Botones del modal (Cancel / Save)
            dbc.ModalFooter([
                dbc.Button("Cancelar", id="umbral-cancel", color="secondary", className="me-2"),
                dbc.Button("Guardar", id="umbral-save", color="primary"),
            ]),
        ], id="umbral-config-modal", is_open=False, centered=True, size="sm"),
    ])


def _severity_card(label: str, color: str) -> dbc.Card:
    """
    Tarjeta simple para capturar un umbral de severidad:
    - Muestra el nombre (excelente/bueno/regular/critico) con un cuadrito de color
    - Input numérico para el valor del corte
    """
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                # “Swatch” de color
                html.Span(style={
                    "display": "inline-block", "width": 12, "height": 12,
                    "background": color, "borderRadius": 2, "marginRight": 6
                }),
                html.Strong(label.title()),
            ], className="mb-2 d-flex align-items-center"),

            # Input del umbral para ese nivel
            dbc.Input(id=f"sev-{label}", type="number", placeholder="Valor", step="any"),
        ]),
        className="h-100"
    )
