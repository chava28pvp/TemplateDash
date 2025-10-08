from dash import html, dcc
import dash_bootstrap_components as dbc

from src.Utils.umbrales.umbrales_manager import UM_MANAGER


def create_umbral_config_modal(network_options=None):
    """
    Construye el modal para configurar umbrales.
    - Incluye selector de Métrica y de Network (Global/ATT/NET/TEF, etc.)
    - Los paneles de Severity y Progress se rellenarán vía callbacks usando:
        UM_MANAGER.get_severity(metric, network or None)
        UM_MANAGER.get_progress(metric, network or None)
    - Para guardar:
        UM_MANAGER.upsert_severity(metric, thresholds=..., network=(value or None))
        UM_MANAGER.upsert_progress(metric, min_v=..., max_v=..., decimals=..., label=..., network=(value or None))
    - Para restablecer override (botón "Restablecer override"):
        borrar per_network[network] en el manager (o usar un helper clear_network_override).
    """
    metrics = UM_MANAGER.list_metrics()

    # Opciones para el selector de network (ámbito).
    # "" == Global (default). Puedes pasar network_options=["ATT","NET","TEF"] desde fuera.
    net_options = [{"label": "(Global)", "value": ""}]
    if network_options:
        net_options += [{"label": n, "value": n} for n in network_options]

    return html.Div([
        dbc.Button(
            "Configurar umbrales",
            id="open-umbral-config",
            color="secondary",
            className="ms-2",
        ),

        # Guarda el JSON completo de configuración en memoria (lo manejarán los callbacks)
        dcc.Store(id="umbral-config-store"),

        # Toast para feedback (guardado/errores)
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

                # Selector de métrica
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

                # Selector de network (ámbito) + botón de restablecer override
                dbc.Row([
                    dbc.Col([
                        html.Label("Network (ámbito)"),
                        dcc.Dropdown(
                            id="umbral-network",
                            options=net_options,
                            value="",          # "" == Global
                            clearable=False,
                        ),
                        html.Small(
                            "Selecciona una red",
                            className="text-muted d-block mt-1",
                        ),
                    ], md=8),

                ], className="mb-3"),

                # Panel de Severity (4 colores)
                html.Div([
                    dbc.Badge("4 niveles", color="primary", className="me-2"),
                    html.Span("Use valores de corte apropiados según la orientación."),
                    dbc.Row([
                        dbc.Col(_severity_card("excelente", "#2ecc71"), md=6),
                        dbc.Col(_severity_card("bueno", "#f1c40f"), md=6),
                        dbc.Col(_severity_card("regular", "#e67e22"), md=6),
                        dbc.Col(_severity_card("critico", "#e74c3c"), md=6),
                    ], className="g-3 mt-1"),

                ], id="severity-panel", hidden=True, className="mb-3"),

                # Panel de Progress (min/max)
                html.Div([
                    dbc.Badge("Progress (min/max)", color="info", className="me-2"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="progress-min", type="number", placeholder="Mínimo", step="any"), md=6),
                        dbc.Col(dbc.Input(id="progress-max", type="number", placeholder="Máximo", step="any"), md=6),
                    ], className="g-3 mt-1"),

                ], id="progress-panel", hidden=True, className="mb-3"),

                # Área de errores
                dbc.Alert(id="umbral-error", color="danger", is_open=False, className="py-2"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancelar", id="umbral-cancel", color="secondary", className="me-2"),
                dbc.Button("Guardar", id="umbral-save", color="primary"),
            ]),
        ], id="umbral-config-modal", is_open=False, centered=True, size="sm"),
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