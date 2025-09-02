from dash import html, dcc
import dash_bootstrap_components as dbc


def create_umbral_config_modal():
    """Crea el modal para configurar umbrales"""
    return dbc.Modal([
        dbc.ModalHeader([
            html.H4("Configuración de Umbrales", className="mb-0")
        ]),
        dbc.ModalBody([
            # Selector de métrica
            dbc.Row([
                dbc.Col([
                    html.Label("Seleccionar Métrica:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id="umbral-metric-selector",
                        placeholder="Elige una métrica...",
                        clearable=False,
                        className="mb-3"
                    ),
                ], width=12)
            ]),

            # Configuración de niveles
            html.Div(id="umbral-config-container", className="mb-3"),

            # Vista previa de colores
            html.Div(id="umbral-color-preview", className="mt-3 p-3 border rounded")
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "↺ Restablecer Default",
                id="umbral-reset-btn",
                color="secondary",
                className="me-2"
            ),
            dbc.Button(
                "💾 Guardar",
                id="umbral-save-btn",
                color="primary",
                className="me-2"
            ),
            dbc.Button(
                "✕ Cerrar",
                id="umbral-close-btn",
                color="light"
            )
        ])
    ],
        id="umbral-config-modal",
        size="lg",
        is_open=False,
        centered=True,
        backdrop="static"  # El modal no se cierra al hacer clic fuera
    )