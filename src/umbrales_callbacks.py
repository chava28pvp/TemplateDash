from dash import Input, Output, State, callback, ctx, html, dcc
import json
import dash_bootstrap_components as dbc
from src.Utils.umbrales_manager import UmbralesManager

umbrales_manager = UmbralesManager()


@callback(
    Output("umbral-metric-selector", "options"),
    Input("umbral-config-modal", "is_open")
)
def load_metric_options(is_open):
    if is_open:
        metrics = umbrales_manager.get_all_metrics()
        return [{"label": umbrales_manager.get_umbral(m)['nombre'], "value": m} for m in metrics]
    return []


@callback(
    Output("umbral-config-container", "children"),
    Input("umbral-metric-selector", "value")
)
def show_metric_config(selected_metric):
    if not selected_metric:
        return "Selecciona una métrica para configurar"

    config = umbrales_manager.get_umbral(selected_metric)

    config_inputs = []
    for i, nivel in enumerate(config['niveles']):
        config_inputs.append(
            dbc.Row([
                dbc.Col([
                    html.Label(nivel['nombre'], className="fw-bold"),
                    dcc.Input(
                        value=nivel['limite'],
                        type="number",
                        step="0.1",
                        id={"type": "umbral-limit", "index": i},
                        className="form-control"
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Color"),
                    dcc.Input(
                        value=nivel['color'],
                        type="text",
                        id={"type": "umbral-color", "index": i},
                        className="form-control"
                    )
                ], width=3)
            ], className="mb-2")
        )

    return config_inputs


@callback(
    Output("umbral-color-preview", "children"),
    Input({"type": "umbral-color", "index": "ALL"}, "value")
)
def show_color_preview(colors):
    if not colors:
        return ""

    previews = []
    for i, color in enumerate(colors):
        if color:
            previews.append(
                html.Div([
                    html.Div(style={
                        "width": "50px",
                        "height": "50px",
                        "backgroundColor": color,
                        "border": "1px solid #ccc"
                    }),
                    html.Small(f"Nivel {i + 1}")
                ], className="d-inline-block mx-2 text-center")
            )

    return previews


@callback(
    Output("umbral-config-modal", "is_open", allow_duplicate=True),
    Input("umbral-save-btn", "n_clicks"),
    State("umbral-metric-selector", "value"),
    State({"type": "umbral-limit", "index": "ALL"}, "value"),
    State({"type": "umbral-color", "index": "ALL"}, "value"),
    prevent_initial_call=True
)
def save_umbral_config(n_clicks, metric, limits, colors):
    if n_clicks and metric:
        config = umbrales_manager.get_umbral(metric)
        for i, (limite, color) in enumerate(zip(limits, colors)):
            if i < len(config['niveles']):
                config['niveles'][i]['limite'] = float(limite)
                config['niveles'][i]['color'] = color

        umbrales_manager.update_umbral(metric, config)
        return False
    return True