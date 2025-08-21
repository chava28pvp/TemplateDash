from dash import dash_table, html
import dash_bootstrap_components as dbc
from config import THRESHOLDS, COLORS


def create_progress_bar(value, max_value=100, reverse=False):
    """Crea una barra de progreso para celdas de tabla"""
    if reverse:
        percentage = 100 - min((value / max_value) * 100, 100)
        value_display = value
    else:
        percentage = min((value / max_value) * 100, 100)
        value_display = value

    # Determinar color basado en el valor (para KPIs de calidad)
    if value >= THRESHOLDS.get('ps_failure_rrc_percent', {}).get('critico', 2.0):
        color = COLORS['critico']
    elif value >= THRESHOLDS.get('ps_failure_rrc_percent', {}).get('advertencia', 1.5):
        color = COLORS['advertencia']
    else:
        color = COLORS['ok']

    return html.Div([
        html.Div(
            style={
                'width': f'{percentage}%',
                'height': '20px',
                'backgroundColor': color,
                'borderRadius': '3px',
                'transition': 'width 0.5s'
            }
        ),
        html.Span(f'{value_display:.2f}%', style={'marginLeft': '10px', 'fontSize': '12px'})
    ], style={'display': 'flex', 'alignItems': 'center', 'width': '100%'})


def apply_threshold_styling(column_name, thresholds_config):
    """Genera estilos condicionales basados en umbrales para telecom KPIs"""
    styles = []

    if column_name in thresholds_config:
        thresholds = thresholds_config[column_name]

        # Para métricas de fallo (valores más altos son peores)
        if column_name.endswith(('_percent', '_rate')):
            if 'critico' in thresholds:
                styles.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {thresholds["critico"]}',
                        'column_id': column_name
                    },
                    'backgroundColor': COLORS['critico'],
                    'color': 'white',
                    'fontWeight': 'bold'
                })

            if 'advertencia' in thresholds:
                styles.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {thresholds["advertencia"]} && {{{column_name}}} < {thresholds["critico"] if "critico" in thresholds else "100"}',
                        'column_id': column_name
                    },
                    'backgroundColor': COLORS['advertencia'],
                    'color': 'black'
                })

            if 'ok' in thresholds:
                styles.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} < {thresholds["advertencia"] if "advertencia" in thresholds else "100"}',
                        'column_id': column_name
                    },
                    'backgroundColor': COLORS['ok'],
                    'color': 'black'
                })

        # Para métricas de éxito (valores más altos son mejores)
        if column_name.startswith('lcs_'):
            if 'excelente' in thresholds:
                styles.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {thresholds["excelente"]}',
                        'column_id': column_name
                    },
                    'backgroundColor': COLORS['excelente'],
                    'color': 'black'
                })

            if 'bueno' in thresholds:
                styles.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {thresholds["bueno"]} && {{{column_name}}} < {thresholds["excelente"] if "excelente" in thresholds else "100"}',
                        'column_id': column_name
                    },
                    'backgroundColor': COLORS['bueno'],
                    'color': 'black'
                })

            if 'regular' in thresholds:
                styles.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} < {thresholds["bueno"] if "bueno" in thresholds else "100"}',
                        'column_id': column_name
                    },
                    'backgroundColor': COLORS['regular'],
                    'color': 'white',
                    'fontWeight': 'bold'
                })

    return styles


def format_telecom_value(value, column_name):
    """Formatea valores según el tipo de métrica"""
    if value is None:
        return "N/A"

    if column_name.endswith('_percent') or column_name.startswith('lcs_'):
        return f"{float(value):.2f}%"
    elif any(x in column_name for x in ['mbytes', 'gb', 'erlang', 'amr']):
        num_value = float(value)
        return f"{num_value:,.0f}" if num_value >= 1000 else f"{num_value:,.2f}"
    elif isinstance(value, (int, float)):
        return f"{value:,.0f}"
    else:
        return str(value)


def create_telecom_data_table(df, id_suffix="main"):
    """Crea una tabla especializada para KPIs de telecomunicaciones"""

    if df.empty:
        return dbc.Alert("No hay datos disponibles para mostrar en la tabla", color="warning")

    # Columnas a mostrar y sus nombres amigables
    column_mapping = {
        'fecha': 'Fecha',
        'region': 'Región',
        'total_mbytes_nocperf': 'Tráfico Datos (MB)',
        'delta_total_mbytes_nocperf': 'Δ Tráfico Datos',
        'ps_failure_rrc_percent': 'Fallos PS RRC (%)',
        'ps_failures_rab_percent': 'Fallos PS RAB (%)',
        'lcs_ps_rate': 'Tasa Éxito PS (%)',
        'ps_abnormal_releases': 'Liberaciones Anormales PS',
        'total_erlangs_nocperf': 'Tráfico Voz (Erlangs)',
        'delta_total_erlangs_nocperf': 'Δ Tráfico Voz',
        'cs_failures_rrc_percent': 'Fallos CS RRC (%)',
        'cs_failures_rab_percent': 'Fallos CS RAB (%)',
        'lcs_cs_rate': 'Tasa Éxito CS (%)',
        'cs_abnormal_releases': 'Liberaciones Anormales CS',
        'traffic_gb_att': 'Tráfico GB ATT',
        'delta_traffic_gb_att': 'Δ Tráfico GB',
        'traffic_amr_att': 'Tráfico AMR ATT',
        'delta_traffic_amr_att': 'Δ Tráfico AMR'
    }

    # Configurar columnas
    columns = []
    style_data_conditional = []

    available_columns = [col for col in column_mapping.keys() if col in df.columns]

    for col in available_columns:
        column_config = {
            'name': column_mapping[col],
            'id': col,
            'deletable': False,
            'selectable': True,
            'type': 'numeric' if df[col].dtype in ['int64', 'float64'] else 'text'
        }

        # Aplicar estilos de umbral
        style_data_conditional.extend(apply_threshold_styling(col, THRESHOLDS))

        columns.append(column_config)

    # Preparar datos para la tabla
    table_data = []
    for _, row in df.iterrows():
        record = {}
        for col in available_columns:
            record[col] = format_telecom_value(row[col], col)
        table_data.append(record)

    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4("Datos Detallados de Telecomunicaciones", className="mb-0"),
                html.Small("Haz clic en los encabezados para ordenar, usa el buscador para filtrar",
                           className="text-muted")
            ]),
            dbc.CardBody([
                dash_table.DataTable(
                    id=f'telecom-table-{id_suffix}',
                    columns=columns,
                    data=table_data,
                    page_size=10,
                    sort_action='native',
                    filter_action='native',
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '12px',
                        'minWidth': '100px'
                    },
                    style_header={
                        'backgroundColor': 'rgb(70, 130, 180)',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data={
                        'backgroundColor': 'white',
                        'color': 'black'
                    },
                    style_data_conditional=style_data_conditional,
                    export_format='csv',
                    tooltip_data=[],
                    tooltip_duration=None,
                    style_table={
                        'overflowX': 'auto',
                        'maxHeight': '600px',
                        'overflowY': 'auto'
                    },
                    css=[{
                        'selector': '.dash-spreadsheet-container',
                        'rule': 'border: 1px solid #ddd; border-radius: 5px;'
                    }],
                    page_current=0,
                    page_action='native'
                )
            ])
        ])
    ])


def create_telecom_kpi_card(title, value, id_suffix, delta=None, unit=""):
    """Crea una tarjeta KPI especializada para telecom"""
    delta_element = html.P(
        f"{delta:+,}{unit}" if delta is not None else "",
        className="card-text text-center text-success" if delta and delta > 0 else
        "card-text text-center text-danger" if delta else
        "card-text text-center"
    ) if delta is not None else html.P("")

    return dbc.Card([
        dbc.CardHeader(title, className="text-center"),
        dbc.CardBody([
            html.H4(f"{value}{unit}", id=f"kpi-{id_suffix}", className="card-title text-center"),
            delta_element
        ])
    ], className="text-center m-2 telecom-kpi-card")