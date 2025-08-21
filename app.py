from dash import Dash, Input, Output, State, callback_context, html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Importar componentes propios
from database.db_connection import get_telecom_data, get_available_regions, get_date_range
from components.tables import create_telecom_data_table, create_telecom_kpi_card
from components.charts import create_telecom_timeseries_chart, create_telecom_quality_gauge

# Inicializar la aplicación
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Telecom Dashboard - Datos Reales"

# Obtener información inicial de la BD
min_date, max_date = get_date_range() or (datetime.now() - timedelta(days=30), datetime.now())
regions = get_available_regions()

app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            html.H2("📶 Telecom Dashboard - BD Real", className="navbar-brand mb-0"),
            dbc.Badge("Conectado a MySQL", color="success", className="ms-2")
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    ),

    dbc.Row([
        # Sidebar con filtros
        dbc.Col([
            html.H4("Filtros", className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Rango de Fechas"),
                dbc.CardBody([
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        start_date=min_date,
                        end_date=max_date,
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Regiones"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='region-filter',
                        options=[{'label': region, 'value': region} for region in regions],
                        value=regions,  # Todas las regiones por defecto
                        multi=True,
                        placeholder="Seleccione regiones"
                    )
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardBody([
                    dbc.Button("🔄 Actualizar Datos",
                               id="update-btn",
                               color="primary",
                               className="w-100 mb-2"),
                    dbc.Button("🧹 Limpiar Filtros",
                               id="clear-btn",
                               color="secondary",
                               className="w-100")
                ])
            ]),

            # Indicador de estado
            dbc.Card([
                dbc.CardHeader("Estado del Sistema"),
                dbc.CardBody([
                    html.Div(id='status-indicator', children="✅ Conectado a BD"),
                    html.Hr(),
                    html.Small(f"Última actualización: ", id='last-update'),
                    html.Br(),
                    html.Small(f"Registros cargados: ", id='record-count')
                ])
            ], className="mt-3")

        ], width=3),

        # Contenido principal
        dbc.Col([
                dbc.Row([
                dbc.Col([
                    html.H4("Datos Detallados de la Base de Datos", className="mb-3"),
                    html.Div(id='data-table-container')
                ], width=12)
            ]),
            # KPIs principales
            dbc.Row([
                dbc.Col(create_telecom_kpi_card("Tráfico Datos Total", "0", "total-data", unit=" MB"), width=3),
                dbc.Col(create_telecom_kpi_card("Tráfico Voz Total", "0", "total-voice", unit=" Erlangs"), width=3),
                dbc.Col(create_telecom_kpi_card("Éxito PS", "0", "ps-success", unit="%"), width=3),
                dbc.Col(create_telecom_kpi_card("Éxito CS", "0", "cs-success", unit="%"), width=3),
            ], className="mb-4"),

            # Gráficas
            dbc.Row([
                dbc.Col(dcc.Graph(id='traffic-chart'), width=6),
                dbc.Col(dcc.Graph(id='quality-chart'), width=6),
            ], className="mb-4"),

            # Gauges de calidad
            dbc.Row([
                dbc.Col(dcc.Graph(id='ps-gauge'), width=4),
                dbc.Col(dcc.Graph(id='cs-gauge'), width=4),
                dbc.Col(dcc.Graph(id='overall-gauge'), width=4),
            ], className="mb-4")

            # Tabla de datos

        ], width=9)
    ]),

    # Almacenamiento y actualización automática
    dcc.Store(id='telecom-data-store'),
    dcc.Interval(id='auto-refresh', interval=300000)  # 5 minutos

], fluid=True, style={'backgroundColor': '#f8f9fa'})


# Callback para cargar datos con filtros
@app.callback(
    [Output('telecom-data-store', 'data'),
     Output('last-update', 'children'),
     Output('record-count', 'children'),
     Output('status-indicator', 'children')],
    [Input('update-btn', 'n_clicks'),
     Input('auto-refresh', 'n_intervals'),
     Input('clear-btn', 'n_clicks')],
    [State('date-range-picker', 'start_date'),
     State('date-range-picker', 'end_date'),
     State('region-filter', 'value')]
)
def load_telecom_data(update_clicks, auto_refresh, clear_clicks, start_date, end_date, selected_regions):
    """Carga datos de la base de datos con los filtros aplicados"""

    # Si se hace clic en limpiar, resetear filtros
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-btn.n_clicks':
        selected_regions = regions  # Todas las regiones
        start_date = min_date
        end_date = max_date
        # Note: Los DatePickers se actualizan automáticamente

    print(f"🔍 Cargando datos con filtros:")
    print(f"   Fechas: {start_date} to {end_date}")
    print(f"   Regiones: {selected_regions}")

    try:
        # Cargar datos REALES de la base de datos
        df = get_telecom_data(start_date, end_date, selected_regions)

        if df.empty:
            return [], "❌ No hay datos", "0 registros", "⚠️ BD vacía"

        record_count = f"{len(df)} registros"
        last_update = f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        print(f"✅ Datos cargados exitosamente: {len(df)} registros")
        return df.to_dict('records'), last_update, record_count, "✅ Conectado a BD"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return [], error_msg, "0 registros", "❌ Error de conexión"


# Callback para actualizar el dashboard con los datos
@app.callback(
    [Output('kpi-total-data', 'children'),
     Output('kpi-total-voice', 'children'),
     Output('kpi-ps-success', 'children'),
     Output('kpi-cs-success', 'children'),
     Output('traffic-chart', 'figure'),
     Output('quality-chart', 'figure'),
     Output('ps-gauge', 'figure'),
     Output('cs-gauge', 'figure'),
     Output('overall-gauge', 'figure'),
     Output('data-table-container', 'children')],
    [Input('telecom-data-store', 'data')]
)
def update_dashboard(data):
    """Actualiza todos los componentes del dashboard con datos reales"""

    if not data:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No hay datos disponibles",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return ("0", "0", "0", "0",
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                dbc.Alert("No se encontraron datos con los filtros aplicados", color="warning"))

    df = pd.DataFrame(data)

    # Calcular KPIs con datos REALES
    total_data = df['total_mbytes_nocperf'].sum()
    total_voice = df['total_erlangs_nocperf'].sum()
    avg_ps = df['lcs_ps_rate'].mean()
    avg_cs = df['lcs_cs_rate'].mean()

    print(f"📊 KPIs calculados:")
    print(f"   Tráfico datos: {total_data:,.0f} MB")
    print(f"   Tráfico voz: {total_voice:,.0f} Erlangs")
    print(f"   Éxito PS: {avg_ps:.1f}%")
    print(f"   Éxito CS: {avg_cs:.1f}%")

    # Gráfico de tráfico de datos
    traffic_fig = px.line(df, x='fecha', y='total_mbytes_nocperf', color='region',
                          title='Evolución del Tráfico de Datos por Región',
                          labels={'total_mbytes_nocperf': 'Tráfico (MB)', 'fecha': 'Fecha'})
    traffic_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # Gráfico de calidad del servicio
    quality_fig = px.line(df, x='fecha', y=['lcs_ps_rate', 'lcs_cs_rate'],
                          title='Evolución de la Calidad del Servicio',
                          labels={'value': 'Tasa de Éxito (%)', 'fecha': 'Fecha'})
    quality_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # Gauges de calidad
    def create_gauge(value, title, thresholds=[95, 98, 99]):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, thresholds[0]], 'color': "red"},
                    {'range': [thresholds[0], thresholds[1]], 'color': "yellow"},
                    {'range': [thresholds[1], thresholds[2]], 'color': "lightgreen"},
                    {'range': [thresholds[2], 100], 'color': "green"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    ps_gauge = create_gauge(avg_ps, "Tasa de Éxito PS")
    cs_gauge = create_gauge(avg_cs, "Tasa de Éxito CS")
    overall_gauge = create_gauge((avg_ps + avg_cs) / 2, "Calidad General")

    # Tabla de datos COMPLETA
    table = create_telecom_data_table(df, "main")

    return (f"{total_data:,.0f}", f"{total_voice:,.0f}",
            f"{avg_ps:.1f}", f"{avg_cs:.1f}",
            traffic_fig, quality_fig, ps_gauge, cs_gauge, overall_gauge,
            table)


# Callback para limpiar filtros
@app.callback(
    [Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date'),
     Output('region-filter', 'value')],
    [Input('clear-btn', 'n_clicks')]
)
def clear_filters(n_clicks):
    if n_clicks:
        print("🧹 Limpiando filtros...")
        return min_date, max_date, regions
    return min_date, max_date, regions


if __name__ == '__main__':
    print("🚀 Iniciando Telecom Dashboard...")
    print("📊 Conectando a base de datos MySQL...")
    app.run(debug=True)