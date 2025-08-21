from dash import dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_telecom_timeseries_chart(df, x_col, y_cols, title, color_col=None, aggregation='mean'):
    """
    Crea un gráfico de series temporales para métricas de telecomunicaciones

    Args:
        df (DataFrame): DataFrame con los datos
        x_col (str): Columna para el eje X (generalmente fecha)
        y_cols (list): Lista de columnas para el eje Y
        title (str): Título del gráfico
        color_col (str, optional): Columna para diferenciar por color
        aggregation (str): Tipo de agregación ('mean', 'sum', 'max', 'min')

    Returns:
        figure: Objeto de figura de Plotly
    """
    if df.empty:
        # Devolver gráfico vacío si no hay datos
        fig = go.Figure()
        fig.update_layout(
            title=title,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            annotations=[dict(
                text="No hay datos disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )]
        )
        return fig

    # Crear copia para no modificar el DataFrame original
    chart_df = df.copy()

    # Convertir la columna de fecha si es necesario
    if pd.api.types.is_string_dtype(chart_df[x_col]):
        chart_df[x_col] = pd.to_datetime(chart_df[x_col])

    # Agrupar datos si se especifica una columna de color
    if color_col and color_col in chart_df.columns:
        # Agrupar por fecha y la columna de color
        grouped_df = chart_df.groupby([x_col, color_col]).agg({col: aggregation for col in y_cols}).reset_index()
    else:
        # Agrupar solo por fecha
        grouped_df = chart_df.groupby(x_col).agg({col: aggregation for col in y_cols}).reset_index()

    fig = go.Figure()

    # Colores para las series
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3

    # Para cada métrica Y
    for i, y_col in enumerate(y_cols):
        # Si hay columna de color, crear series separadas por cada valor
        if color_col and color_col in chart_df.columns:
            for j, color_value in enumerate(grouped_df[color_col].unique()):
                filtered_df = grouped_df[grouped_df[color_col] == color_value]

                # Nombre de la serie
                series_name = f"{y_col} - {color_value}"

                fig.add_trace(go.Scatter(
                    x=filtered_df[x_col],
                    y=filtered_df[y_col],
                    name=series_name,
                    mode='lines+markers',
                    line=dict(
                        color=colors[(i * len(grouped_df[color_col].unique()) + j) % len(colors)],
                        width=2
                    ),
                    marker=dict(size=6),
                    hovertemplate=(
                            f"<b>{series_name}</b><br>" +
                            f"{x_col}: %{{x|%Y-%m-%d}}<br>" +
                            f"Valor: %{{y:.2f}}<br>" +
                            "<extra></extra>"
                    )
                ))
        else:
            # Sin columna de color, una serie por métrica
            fig.add_trace(go.Scatter(
                x=grouped_df[x_col],
                y=grouped_df[y_col],
                name=y_col,
                mode='lines+markers',
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                ),
                marker=dict(size=6),
                hovertemplate=(
                        f"<b>{y_col}</b><br>" +
                        f"{x_col}: %{{x|%Y-%m-%d}}<br>" +
                        f"Valor: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                )
            ))

    # Personalizar el layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='Fecha',
            showgrid=False,
            tickformat='%Y-%m-%d',
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig


def create_telecom_quality_gauge(value, title, threshold_low=95, threshold_medium=98, threshold_high=99):
    """Crea un indicador de gauge para métricas de calidad - SOLO LA FIGURA"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold_low], 'color': "red"},
                {'range': [threshold_low, threshold_medium], 'color': "yellow"},
                {'range': [threshold_medium, threshold_high], 'color': "lightgreen"},
                {'range': [threshold_high, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_telecom_comparison_chart(df, x_col, y_col, title, chart_type='bar'):
    """Crea gráfico de comparación entre regiones"""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    if chart_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col, color='region', title=title, barmode='group')
    else:  # line
        fig = px.line(df, x=x_col, y=y_col, color='region', title=title)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode='x unified'
    )
    return fig


def create_telecom_correlation_heatmap(df, metrics, title):
    """Crea un heatmap de correlación entre métricas"""
    if df.empty or len(metrics) < 2:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    correlation_matrix = df[metrics].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        hoverongaps=False,
        hovertemplate='Correlación entre %{x} y %{y}: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Métricas",
        yaxis_title="Métricas",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig