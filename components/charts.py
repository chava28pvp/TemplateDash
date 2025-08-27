# charts.py
from dash import dcc
import plotly.express as px
from src.utils import HEADER_MAP  # {"total_erlangs_nocperf":"CS_TRAFF_ERL", ...}

def line_by_time_multi(df, metrics):
    """
    Dibuja múltiples métricas a la vez.
    Cada métrica se convierte en una serie con color distinto.
    """
    if df is None or df.empty:
        return dcc.Graph(
            figure={"layout": {"annotations":[{"text":"Sin datos","xref":"paper","yref":"paper","showarrow":False}]}},
            config={"displayModeBar": False}
        )

    # Filtra sólo métricas presentes
    metrics = [m for m in (metrics or []) if m in df.columns]
    if not metrics:
        return dcc.Graph(
            figure={"layout": {"annotations":[{"text":"Sin métricas válidas","xref":"paper","yref":"paper","showarrow":False}]}},
            config={"displayModeBar": False}
        )

    df = df.copy()
    df["fecha_hora"] = df["fecha"].astype(str) + " " + df["hora"].astype(str)

    # Wide -> Long: una fila por (fecha_hora, métrica, valor)
    long_df = (
        df.sort_values(["fecha", "hora"])[["fecha_hora"] + metrics]
          .melt(id_vars="fecha_hora", value_vars=metrics,
                var_name="metric", value_name="valor")
    )

    # Etiqueta amigable para la leyenda/tooltip
    long_df["metric_header"] = long_df["metric"].map(lambda c: HEADER_MAP.get(c, c))

    fig = px.line(
        long_df,
        x="fecha_hora",
        y="valor",
        color="metric_header",  # 👈 color por nombre amigable de la métrica
        markers=True,
        title="Métricas vs tiempo",
        labels={
            "fecha_hora": "Tiempo",
            "valor": "Valor",
            "metric_header": "Métrica"
        }
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Tiempo",
        yaxis_title="Valor"
    )
    return dcc.Graph(figure=fig, config={"displaylogo": False})
