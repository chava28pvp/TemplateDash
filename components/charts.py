from dash import dcc
import plotly.express as px

def line_by_time(df, y_col="total_mbytes_nocperf", color_col="vendor"):
    if df is None or df.empty:
        return dcc.Graph(figure={"layout": {"annotations":[{"text":"Sin datos","xref":"paper","yref":"paper","showarrow":False}]}}, config={"displayModeBar": False})

    # Convertir a tiempo para eje X
    # Concatenamos fecha + hora
    df = df.copy()
    df["fecha_hora"] = df["fecha"].astype(str) + " " + df["hora"].astype(str)

    fig = px.line(
        df.sort_values(["fecha","hora"]),
        x="fecha_hora", y=y_col, color=color_col,
        markers=True,
        title=f"{y_col} vs tiempo"
    )
    fig.update_layout(
        margin=dict(l=10,r=10,t=50,b=10),
        xaxis_title="Tiempo",
        yaxis_title=y_col
    )
    return dcc.Graph(figure=fig, config={"displaylogo": False})
