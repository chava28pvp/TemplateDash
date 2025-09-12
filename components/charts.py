# components/charts.py
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def _as_metric_list(metrics):
    # Acepta lista o callable(df)->lista
    if callable(metrics):
        try:
            return list(metrics())
        except TypeError:
            # algunos helpers esperan df; deja que el caller pase explícitamente la lista si es el caso
            return []
    return list(metrics or [])

def _build_ts(df):
    """Crea timestamp a partir de fecha+hora en df y ordena."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts"])
    df2 = df.copy()
    # tolera tipos: fecha (date/str), hora ('HH:MM:SS'/str)
    fecha_str = df2["fecha"].astype(str) if "fecha" in df2.columns else ""
    hora_str  = df2["hora"].astype(str)  if "hora"  in df2.columns else "00:00:00"
    df2["ts"] = pd.to_datetime(fecha_str + " " + hora_str, errors="coerce")
    df2 = df2.dropna(subset=["ts"]).sort_values("ts")
    return df2

def line_by_time_multi(df, metrics, *, group_col="network", title="Evolución por hora"):
    """
    Renderiza un gráfico de líneas multi-métrica (opcionalmente agrupado por 'network').
    - df: dataframe filtrado/paginado actual
    - metrics: lista de nombres de columna o callable() -> lista
    - group_col: 'network' por defecto; si no existe, grafica sin agrupar
    """
    if df is None or df.empty:
        return dbc.Alert("Sin datos para graficar.", color="secondary")

    metric_list = _as_metric_list(metrics)
    # si el callable necesitaba df, intenta de nuevo:
    if not metric_list and callable(metrics):
        try:
            metric_list = list(metrics(df))
        except Exception:
            metric_list = []

    # filtra solo métricas disponibles
    metric_list = [m for m in metric_list if m in df.columns]
    if not metric_list:
        return dbc.Alert("No hay métricas disponibles en el dataset actual.", color="secondary")

    dfts = _build_ts(df)
    if dfts.empty:
        return dbc.Alert("No hay timestamps válidos (fecha/hora).", color="secondary")

    fig = go.Figure()

    # agrupación opcional por network (si existe)
    groups = dfts[group_col].dropna().unique().tolist() if group_col in dfts.columns else [None]

    for m in metric_list:
        y = pd.to_numeric(dfts[m], errors="coerce")
        if group_col in dfts.columns:
            for g in groups:
                sel = dfts[dfts[group_col] == g]
                y_sel = pd.to_numeric(sel[m], errors="coerce")
                if sel.empty or not np.isfinite(y_sel).any():
                    continue
                fig.add_trace(go.Scattergl(
                    x=sel["ts"], y=y_sel,
                    mode="lines",
                    name=f"{m} | {g}",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.4g}<extra></extra>",
                ))
        else:
            if not np.isfinite(y).any():
                continue
            fig.add_trace(go.Scattergl(
                x=dfts["ts"], y=y,
                mode="lines",
                name=m,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.4g}<extra></extra>",
            ))

    if not fig.data:
        return dbc.Alert("No hay valores numéricos para las métricas seleccionadas.", color="secondary")

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#202225",
        paper_bgcolor="#202225",
        font=dict(color="#eaeaea", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(zeroline=False, gridcolor="#333")

    return dcc.Graph(figure=fig, config={"displayModeBar": False})
