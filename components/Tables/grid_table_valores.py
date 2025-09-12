# === grid_valores_heatmaps_only.py ===
import pandas as pd
from datetime import datetime, timedelta

# =========================
# Config
# =========================
VALORES_MAP = {
    "PS_RCC":  ("ps_rrc_ia_percent", "ps_rrc_fail"),
    "CS_RCC":  ("cs_rrc_ia_percent", "cs_rrc_fail"),
    "PS_RAB":  ("ps_rab_ia_percent", "ps_rab_fail"),
    "CS_RAB":  ("cs_rab_ia_percent", "cs_rab_fail"),
    "PS_DROP": ("ps_drop_dc_percent", "ps_drop_abnrel"),
    "CS_DROP": ("cs_drop_dc_percent", "cs_drop_abnrel"),
}

# =========================
# Helpers
# =========================
def _infer_networks(df_long: pd.DataFrame) -> list[str]:
    if df_long is None or df_long.empty or "network" not in df_long.columns:
        return []
    return sorted(df_long["network"].dropna().unique().tolist())

def _day_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def _safe_hour_to_idx(hhmmss) -> int | None:
    try:
        hh = int(str(hhmmss).split(":")[0])
        if 0 <= hh <= 23:
            return hh
    except Exception:
        pass
    return None

def _empty24():
    return [None]*24

def _max_date_str(series: pd.Series) -> str | None:
    try:
        return max(pd.to_datetime(series).dt.date).strftime("%Y-%m-%d")
    except Exception:
        return None

def _key_label(tech, vend, clus, net, valores):
    return f"{tech}/{vend}/{clus}/{net}/{valores}"

def build_series_index(df_ts: pd.DataFrame, metrics: set[str]) -> dict:
    """
    Indexa df_ts por (fecha, technology, vendor, cluster, network, metric) -> lista 24
    """
    if df_ts is None or df_ts.empty:
        return {}

    need_cols = {"fecha","hora","technology","vendor","noc_cluster","network"} | metrics
    miss = [c for c in need_cols if c not in df_ts.columns]
    if miss:
        return {}

    df = df_ts.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")

    index = {}
    cols_key = ["fecha","technology","vendor","noc_cluster","network"]
    for (fecha, tech, vend, clus, net), grp in df.groupby(cols_key, sort=False):
        for metric in metrics:
            out = _empty24()
            if metric in grp.columns:
                for _, r in grp.iterrows():
                    idx = _safe_hour_to_idx(r["hora"])
                    if idx is None:
                        continue
                    val = r.get(metric, None)
                    out[idx] = (val if isinstance(val,(int,float)) else None)
            index[(fecha, tech, vend, clus, net, metric)] = out
    return index

# =========================
# Payloads de heatmap (48 columnas: Ayer 0–23 | Hoy 24–47)
# =========================
def build_heatmap_payloads(
    df_meta: pd.DataFrame,
    df_ts: pd.DataFrame,
    *,
    networks=None,
    valores_order=("PS_RCC","CS_RCC","PS_DROP","CS_DROP","PS_RAB","CS_RAB"),
    today=None,
    yday=None
):
    """
    Devuelve (pct_payload, unit_payload) para graficar dos heatmaps.
    NOTA: df_meta define qué filas aparecen (normalmente tu snapshot paginado).
    """
    if df_meta is None or df_meta.empty:
        return None, None

    # redes
    if networks is None or not networks:
        networks = _infer_networks(df_ts if df_ts is not None else df_meta)
    if not networks:
        return None, None

    # Fechas
    if today is None:
        today = _max_date_str(df_ts["fecha"]) if (df_ts is not None and "fecha" in df_ts.columns) else _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Métricas requeridas
    metrics_needed = set()
    for v in valores_order:
        pm, um = VALORES_MAP.get(v, (None, None))
        if pm: metrics_needed.add(pm)
        if um: metrics_needed.add(um)

    # Index
    series_index = build_series_index(df_ts, metrics_needed)
    def _series(fecha, tech, vend, clus, net, metric):
        return series_index.get((fecha, tech, vend, clus, net, metric), _empty24())

    # Ejes
    x_labels = [f"Ay {h:02d}" for h in range(24)] + [f"Hoy {h:02d}" for h in range(24)]
    z_pct, z_unit, y_labels = [], [], []

    # Filas de df_meta (visible)
    meta_cols = ["technology", "vendor", "noc_cluster"]
    meta = df_meta.drop_duplicates(subset=meta_cols)[meta_cols].reset_index(drop=True)

    for _, r in meta.iterrows():
        tech = r["technology"]; vend = r["vendor"]; clus = r["noc_cluster"]
        for valores in valores_order:
            percent_metric, unit_metric = VALORES_MAP.get(valores, (None, None))
            for net in networks:
                label = _key_label(tech, vend, clus, net, valores)
                # % (48)
                if percent_metric:
                    s_t = _series(today, tech, vend, clus, net, percent_metric) or []
                    s_y = _series(yday,  tech, vend, clus, net, percent_metric) or []
                    row48_pct = [None]*48
                    for h in range(24):
                        row48_pct[h]      = s_y[h] if s_y[h] is not None else None
                        row48_pct[24 + h] = s_t[h] if s_t[h] is not None else None
                    z_pct.append(row48_pct)
                else:
                    z_pct.append([None]*48)
                # UNIT (48)
                if unit_metric:
                    s_tu = _series(today, tech, vend, clus, net, unit_metric) or []
                    s_yu = _series(yday,  tech, vend, clus, net, unit_metric) or []
                    row48_unit = [None]*48
                    for h in range(24):
                        row48_unit[h]      = s_yu[h] if s_yu[h] is not None else None
                        row48_unit[24 + h] = s_tu[h] if s_tu[h] is not None else None
                    z_unit.append(row48_unit)
                else:
                    z_unit.append([None]*48)
                y_labels.append(label)

    # %: 0–100 fijo
    pct_payload = {
        "z": z_pct, "x": x_labels, "y": y_labels,
        "zmin": 0, "zmax": 100, "title": "% IA / % DC"
    }
    # UNIT: rango dinámico
    flat_unit = [v for row in z_unit for v in row if isinstance(v,(int,float))]
    umin = (min(flat_unit) if flat_unit else 0)
    umax = (max(flat_unit) if flat_unit else 1)
    if umin == umax:
        umax = (umin or 1)
    unit_payload = {
        "z": z_unit, "x": x_labels, "y": y_labels,
        "zmin": umin, "zmax": umax, "title": "Unidades (fails / abnrel)"
    }
    return pct_payload, unit_payload


# =========================
# Figura de Heatmap (Plotly)
# =========================
def build_heatmap_figure(payload, *, height=720, colorscale="Inferno"):
    import plotly.graph_objs as go
    if not payload:
        return go.Figure()
    z = payload["z"]; x = payload["x"]; y = payload["y"]
    zmin = payload["zmin"]; zmax = payload["zmax"]
    title = payload.get("title","")
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(title=title),
        hovertemplate="Fila %{y}<br>%{x}: %{z}<extra></extra>",
        xgap=0.2, ygap=0.2,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=90, r=16, t=10, b=40),
        xaxis=dict(title="Horas (Ayer | Hoy)", tickangle=0, automargin=True),
        yaxis=dict(title="Registro / Net / Valor", automargin=True),
        uirevision="keep",
    )
    return fig
