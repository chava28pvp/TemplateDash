# === grid_valores_heatmaps_only.py ===
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go

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

def _hora_to_int(series) -> pd.Series:
    """
    Convierte 'HH:MM:SS' o enteros/strings a 0..23; inválidos -> NaN.
    """
    s = series.astype(str).str.split(":", n=1, expand=True)[0]
    s = pd.to_numeric(s, errors="coerce")
    s = s.where((s >= 0) & (s <= 23))
    return s


def _key_label(tech, vend, clus, net, valores):
    return f"{tech}/{vend}/{clus}/{net}/{valores}"

def build_series_index(df_ts: pd.DataFrame, metrics: set[str]) -> dict:
    """
    Indexa df_ts por (fecha, technology, vendor, cluster, network, metric) -> lista 24
    """
    if df_ts is None or df_ts.empty:
        return {}

    need_cols = {"fecha","hora","technology","vendor","noc_cluster","network"} | set(metrics)
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
# Payloads de heatmap (48 columnas: Ayer 0–23 | Hoy 24–47) con paginado
# =========================
def build_heatmap_payloads_fast(
    df_meta: pd.DataFrame,
    df_ts: pd.DataFrame,
    *,
    networks=None,
    valores_order=("PS_RCC","CS_RCC","PS_DROP","CS_DROP","PS_RAB","CS_RAB"),
    today=None,
    yday=None,
    alarm_keys=None,
    alarm_only=False,
    offset=0,
    limit=5,
):
    """
    Ultrafast: filtra df_ts a las filas visibles y asigna por vectorización.
    Devuelve (pct_payload, unit_payload, page_info).
    """
    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Redes
    if networks is None or not networks:
        networks = _infer_networks(df_ts if df_ts is not None else df_meta)
    if not networks:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Fechas
    if today is None:
        today = _max_date_str(df_ts["fecha"]) if (df_ts is not None and "fecha" in df_ts.columns) else _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Métricas necesarias
    metrics_needed = set()
    for v in valores_order:
        pm, um = VALORES_MAP.get(v, (None, None))
        if pm: metrics_needed.add(pm)
        if um: metrics_needed.add(um)
    if not metrics_needed:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Meta base sin duplicados
    meta_cols = ["technology", "vendor", "noc_cluster"]
    base = df_meta.drop_duplicates(subset=meta_cols)[meta_cols].reset_index(drop=True)

    # Cross con networks y valores_order (todas las filas posibles)
    rows_full = (
        base.assign(_tmp=1)
        .merge(pd.DataFrame({"network": networks, "_tmp": 1}), on="_tmp", how="left")
        .drop(columns=["_tmp"])
    )
    rows_full = rows_full.assign(key5=rows_full[["technology","vendor","noc_cluster","network"]].astype(str).agg("/".join, axis=1))

    rows_all = []
    for v in valores_order:
        rf = rows_full.copy()
        rf["valores"] = v
        if alarm_only and alarm_keys is not None:
            # alarm_keys son tuplas (tech, vend, clus, net)
            keys_ok = set(alarm_keys)
            mask = list(zip(rf["technology"], rf["vendor"], rf["noc_cluster"], rf["network"]))
            rf = rf[[m in keys_ok for m in mask]]
        rows_all.append(rf)
    if not rows_all:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}
    rows_all = pd.concat(rows_all, ignore_index=True)

    total_rows = len(rows_all)
    if total_rows == 0:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # Paginado de filas visibles
    start = max(0, int(offset))
    end   = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # === Filtra df_ts SOLO a las combinaciones visibles y fechas relevantes ===
    if df_ts is None or df_ts.empty:
        df_small = pd.DataFrame()
    else:
        # Claves visibles (technology, vendor, noc_cluster, network)
        keys_df = rows_page[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)
        keys_df["rid"] = np.arange(len(keys_df), dtype=int)

        # Filter por fecha y merge a claves visibles (esto reduce drásticamente el tamaño)
        df_small = df_ts.loc[
            df_ts["fecha"].astype(str).isin([yday, today]) &
            df_ts["network"].astype(str).isin(keys_df["network"].astype(str))
        ].copy()

        # Join para obtener 'rid' por combinación visible
        df_small = df_small.merge(
            keys_df,
            on=["technology","vendor","noc_cluster","network"],
            how="inner",
            copy=False,
            validate="many_to_one"
        )

        # Hora como 0..23
        df_small["h"] = _hora_to_int(df_small["hora"])
        df_small = df_small.loc[df_small["h"].notna()].copy()
        df_small["h"] = df_small["h"].astype(int)

        # Filtra a solo columnas que interesan (reduce memoria)
        keep_cols = {"fecha","h","rid"} | set(metrics_needed)
        df_small = df_small[[c for c in keep_cols if c in df_small.columns]]

    # === Construye arrays por métrica y por día (vectorizado) ===
    N = len(rows_page)  # filas visibles (por valores), pero arrays por key 4D requieren map filas->rid
    # OJO: 'rid' está por combinación (tech,vend,clus,net). Varias filas pueden compartir rid (distinto 'valores').
    # Calculamos arrays por 'rid' y luego elegimos por fila según su 'rid'.

    # Map de fila->rid
    # Unimos rows_page con keys_df para conocer el rid de cada fila
    if not df_small.empty:
        rows_page = rows_page.merge(
            keys_df,
            on=["technology","vendor","noc_cluster","network"],
            how="left",
            validate="many_to_one"
        )
        rid_per_row = rows_page["rid"].to_numpy()
    else:
        rows_page["rid"] = -1
        rid_per_row = rows_page["rid"].to_numpy()

    # Prepara dict de arrays por métrica
    metric_arrays = {}  # metric -> (arr_y[RID,24], arr_t[RID,24])
    if not df_small.empty:
        for metric in metrics_needed:
            if metric not in df_small.columns:
                # métrica ausente -> arrays NaN
                metric_arrays[metric] = (np.full((len(keys_df), 24), np.nan), np.full((len(keys_df), 24), np.nan))
                continue

            # Para AYER
            sub_y = df_small.loc[(df_small["fecha"].astype(str) == yday) & df_small[metric].notna(), ["rid","h",metric]]
            arr_y = np.full((len(keys_df), 24), np.nan, dtype=float)
            if not sub_y.empty:
                r = sub_y["rid"].to_numpy()
                h = sub_y["h"].to_numpy()
                v = sub_y[metric].astype(float).to_numpy()
                # Asignación vectorizada
                arr_y[r, h] = v

            # Para HOY
            sub_t = df_small.loc[(df_small["fecha"].astype(str) == today) & df_small[metric].notna(), ["rid","h",metric]]
            arr_t = np.full((len(keys_df), 24), np.nan, dtype=float)
            if not sub_t.empty:
                r = sub_t["rid"].to_numpy()
                h = sub_t["h"].to_numpy()
                v = sub_t[metric].astype(float).to_numpy()
                arr_t[r, h] = v

            metric_arrays[metric] = (arr_y, arr_t)
    else:
        # Sin datos: todo NaN
        for metric in metrics_needed:
            metric_arrays[metric] = (np.full((0, 24), np.nan), np.full((0, 24), np.nan))

    # === Construye z de la página (solo filas visibles) ===
    x_dt = [f"{yday}T{h:02d}:00:00" for h in range(24)] + [f"{today}T{h:02d}:00:00" for h in range(24)]
    x_labels = [f"Ay {h:02d}" for h in range(24)] + [f"Hoy {h:02d}" for h in range(24)]
    z_pct, z_unit, y_labels, row_detail = [], [], [], []

    for i, r in rows_page.iterrows():
        valores = r["valores"]
        pm, um = VALORES_MAP.get(valores, (None, None))
        rid = int(r.get("rid", -1))

        # Etiqueta corta + detalle completo
        y_labels.append(f"r{i+1:03d}")
        row_detail.append(f'{r["technology"]}/{r["vendor"]}/{r["noc_cluster"]}/{r["network"]}/{valores}')

        # % (48)
        if pm and pm in metric_arrays and rid >= 0:
            arr_y, arr_t = metric_arrays[pm]
            row48 = np.concatenate([arr_y[rid], arr_t[rid]]).tolist()
        else:
            row48 = [None]*48
        z_pct.append(row48)

        # UNIT (48)
        if um and um in metric_arrays and rid >= 0:
            arr_y, arr_t = metric_arrays[um]
            row48u = np.concatenate([arr_y[rid], arr_t[rid]]).tolist()
        else:
            row48u = [None]*48
        z_unit.append(row48u)

    # % fijo
    pct_payload = {
        "z": z_pct, "x": x_labels, "x_dt": x_dt, "y": y_labels,
        "zmin": 0, "zmax": 100, "title": "% IA / % DC",
        "row_detail": row_detail,
    }
    # UNIT dinámico (página)
    flat_unit = [v for row in z_unit for v in row if isinstance(v,(int,float))]
    umin = (min(flat_unit) if flat_unit else 0)
    umax = (max(flat_unit) if flat_unit else 1)
    if umin == umax:
        umax = (umin or 1)
    unit_payload = {
        "z": z_unit, "x": x_labels, "x_dt": x_dt, "y": y_labels,
        "zmin": umin, "zmax": umax, "title": "Unidades (fails / abnrel)",
        "row_detail": row_detail,
    }

    page_info = {
        "total_rows": total_rows,
        "offset": start,
        "limit": limit,
        "showing": len(rows_page),
    }
    return pct_payload, unit_payload, page_info


# =========================
# Figura de Heatmap (Plotly) — detalle en hover, eje Y ligero
# =========================
def build_heatmap_figure(
    payload,
    *,
    height=720,
    colorscale="Inferno",
    decimals=2,
):
    import plotly.graph_objs as go
    import numpy as np

    if not payload:
        return go.Figure()

    # Usa x_dt (ISO datetime) si está presente
    x = payload.get("x_dt") or payload["x"]
    z = payload["z"]
    y = payload["y"]
    zmin = payload["zmin"]
    zmax = payload["zmax"]
    title = payload.get("title", "")
    row_detail = payload.get("row_detail") or y  # fallback

    # --- customdata por fila (como ya lo tenías, con Net, Máx, Mín, etc.) ---
    customdata = []
    for i, row in enumerate(z):
        arr = np.array([v if isinstance(v, (int, float)) else np.nan for v in row], dtype=float)

        if np.isfinite(arr).any():
            rmax = np.nanmax(arr); rmin = np.nanmin(arr)
            valid_idx = np.where(np.isfinite(arr))[0]
            last_idx = int(valid_idx[-1])
            # Si usamos x_dt, podemos obtener directamente la etiqueta desde x[last_idx]
            last_label = (x[last_idx] if isinstance(x[last_idx], str) else str(x[last_idx]))
            # normaliza a 'YYYY-MM-DD HH:MM'
            last_label = last_label.replace("T", " ")[:16]
        else:
            rmax = np.nan; rmin = np.nan; last_label = "—"

        det = row_detail[i] if i < len(row_detail) else str(y[i])
        parts = det.split("/", 4)  # [tech, vendor, cluster, net, valor]
        tech    = parts[0] if len(parts) > 0 else ""
        vendor  = parts[1] if len(parts) > 1 else ""
        cluster = parts[2] if len(parts) > 2 else ""
        net     = parts[3] if len(parts) > 3 else ""
        valor   = parts[4] if len(parts) > 4 else ""

        def _fmt(v):
            if not np.isfinite(v): return ""
            return f"{v:,.{decimals}f}" if decimals > 0 else f"{v:,.0f}"

        rmax_s = _fmt(rmax)
        rmin_s = _fmt(rmin)

        # [0] tech [1] vendor [2] cluster [3] net [4] valor [5] last_label [6] max [7] min
        row_cd = [[tech, vendor, cluster, net, valor, last_label, rmax_s, rmin_s] for _ in x]
        customdata.append(row_cd)

    # Formato del valor de la celda (sin notación científica)
    z_fmt = f",.{decimals}f" if decimals > 0 else ",.0f"

    # Hover: x como fecha/hora con formato
    hover_tmpl = (
        "DETALLE<br>"
        "<b>Tech:</b> %{customdata[0]}<br>"
        "<b>Vendor:</b> %{customdata[1]}<br>"
        "<b>Cluster:</b> %{customdata[2]}<br>"
        "<b>Net:</b> %{customdata[3]}<br>"
        "<b>Valor:</b> %{customdata[4]}<br>"
        "<b>Última hora con registro:</b> %{customdata[5]}<br>"
        "<b>Máx:</b> %{customdata[6]}<br>"
        "<b>Mín:</b> %{customdata[7]}<br>"
        "<b>%{x|%Y-%m-%d %H:%M}:</b> %{z:" + z_fmt + "}<extra></extra>"
    )

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(title=title),
        customdata=customdata,
        hovertemplate=hover_tmpl,
        hoverongaps=False,
        xgap=0.2, ygap=0.2,
    ))

    # Eje X como fecha: cada 3 horas, rotado, con formato compacto
    # dtick en ms: 3 horas = 3 * 3600 * 1000
    THREE_H_MS = 3 * 3600 * 1000
    fig.update_xaxes(
        type="date",
        dtick=THREE_H_MS,
        tickformat="%b %d %H:%M",   # Ej. "Sep 15 03:00"
        tickangle=-45,
        ticklabelmode="instant",
        ticks="outside",
        ticklen=5,
    )

    # Línea vertical en el corte entre días (columna 24 = hoy 00:00)
    if isinstance(x, (list, tuple)) and len(x) >= 25:
        try:
            boundary_x = x[24]  # hoy 00:00
            fig.add_vline(x=boundary_x, line_dash="dot", line_color="rgba(255,255,255,0.5)", line_width=1)
        except Exception:
            pass

    fig.update_layout(
        height=height,
        margin=dict(l=70, r=16, t=10, b=60),
        xaxis_title="Fecha y hora",
        # 👉 fondos transparentes para que herede el color del contenedor (Card)
        paper_bgcolor="rgba(0,0,0,0)",

        # 👉 tipografías claras
        font=dict(color="#eaeaea"),
        xaxis=dict(
            title="Fecha y hora",
            tickangle=-45,
            ticklabelmode="instant",
            ticks="outside",
            ticklen=5,
            titlefont=dict(color="#eaeaea"),
            tickfont=dict(color="#eaeaea"),
            fixedrange=True,
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            titlefont=dict(color="#eaeaea"),
            tickfont=dict(color="#eaeaea"),
            fixedrange=True,
        ),
        # 👉 hover estilo dark
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff")),
        uirevision="keep",
    )
    return fig



