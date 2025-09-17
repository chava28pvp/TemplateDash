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
SEV_COLORS = {
    "excelente": "#2ecc71",  # verde
    "bueno":     "#f1c40f",  # amarillo
    "regular":   "#e67e22",  # naranja
    "critico":   "#e74c3c",  # rojo
}
SEV_ORDER = ["excelente", "bueno", "regular", "critico"]
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

def _sev_cfg(metric: str, net: str | None, cfg: dict):
    """Obtiene thresholds y orientación para métricas de % (severity)."""
    s = (cfg.get("severity") or {}).get(metric) or {}
    # soporta tanto {"orientation":..., "thresholds":{...}} como {"default":{...}, "per_network":{...}}
    if "thresholds" in s:
        thresholds = s.get("thresholds") or {}
        orient = s.get("orientation", s.get("default", {}).get("orientation", "lower_is_better"))
    else:
        orient = (s.get("default") or {}).get("orientation", "lower_is_better")
        pern = (s.get("per_network") or {})
        if net and net in pern and "thresholds" in pern[net]:
            thresholds = pern[net]["thresholds"]
        else:
            thresholds = (s.get("default") or {}).get("thresholds") or {}
    # Asegura orden y float
    thr = {k: float(thresholds.get(k)) for k in SEV_ORDER if k in thresholds}
    # rellena por si faltan
    for k in SEV_ORDER:
        thr.setdefault(k, thr.get("regular", 0.0))
    return orient, thr

def _sev_bucket(value: float | None, orient: str, thr: dict) -> int | None:
    """Mapea valor → 0..3 (0 verde .. 3 rojo). None si valor no numérico."""
    if value is None:
        return None
    v = float(value)
    # Solo implementamos lower_is_better (tu JSON usa eso). Para higher_is_better invierte.
    if orient == "higher_is_better":
        # Invertimos los cortes (mejor alto)
        if v >= thr["excelente"]: return 0
        elif v >= thr["bueno"]:   return 1
        elif v >= thr["regular"]: return 2
        else:                     return 3
    else:  # lower_is_better
        if v <= thr["excelente"]: return 0
        elif v <= thr["bueno"]:   return 1
        elif v <= thr["regular"]: return 2
        else:                     return 3

def _prog_cfg(metric: str, net: str | None, cfg: dict):
    """Obtiene min/max para métricas UNIT (progress), con per_network si existe."""
    p = (cfg.get("progress") or {}).get(metric) or {}
    if "default" in p or "per_network" in p:
        d = p.get("default") or {}
        mn = d.get("min", 0.0); mx = d.get("max", 1.0)
        pern = p.get("per_network") or {}
        if net and net in pern:
            mn = pern[net].get("min", mn)
            mx = pern[net].get("max", mx)
        return float(mn), float(mx)
    # forma plana {min,max}
    return float(p.get("min", 0.0)), float(p.get("max", 1.0))

def _normalize(v: float | None, vmin: float, vmax: float) -> float | None:
    if v is None:
        return None
    if vmax <= vmin:
        return 0.0
    x = (float(v) - vmin) / (vmax - vmin)
    return 0.0 if x < 0 else (1.0 if x > 1 else x)
# =========================
# Payloads de heatmap (48 columnas: Ayer 0–23 | Hoy 24–47) con paginado
# =========================
def build_heatmap_payloads_fast(
    df_meta: pd.DataFrame,
    df_ts: pd.DataFrame,
    *,
    UMBRAL_CFG: dict,
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
    Igual que el ULTRA, pero:
      - %: clasifica 0..3 según umbrales (verde→rojo).
      - UNIT: normaliza 0..1 según min/max por (métrica, red).
      - En customdata guarda el valor real por celda y máx/mín por fila.
    Devuelve (pct_payload, unit_payload, page_info).
    """
    import numpy as np
    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # redes
    if networks is None or not networks:
        networks = _infer_networks(df_ts if df_ts is not None else df_meta)
    if not networks:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # fechas
    if today is None:
        today = _max_date_str(df_ts["fecha"]) if (df_ts is not None and "fecha" in df_ts.columns) else _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # métricas necesarias
    metrics_needed = set()
    for v in valores_order:
        pm, um = VALORES_MAP.get(v, (None, None))
        if pm: metrics_needed.add(pm)
        if um: metrics_needed.add(um)
    if not metrics_needed:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # meta base
    meta_cols = ["technology", "vendor", "noc_cluster"]
    base = df_meta.drop_duplicates(subset=meta_cols)[meta_cols].reset_index(drop=True)

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

    # paginado
    start = max(0, int(offset)); end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # reduce df_ts a visibles
    if df_ts is None or df_ts.empty:
        df_small = pd.DataFrame()
        keys_df = rows_page[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)
        keys_df["rid"] = np.arange(len(keys_df), dtype=int)
    else:
        keys_df = rows_page[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)
        keys_df["rid"] = np.arange(len(keys_df), dtype=int)
        df_small = df_ts.loc[
            df_ts["fecha"].astype(str).isin([yday, today]) &
            df_ts["network"].astype(str).isin(keys_df["network"].astype(str))
        ].copy()
        df_small = df_small.merge(
            keys_df,
            on=["technology","vendor","noc_cluster","network"],
            how="inner",
            validate="many_to_one"
        )
        # hora 0..23
        hh = df_small["hora"].astype(str).str.split(":", n=1, expand=True)[0]
        df_small["h"] = pd.to_numeric(hh, errors="coerce").where(lambda s: (s>=0) & (s<=23)).astype("Int64")

        keep_cols = {"fecha","h","rid"} | set(metrics_needed)
        df_small = df_small[[c for c in keep_cols if c in df_small.columns]].dropna(subset=["h"])
        df_small["h"] = df_small["h"].astype(int)

    # map fila -> rid
    rows_page = rows_page.merge(
        keys_df,
        on=["technology","vendor","noc_cluster","network"],
        how="left",
        validate="many_to_one"
    )
    x_dt = [f"{yday}T{h:02d}:00:00" for h in range(24)] + [f"{today}T{h:02d}:00:00" for h in range(24)]
    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []
    y_labels, row_detail = [], []

    for i, r in rows_page.iterrows():
        tech = r["technology"]; vend = r["vendor"]; clus = r["noc_cluster"]; net = r["network"]; valores = r["valores"]
        pm, um = VALORES_MAP.get(valores, (None, None))
        rid = int(r.get("rid", -1))
        y_labels.append(f"r{i+1:03d}")
        row_detail.append(f"{tech}/{vend}/{clus}/{net}/{valores}")

        # Helper para tomar serie (ayer+hoy) del df_small por métrica
        def _row48_raw(metric):
            if metric is None or df_small.empty or rid < 0 or metric not in df_small.columns:
                return [None]*48
            sub_y = df_small.loc[(df_small["rid"]==rid) & (df_small["fecha"].astype(str)==yday), ["h", metric]]
            sub_t = df_small.loc[(df_small["rid"]==rid) & (df_small["fecha"].astype(str)==today), ["h", metric]]
            arr_y = [None]*24; arr_t = [None]*24
            if not sub_y.empty:
                for _, rr in sub_y.iterrows():
                    v = rr[metric]; arr_y[int(rr["h"])] = (float(v) if pd.notna(v) else None)
            if not sub_t.empty:
                for _, rr in sub_t.iterrows():
                    v = rr[metric]; arr_t[int(rr["h"])] = (float(v) if pd.notna(v) else None)
            return arr_y + arr_t

        # %: clasifica 0..3; guarda también crudos para hover/máx/mín
        if pm:
            row_raw = _row48_raw(pm)
            # thresholds por (pm, net)
            orient, thr = _sev_cfg(pm, net, UMBRAL_CFG)
            row_color = [ _sev_bucket(v, orient, thr) if v is not None else None for v in row_raw ]
            z_pct.append(row_color)
            z_pct_raw.append(row_raw)
        else:
            z_pct.append([None]*48); z_pct_raw.append([None]*48)

        # UNIT: normaliza 0..1 por (um, net); guarda crudos
        if um:
            row_raw_u = _row48_raw(um)
            mn, mx = _prog_cfg(um, net, UMBRAL_CFG)
            row_norm = [ _normalize(v, mn, mx) if v is not None else None for v in row_raw_u ]
            z_unit.append(row_norm)
            z_unit_raw.append(row_raw_u)
        else:
            z_unit.append([None]*48); z_unit_raw.append([None]*48)

    # payloads
    pct_payload = {
        "z": z_pct, "z_raw": z_pct_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "severity",    # 👈 importante
        "zmin": -0.5, "zmax": 3.5,   # buckets 0..3
        "title": "% IA / % DC (color por umbral)",
        "row_detail": row_detail,
    }
    # UNIT: rango 0..1
    unit_payload = {
        "z": z_unit, "z_raw": z_unit_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "progress",   # 👈 importante
        "zmin": 0.0, "zmax": 1.0,
        "title": "Unidades (intensidad por Fail/Abnrel)",
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
    decimals=2,
):
    import plotly.graph_objs as go
    import numpy as np

    if not payload:
        return go.Figure()

    z       = payload["z"]                 # z para COLOR (clase 0..3 o 0..1)
    z_raw   = payload.get("z_raw") or z    # valores reales para hover
    x       = payload.get("x_dt") or payload.get("x")  # usamos datetime si existe
    y       = payload["y"]
    zmin    = payload["zmin"]
    zmax    = payload["zmax"]
    title   = payload.get("title", "")
    mode    = payload.get("color_mode", "severity")
    detail  = payload.get("row_detail") or y

    # ----- Colores según modo -----
    if mode == "severity":
        # 0..3 -> verde, amarillo, naranja, rojo
        colorscale = [
            [0/3, SEV_COLORS["excelente"]],
            [1/3, SEV_COLORS["bueno"]],
            [2/3, SEV_COLORS["regular"]],
            [3/3, SEV_COLORS["critico"]],
        ]
        colorbar = dict(
            title=title,
            tickmode="array",
            tickvals=[0,1,2,3],
            ticktext=["Excelente","Bueno","Regular","Crítico"],
        )
    else:  # progress
        # 0..1 -> blanco a azul
        colorscale = [
            [0.0, "#f8f9fa"],
            [1.0, "#0d6efd"],
        ]
        colorbar = dict(title=title)

    # ----- customdata por celda: detalle + máx/mín de la fila + valor crudo -----
    customdata = []
    for i, row in enumerate(z_raw):
        arr = np.array([v if isinstance(v,(int,float)) else np.nan for v in row], dtype=float)
        if np.isfinite(arr).any():
            rmax = np.nanmax(arr); rmin = np.nanmin(arr)
            valid_idx = np.where(np.isfinite(arr))[0]
            last_idx = int(valid_idx[-1])
            last_label = (x[last_idx] if isinstance(x[last_idx], str) else str(x[last_idx]))
            last_label = last_label.replace("T", " ")[:16]
        else:
            rmax = np.nan; rmin = np.nan; last_label = "—"

        parts = (detail[i] if i < len(detail) else str(y[i])).split("/", 4)
        tech   = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        clus   = parts[2] if len(parts) > 2 else ""
        net    = parts[3] if len(parts) > 3 else ""
        valor  = parts[4] if len(parts) > 4 else ""

        def _fmt(v):
            if not np.isfinite(v): return ""
            return f"{v:,.{decimals}f}" if decimals > 0 else f"{v:,.0f}"
        rmax_s = _fmt(rmax); rmin_s = _fmt(rmin)

        # customdata por celda: [0]tech [1]vend [2]clus [3]net [4]valor [5]last [6]max [7]min [8]raw_cell
        row_cd = []
        for j in range(len(x)):
            raw_cell = arr[j] if j < len(arr) else np.nan
            raw_s = _fmt(raw_cell)
            row_cd.append([tech, vendor, clus, net, valor, last_label, rmax_s, rmin_s, raw_s])
        customdata.append(row_cd)

    # d3 format para raw en hover (aunque ya viene formateado, por si decides usar %{z} crudo)
    # z_fmt = f",.{decimals}f" if decimals > 0 else ",.0f"

    hover_tmpl = (
        # Valor destacado (arriba)
        "<span style='font-size:120%; font-weight:700'>%{customdata[8]}</span><br>"
        # Fecha/hora en línea aparte
        "<span style='opacity:0.85'>%{x|%Y-%m-%d %H:%M}</span><br>"
        "──────────<br>"
        "DETALLE<br>"
        "<b>Tech:</b> %{customdata[0]}<br>"
        "<b>Vendor:</b> %{customdata[1]}<br>"
        "<b>Cluster:</b> %{customdata[2]}<br>"
        "<b>Net:</b> %{customdata[3]}<br>"
        "<b>Valor:</b> %{customdata[4]}<br>"
        "<b>Última hora con registro:</b> %{customdata[5]}<br>"
        "<b>Máx:</b> %{customdata[6]}<br>"
        "<b>Mín:</b> %{customdata[7]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        colorbar=colorbar,
        customdata=customdata,
        hovertemplate=hover_tmpl,
        hoverongaps=False,
        xgap=0.2, ygap=0.2,
    ))

    # Eje X como fecha con ticks espaciados (cada 3h)
    THREE_H_MS = 3 * 3600 * 1000
    fig.update_xaxes(
        type="date",
        dtick=THREE_H_MS,
        tickformat="%b %d %H:%M",
        tickangle=-45,
        ticklabelmode="instant",
        ticks="outside",
        ticklen=5,
        fixedrange=True,
    )
    # Línea del corte entre días
    if isinstance(x, (list, tuple)) and len(x) >= 25:
        try:
            fig.add_vline(x=x[24], line_dash="dot", line_color="rgba(255,255,255,0.5)", line_width=1)
        except Exception:
            pass

    # Dark look & feel
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=16, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff", size=13)),
        yaxis=dict(title="", showticklabels=False, fixedrange=True),
        uirevision="keep",
    )

    return fig




