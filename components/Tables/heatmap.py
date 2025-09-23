import pandas as pd
from datetime import datetime, timedelta
import numpy as np

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
    import numpy as np

    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    if networks is None or not networks:
        networks = _infer_networks(df_ts if df_ts is not None else df_meta)
    if not networks:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    if today is None:
        today = _max_date_str(df_ts["fecha"]) if (df_ts is not None and "fecha" in df_ts.columns) else _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    metrics_needed = {m for v in valores_order for m in VALORES_MAP.get(v, (None, None)) if m}
    if not metrics_needed:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    meta_cols = ["technology", "vendor", "noc_cluster"]
    base = df_meta.drop_duplicates(subset=meta_cols)[meta_cols].reset_index(drop=True)
    rows_full = base.assign(_tmp=1).merge(pd.DataFrame({"network": networks, "_tmp": 1}), on="_tmp").drop(columns="_tmp")

    rows_all_list = []
    for v in valores_order:
        rf = rows_full.copy()
        rf["valores"] = v
        if alarm_only and alarm_keys is not None:
            keys_ok = set(alarm_keys)
            mask = list(zip(rf["technology"], rf["vendor"], rf["noc_cluster"], rf["network"]))
            rf = rf[[m in keys_ok for m in mask]]
        rows_all_list.append(rf)
    rows_all = pd.concat(rows_all_list, ignore_index=True)

    # ---- OPTIMIZADO: calcular Max UNIT en un pass ----
    if df_ts is not None and not df_ts.empty:
        um_cols = [um for _, um in VALORES_MAP.values() if um and um in df_ts.columns]
        if um_cols:
            df_long = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today]),
                                ["technology","vendor","noc_cluster","network"]+um_cols]
            df_long = df_long.melt(id_vars=["technology","vendor","noc_cluster","network"],
                                   value_vars=um_cols,
                                   var_name="metric", value_name="value")
            UM_TO_VAL = {um: name for name, (_, um) in VALORES_MAP.items() if um}
            df_long["valores"] = df_long["metric"].map(UM_TO_VAL)
            df_maxu = (df_long.dropna(subset=["valores"])
                               .groupby(["technology","vendor","noc_cluster","network","valores"], as_index=False)["value"]
                               .max()
                               .rename(columns={"value":"max_unit"}))
            rows_all = rows_all.merge(df_maxu,
                                      on=["technology","vendor","noc_cluster","network","valores"],
                                      how="left")
        else:
            rows_all["max_unit"] = np.nan
    else:
        rows_all["max_unit"] = np.nan

    rows_all["__ord_max_unit__"] = rows_all["max_unit"].astype(float).fillna(float("-inf"))
    rows_all = rows_all.sort_values("__ord_max_unit__", ascending=False, kind="stable")

    # --- paginado
    total_rows = len(rows_all)
    start = max(0, int(offset)); end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # --- df_small reducido
    if df_ts is None or df_ts.empty:
        df_small = pd.DataFrame()
        keys_df = rows_page[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)
        keys_df["rid"] = np.arange(len(keys_df))
    else:
        keys_df = rows_page[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)
        keys_df["rid"] = np.arange(len(keys_df))
        df_small = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today])].merge(keys_df, on=["technology","vendor","noc_cluster","network"])
        hh = df_small["hora"].astype(str).str.split(":", n=1, expand=True)[0]
        df_small["h"] = pd.to_numeric(hh, errors="coerce").where(lambda s: (s>=0)&(s<=23))
        df_small["offset48"] = df_small["h"] + np.where(df_small["fecha"].astype(str)==today, 24, 0)
        df_small = df_small.dropna(subset=["offset48"])
        df_small["offset48"] = df_small["offset48"].astype(int)

    # --- diccionarios (rid, offset48) → valor
    metric_maps = {}
    if not df_small.empty:
        for m in metrics_needed:
            if m in df_small.columns:
                sub = df_small[["rid","offset48",m]].dropna()
                metric_maps[m] = dict(zip(zip(sub["rid"], sub["offset48"]), sub[m]))
            else:
                metric_maps[m] = {}
    else:
        metric_maps = {m:{} for m in metrics_needed}

    def _row48_raw(metric, rid):
        mp = metric_maps.get(metric)
        if not mp: return [None]*48
        return [mp.get((rid, off)) for off in range(48)]

    # --- matrices y stats
    x_dt = [f"{yday}T{h:02d}:00:00" for h in range(24)] + [f"{today}T{h:02d}:00:00" for h in range(24)]
    z_pct, z_unit, z_pct_raw, z_unit_raw = [], [], [], []
    y_labels, row_detail = [], []
    row_last_ts, row_max_pct, row_max_unit = [], [], []

    for rid, r in enumerate(rows_page.itertuples(index=False)):
        tech, vend, clus, net, valores = r.technology, r.vendor, r.noc_cluster, r.network, r.valores
        pm, um = VALORES_MAP.get(valores, (None,None))

        y_labels.append(f"{clus} | {tech}/{vend}/{valores}")
        row_detail.append(f"{tech}/{vend}/{clus}/{net}/{valores}")

        row_raw = _row48_raw(pm, rid) if pm else [None]*48
        row_raw_u = _row48_raw(um, rid) if um else [None]*48

        if pm:
            orient, thr = _sev_cfg(pm, net, UMBRAL_CFG)
            row_color = [_sev_bucket(v, orient, thr) if v is not None else None for v in row_raw]
            z_pct.append(row_color); z_pct_raw.append(row_raw)
        else:
            z_pct.append([None]*48); z_pct_raw.append(row_raw)

        if um:
            mn, mx = _prog_cfg(um, net, UMBRAL_CFG)
            row_norm = [_normalize(v, mn, mx) if v is not None else None for v in row_raw_u]
            z_unit.append(row_norm); z_unit_raw.append(row_raw_u)
        else:
            z_unit.append([None]*48); z_unit_raw.append(row_raw_u)

        arr_u = np.array([v if isinstance(v,(int,float)) else np.nan for v in row_raw_u], float)
        arr_p = np.array([v if isinstance(v,(int,float)) else np.nan for v in row_raw], float)
        if np.isfinite(arr_u).any():
            rmax_u = np.nanmax(arr_u)
            valid_idx = np.where(np.isfinite(arr_u))[0]
        else:
            rmax_u = np.nan
            valid_idx = np.where(np.isfinite(arr_p))[0]
        rmax_p = np.nanmax(arr_p) if np.isfinite(arr_p).any() else np.nan
        if valid_idx.size:
            last_label = str(x_dt[int(valid_idx[-1])]).replace("T"," ")[:16]
        else:
            last_label = ""
        row_last_ts.append(last_label)
        row_max_pct.append(rmax_p); row_max_unit.append(rmax_u)

    pct_payload = {"z":z_pct,"z_raw":z_pct_raw,"x_dt":x_dt,"y":y_labels,"color_mode":"severity",
                   "zmin":-0.5,"zmax":3.5,"title":"% IA / % DC","row_detail":row_detail,
                   "row_last_ts":row_last_ts,"row_max_pct":row_max_pct,"row_max_unit":row_max_unit}
    unit_payload = {"z":z_unit,"z_raw":z_unit_raw,"x_dt":x_dt,"y":y_labels,"color_mode":"progress",
                    "zmin":0.0,"zmax":1.0,"title":"Unidades","row_detail":row_detail,
                    "row_last_ts":row_last_ts,"row_max_pct":row_max_pct,"row_max_unit":row_max_unit}

    page_info = {"total_rows": total_rows, "offset": start, "limit": limit, "showing": len(rows_page)}
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
        tickfont=dict(size=10),
        ticklabelmode="instant",
        ticks="outside",
        ticklen=5,
        fixedrange=True,
        automargin=True,
    )
    fig.update_yaxes(
        showticklabels=True,
        automargin=True,
        fixedrange=True,
        categoryorder="array",
        categoryarray=y,
        autorange="reversed",
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
        margin=dict(l=200, r=16, t=10, b=140), # más espacio abajo e izquierda
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff", size=13)),
        xaxis=dict(
            type="date",
            tickangle=-45,
            ticklabelmode="instant",
            ticks="outside",
            ticklen=5,
            fixedrange=True,
        ),
        yaxis=dict(
            title="",
            showticklabels=True,
            automargin=True,
            tickfont=dict(size=11, color="#eaeaea"),
            categoryorder="array",
            categoryarray=y,
            fixedrange=True,
        ),
        uirevision="keep",
    )

    return fig

def build_heatmap_table_df(pct_payload, unit_payload, *, pct_decimals=2, unit_decimals=0) -> pd.DataFrame:
    src = pct_payload or unit_payload
    if not src:
        return pd.DataFrame(columns=["Cluster","Tech","Vendor","Valor","Última hora","Max %","Max UNIT"])
    y = src.get("y") or []
    detail = src.get("row_detail") or y
    row_last_ts  = (unit_payload or pct_payload).get("row_last_ts") or []
    row_max_pct  = (pct_payload or {}).get("row_max_pct") or []
    row_max_unit = (unit_payload or {}).get("row_max_unit") or []
    rows=[]
    for i in range(len(y)):
        parts=(detail[i] if i<len(detail) else str(y[i])).split("/",4)
        tech=parts[0] if len(parts)>0 else ""
        vendor=parts[1] if len(parts)>1 else ""
        cluster=parts[2] if len(parts)>2 else ""
        valor=parts[4] if len(parts)>4 else ""
        def _fmt(v,dec): return f"{float(v):,.{dec}f}" if v is not None and np.isfinite(v) else ""
        rows.append({"Cluster":cluster,"Tech":tech,"Vendor":vendor,"Valor":valor,
                     "Última hora":row_last_ts[i] if i<len(row_last_ts) else "",
                     "Max %":_fmt(row_max_pct[i] if i<len(row_max_pct) else None,pct_decimals),
                     "Max UNIT":_fmt(row_max_unit[i] if i<len(row_max_unit) else None,unit_decimals)})
    return pd.DataFrame(rows,columns=["Cluster","Tech","Vendor","Valor","Última hora","Max %","Max UNIT"])



