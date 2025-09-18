import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
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

def _interp_nan(v):
    """Interpola NaN/None en 1D; si todo es NaN devuelve ceros."""
    arr = np.array([
        np.nan if (vv is None or not isinstance(vv, (int, float, np.floating))) else float(vv)
        for vv in (v or [])
    ], dtype=float)
    n = arr.size
    if n == 0:
        return arr
    mask = np.isfinite(arr)
    if not mask.any():
        return np.zeros_like(arr)
    x = np.arange(n, dtype=float)
    arr[~mask] = np.interp(x[~mask], x[mask], arr[mask])
    return arr

def _smooth_1d(y, win=3):
    """Media móvil simple."""
    y = np.asarray(y, dtype=float)
    win = max(1, int(win))
    if win == 1 or y.size == 0:
        return y
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")

def _bucket_for_value(v, orient, thr):
    """Devuelve etiqueta de bucket para un valor dado."""
    v = float(v)
    if orient == "higher_is_better":
        if v >= thr["excelente"]: return "excelente"
        elif v >= thr["bueno"]:   return "bueno"
        elif v >= thr["regular"]: return "regular"
        else:                     return "critico"
    else:
        if v <= thr["excelente"]: return "excelente"
        elif v <= thr["bueno"]:   return "bueno"
        elif v <= thr["regular"]: return "regular"
        else:                     return "critico"

def _hex_to_rgba(hex_color, alpha):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"
# =========================
# Payloads de histograma (48 columnas: Ayer 0–23 | Hoy 24–47) con paginado
# =========================
def build_histo_payloads_fast(
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
    Construye payloads de heatmap (% y UNIT) paginados, agrupando por cluster y
    manteniendo el orden por 'alarmados' (vía alarm_keys o el orden de df_meta).
    - %: clasifica 0..3 (excelente→crítico) usando UMBRAL_CFG (severity).
    - UNIT: normaliza 0..1 con min/max por (métrica, red) usando UMBRAL_CFG (progress).
    Devuelve: (pct_payload, unit_payload, page_info).
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

    # --- Construir todas las filas (antes de ordenar/paginar) ---
    rows_full = (
        base.assign(_tmp=1)
        .merge(pd.DataFrame({"network": networks, "_tmp": 1}), on="_tmp", how="left")
        .drop(columns=["_tmp"])
    )
    rows_full = rows_full.assign(
        key5=rows_full[["technology","vendor","noc_cluster","network"]].astype(str).agg("/".join, axis=1)
    )

    rows_all_list = []
    for v in valores_order:
        rf = rows_full.copy()
        rf["valores"] = v
        if alarm_only and alarm_keys is not None:
            keys_ok = set(alarm_keys)
            mask = list(zip(rf["technology"], rf["vendor"], rf["noc_cluster"], rf["network"]))
            rf = rf[[m in keys_ok for m in mask]]
        rows_all_list.append(rf)

    if not rows_all_list:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    rows_all = pd.concat(rows_all_list, ignore_index=True)

    # --- Rank de alarmados + orden por cluster ---
    if alarm_keys:
        order_df = pd.DataFrame(list(alarm_keys), columns=["technology","vendor","noc_cluster","network"])
        order_df = order_df.drop_duplicates().reset_index(drop=True)
    else:
        order_cols = [c for c in ["technology","vendor","noc_cluster","network"] if c in df_meta.columns]
        if order_cols:
            order_df = df_meta[order_cols].drop_duplicates().reset_index(drop=True)
        else:
            order_df = rows_all[["technology","vendor","noc_cluster","network"]].drop_duplicates().reset_index(drop=True)

    order_df = order_df.assign(alarm_rank=lambda d: np.arange(len(d), dtype=int))
    val_rank = {name: i for i, name in enumerate(valores_order)}
    merge_cols = [c for c in ["technology","vendor","noc_cluster","network"] if c in rows_all.columns and c in order_df.columns]

    rows_all = rows_all.merge(order_df, on=merge_cols, how="left")
    rows_all["alarm_rank"] = rows_all["alarm_rank"].fillna(10**9).astype(int)
    rows_all["val_order"]  = rows_all["valores"].map(val_rank).astype(int)

    # Orden final: cluster → alarm_rank → valor → tech/vendor/net
    rows_all = rows_all.sort_values(
        by=["noc_cluster", "alarm_rank", "val_order", "technology", "vendor", "network"],
        kind="stable"
    ).reset_index(drop=True)



    # --- paginado
    total_rows = len(rows_all)
    start = max(0, int(offset)); end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # --- reducir df_ts a visibles
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

    # --- map fila -> rid
    rows_page = rows_page.merge(
        keys_df,
        on=["technology","vendor","noc_cluster","network"],
        how="left",
        validate="many_to_one"
    )

    # --- construir matrices
    x_dt = [f"{yday}T{h:02d}:00:00" for h in range(24)] + [f"{today}T{h:02d}:00:00" for h in range(24)]
    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []
    y_labels, row_detail = [], []

    for _, r in rows_page.iterrows():
        tech = r["technology"]; vend = r["vendor"]; clus = r["noc_cluster"]; net = r["network"]; valores = r["valores"]
        pm, um = VALORES_MAP.get(valores, (None, None))
        rid = int(r.get("rid", -1))

        # Etiqueta Y (ponemos cluster adelante para reforzar el grupo)
        y_labels.append(f"{clus} | {tech}/{vend}/{valores}")
        row_detail.append(f"{tech}/{vend}/{clus}/{net}/{valores}")

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

        # %: clasifica 0..3; guarda crudos para hover/máx/mín
        if pm:
            row_raw = _row48_raw(pm)
            orient, thr = _sev_cfg(pm, net, UMBRAL_CFG)
            row_color = [_sev_bucket(v, orient, thr) if v is not None else None for v in row_raw]
            z_pct.append(row_color); z_pct_raw.append(row_raw)
        else:
            z_pct.append([None]*48); z_pct_raw.append([None]*48)

        # UNIT: normaliza 0..1 por (um, net); guarda crudos
        if um:
            row_raw_u = _row48_raw(um)
            mn, mx = _prog_cfg(um, net, UMBRAL_CFG)
            row_norm = [_normalize(v, mn, mx) if v is not None else None for v in row_raw_u]
            z_unit.append(row_norm); z_unit_raw.append(row_raw_u)
        else:
            z_unit.append([None]*48); z_unit_raw.append([None]*48)

    # --- payloads
    pct_payload = {
        "z": z_pct, "z_raw": z_pct_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "severity",
        "zmin": -0.5, "zmax": 3.5,
        "title": "% IA / % DC (color por umbral)",
        "row_detail": row_detail,
    }
    unit_payload = {
        "z": z_unit, "z_raw": z_unit_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "progress",
        "zmin": 0.0, "zmax": 1.0,
        "title": "Unidades (intensidad por Fail/Abnrel)",
        "row_detail": row_detail,
    }

    page_info = {"total_rows": total_rows, "offset": start, "limit": limit, "showing": len(rows_page)}
    return pct_payload, unit_payload, page_info




# =========================
# Figura de Histograma (Plotly) — detalle en hover, eje Y ligero
# =========================


def build_overlay_waves_figure(
    payload,
    *,
    UMBRAL_CFG: dict,
    mode="severity",       # "severity" = usa % y umbrales; "progress" = usa UNIT
    height=460,            # altura total de la figura (compacta)
    smooth_win=3,          # suavizado
    opacity=0.28,          # opacidad del relleno
    line_width=1.25,       # grosor de línea
    decimals=2
):
    """
    Ondas superpuestas: todas parten de baseline=0.
    - X = payload['x_dt'] (fechas/horas)
    - Y = amplitud normalizada (sin eje Y)
    - Hover con detalle (fila + valor crudo + metadatos)
    - %: color por bucket del pico (SEV_COLORS)
    - UNIT: onda azul; SOLO el pico en rojo si es 'critico' (si no, pico azul)
    """
    if not payload:
        return go.Figure()

    x = payload.get("x_dt") or payload.get("x") or []
    y_labels = payload.get("y") or []
    detail   = payload.get("row_detail") or y_labels
    z_raw    = payload.get("z_raw") or payload.get("z") or []
    n = len(y_labels)
    if n == 0:
        return go.Figure()

    fig = go.Figure()
    val_fmt = f",.{decimals}f" if decimals > 0 else ",.0f"

    for i in range(n):
        row_vals = z_raw[i] if i < len(z_raw) else []
        raw = _interp_nan(row_vals)  # valores crudos por hora (48)

        # Parse detail para net/valores/labels
        parts   = (detail[i] if i < len(detail) else "").split("/", 4)
        tech    = parts[0] if len(parts) > 0 else ""
        vendor  = parts[1] if len(parts) > 1 else ""
        cluster = parts[2] if len(parts) > 2 else ""
        net     = parts[3] if len(parts) > 3 else ""
        valores = parts[4] if len(parts) > 4 else ""

        if mode == "severity":
            # Normaliza por min-max de la fila para formar onda (visual)
            rmin = float(np.nanmin(raw)) if raw.size else 0.0
            rmax = float(np.nanmax(raw)) if raw.size else 1.0
            if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin == rmax:
                amp = np.zeros_like(raw)
            else:
                amp = (raw - rmin) / (rmax - rmin)
            amp = _smooth_1d(np.clip(amp, 0, 1), win=smooth_win)

            # Color por bucket del pico
            pm, _um = VALORES_MAP.get(valores, (None, None))
            orient, thr = _sev_cfg(pm or "", net, UMBRAL_CFG)
            pk = int(np.nanargmax(amp)) if amp.size else 0
            bucket = _bucket_for_value(raw[pk] if raw.size else 0.0, orient, thr)
            color_hex = SEV_COLORS.get(bucket, "#999")
            line_color = color_hex
            fill_color = _hex_to_rgba(color_hex, opacity)
            legend_name = y_labels[i]

            # --- Onda (baseline=0) ---
            fig.add_trace(go.Scatter(
                x=x,
                y=amp,
                mode="lines",
                line=dict(width=line_width, color=line_color),
                fill="tozeroy",
                fillcolor=fill_color,
                name=legend_name,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    f"Valor: %{{customdata[1]:{val_fmt}}}"
                    "<br><span style='opacity:0.85'>Bucket:</span> %{customdata[7]}"
                    "<br><span style='opacity:0.85'>Tech:</span> %{customdata[2]} | "
                    "<span style='opacity:0.85'>Vendor:</span> %{customdata[3]} | "
                    "<span style='opacity:0.85'>Cluster:</span> %{customdata[4]} | "
                    "<span style='opacity:0.85'>Net:</span> %{customdata[5]} | "
                    "<span style='opacity:0.85'>Valor:</span> %{customdata[6]}<extra></extra>"
                ),
                customdata=np.column_stack([
                    np.full(len(x), y_labels[i], dtype=object),  # 0
                    raw if raw.size else np.zeros(len(x)),       # 1
                    np.full(len(x), tech, dtype=object),         # 2
                    np.full(len(x), vendor, dtype=object),       # 3
                    np.full(len(x), cluster, dtype=object),      # 4
                    np.full(len(x), net, dtype=object),          # 5
                    np.full(len(x), valores, dtype=object),      # 6
                    np.full(len(x), bucket, dtype=object),       # 7
                ])
            ))

            # (Opcional) marcador del pico con mismo color de la onda
            fig.add_trace(go.Scatter(
                x=[x[pk] if len(x) else None],
                y=[amp[pk] if amp.size else None],
                mode="markers",
                marker=dict(size=7, color=line_color, symbol="diamond"),
                hovertemplate=(
                    "<b>Pico</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    f"Valor: %{{customdata:{val_fmt}}}<extra></extra>"
                ),
                customdata=[raw[pk] if raw.size else np.nan],
                showlegend=False
            ))

        else:  # progress (UNIT)
            # Normaliza con min/max de umbrales UNIT (como en tu payload UNIT)
            _pm, um = VALORES_MAP.get(valores, (None, None))
            vmin, vmax = _prog_cfg(um or "", net, UMBRAL_CFG)
            norm = np.array([_normalize(v, vmin, vmax) if np.isfinite(v) else 0.0 for v in raw], dtype=float)
            amp  = _smooth_1d(np.clip(norm, 0, 1), win=smooth_win)

            # Pico y bucket SOLO para decidir color del marker (onda SIEMPRE azul)
            orient, thr = _sev_cfg(um or "", net, UMBRAL_CFG)
            pk = int(np.nanargmax(amp)) if amp.size else 0
            bucket = _bucket_for_value(raw[pk] if raw.size else 0.0, orient, thr)

            blue_line = "#0d6efd"
            blue_fill = "rgba(13,110,253,0.25)"
            red_hex   = SEV_COLORS.get("critico", "#e74c3c")

            # --- Onda azul (siempre) ---
            fig.add_trace(go.Scatter(
                x=x,
                y=amp,
                mode="lines",
                line=dict(width= line_width, color=blue_line),
                fill="tozeroy",
                fillcolor=blue_fill,
                name=y_labels[i],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    f"Unidad: %{{customdata[1]:{val_fmt}}}"
                    "<br><span style='opacity:0.85'>Tech:</span> %{customdata[2]} | "
                    "<span style='opacity:0.85'>Vendor:</span> %{customdata[3]} | "
                    "<span style='opacity:0.85'>Cluster:</span> %{customdata[4]} | "
                    "<span style='opacity:0.85'>Net:</span> %{customdata[5]} | "
                    "<span style='opacity:0.85'>Valor:</span> %{customdata[6]}<extra></extra>"
                ),
                customdata=np.column_stack([
                    np.full(len(x), y_labels[i], dtype=object),  # 0
                    raw if raw.size else np.zeros(len(x)),       # 1
                    np.full(len(x), tech, dtype=object),         # 2
                    np.full(len(x), vendor, dtype=object),       # 3
                    np.full(len(x), cluster, dtype=object),      # 4
                    np.full(len(x), net, dtype=object),          # 5
                    np.full(len(x), valores, dtype=object),      # 6
                ])
            ))

            # --- Marker del pico: rojo si crítico; si no, azul ---
            peak_color = (red_hex if bucket == "critico" else blue_line)
            fig.add_trace(go.Scatter(
                x=[x[pk] if len(x) else None],
                y=[amp[pk] if amp.size else None],
                mode="markers",
                marker=dict(size=7, color=peak_color, symbol="diamond"),
                hovertemplate=(
                    "<b>Pico</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    f"Unidad: %{{customdata:{val_fmt}}}"
                    + ( "<br><span style='opacity:0.85'>Crítico</span>" if bucket == "critico" else "" ) +
                    "<extra></extra>"
                ),
                customdata=[raw[pk] if raw.size else np.nan],
                showlegend=False
            ))

    # Ejes
    fig.update_xaxes(
        type="date",
        dtick=3*3600*1000,     # cada 3 horas
        tickformat="%b %d %H:%M",
        tickangle=-45,
        ticks="outside",
        ticklen=5,
        fixedrange=True
    )
    fig.update_yaxes(visible=False, fixedrange=True)

    # Línea vertical entre días (entre idx 23–24) si aplica
    if isinstance(x, (list, tuple)) and len(x) >= 25:
        try:
            fig.add_vline(x=x[24], line_dash="dot", line_color="rgba(255,255,255,0.45)", line_width=1)
        except Exception:
            pass

    # Layout compacto (para que las dos gráficas queden más juntas)
    fig.update_layout(
        height=height,
        margin=dict(l=14, r=14, t=4, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
        uirevision="keep"
    )
    return fig

def build_histo_table_df(pct_payload, unit_payload, *, pct_decimals=2, unit_decimals=0) -> pd.DataFrame:
    """
    Construye una tabla con las filas visibles (página actual) del heatmap.
    Columnas: Cluster, Tech, Vendor, Valor, Max %, Min %, Max UNIT, Min UNIT
    Se alinea 1:1 con el orden del eje Y (y_labels) del heatmap.
    """
    # Selecciona la fuente de "detalle" y etiquetas Y (orden de filas)
    src = pct_payload or unit_payload
    if not src:
        return pd.DataFrame(columns=["Cluster","Tech","Vendor","Valor","Max %","Max UNIT"])

    y = src.get("y") or []
    detail = src.get("row_detail") or y  # "tech/vendor/cluster/net/valores"
    n = len(y)

    # Prepara helpers para obtener min/máx por fila
    def _minmax_from(payload, i, decimals):
        if not payload:
            return "", ""
        z_raw = payload.get("z_raw")
        if not z_raw or i >= len(z_raw):
            return "", ""
        arr = [v for v in (z_raw[i] or []) if isinstance(v, (int, float, np.floating))]
        if not arr:
            return "", ""
        return (f"{max(arr):,.{decimals}f}", f"{min(arr):,.{decimals}f}")

    rows = []
    for i in range(n):
        parts = (detail[i] if i < len(detail) else str(y[i])).split("/", 4)
        tech   = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        cluster= parts[2] if len(parts) > 2 else ""
        # net   = parts[3] if len(parts) > 3 else ""   # si luego quieres añadirlo a la tabla
        valor  = parts[4] if len(parts) > 4 else ""

        max_pct, min_pct   = _minmax_from(pct_payload,  i, pct_decimals)
        max_unit, min_unit = _minmax_from(unit_payload, i, unit_decimals)

        rows.append({
            "Cluster": cluster,
            "Tech": tech,
            "Vendor": vendor,
            "Valor": valor,
            "Max %": max_pct,
            "Max UNIT": max_unit,
        })

    df = pd.DataFrame(rows, columns=["Cluster","Tech","Vendor","Valor","Max %","Max UNIT"])
    return df


