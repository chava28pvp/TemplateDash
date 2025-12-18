import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import numpy as np

from components.main.heatmap import _infer_networks, _max_date_str, VALORES_MAP, _day_str, _sev_cfg, _prog_cfg, \
    _normalize, SEV_COLORS


# =========================
# Config
# =========================
VALOR_COLORS = {
    "RRC":  "#3b82f6",  # azul
    "RAB":  "#ef4444",  # rojo
    "S1":   "#facc15",  # amarillo
    "DROP": "#a855f7",  # morado
}

# =========================
# Helpers
# =========================
def _sev_bucket(value: float | None, orient: str, thr: dict) -> int | None:
    """Mapea valor → 0..3. None si valor no numérico."""
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

def _fill_missing_with_zero(v):
    arr = np.array([
        0.0 if (vv is None or not isinstance(vv, (int, float, np.floating)) or not np.isfinite(float(vv)))
        else float(vv)
        for vv in (v or [])
    ], dtype=float)
    return arr

# =========================
# Payloads de histograma (48 columnas: Ayer 0–23 | Hoy 24–47) con paginado
# =========================
def build_histo_payloads_fast(
    df_meta: pd.DataFrame,
    df_ts: pd.DataFrame,
    *,
    UMBRAL_CFG: dict,
    networks=None,
    valores_order=("PS_RRC","CS_RRC","PS_DROP","CS_DROP","PS_RAB","CS_RAB"),
    today=None,
    yday=None,
    alarm_keys=None,
    alarm_only=False,
    offset=0,
    limit=5,
    traffic_metric: str | None = None,
):
    """
    Construye payloads de heatmap (% y UNIT) paginados, agrupando por cluster y
    manteniendo el orden por 'alarmados' (vía alarm_keys o el orden de df_meta).
    - %: clasifica 0..3 (excelente→crítico) usando UMBRAL_CFG (severity).
    - UNIT: normaliza 0..1 con min/max por (métrica, red) usando UMBRAL_CFG (progress).
    Devuelve: (pct_payload, unit_payload, page_info).
    """

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
        if traffic_metric:metrics_needed.add(traffic_metric)
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
    if df_ts is not None and not df_ts.empty:
        um_cols = [um for _, um in VALORES_MAP.values() if um and um in df_ts.columns]
        if um_cols:
            df_long = df_ts.loc[
                df_ts["fecha"].astype(str).isin([yday, today]),
                ["technology", "vendor", "noc_cluster", "network"] + um_cols
            ].melt(
                id_vars=["technology", "vendor", "noc_cluster", "network"],
                value_vars=um_cols,
                var_name="metric", value_name="value"
            )
            UM_TO_VAL = {um: name for name, (_, um) in VALORES_MAP.items() if um}
            df_long["valores"] = df_long["metric"].map(UM_TO_VAL)

            df_maxu = (
                df_long.dropna(subset=["valores"])
                .groupby(["technology", "vendor", "noc_cluster", "network", "valores"], as_index=False)["value"]
                .max()
                .rename(columns={"value": "max_unit"})
            )
            rows_all = rows_all.merge(
                df_maxu,
                on=["technology", "vendor", "noc_cluster", "network", "valores"],
                how="left"
            )
        else:
            rows_all["max_unit"] = np.nan
    else:
        rows_all["max_unit"] = np.nan

    # Orden idéntico al heatmap: por mayor max_unit
    rows_all["__ord_max_unit__"] = rows_all["max_unit"].astype(float).fillna(float("-inf"))
    rows_all = rows_all.sort_values("__ord_max_unit__", ascending=False, kind="stable").reset_index(drop=True)



    # --- paginado
    total_rows = len(rows_all)
    start = max(0, int(offset)); end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # --- reducir df_ts a visibles
    # --- reducir df_ts a visibles (mismo patrón que el heatmap) ---
    keys_df = rows_page[["technology", "vendor", "noc_cluster", "network"]].drop_duplicates().reset_index(drop=True)
    keys_df["rid"] = np.arange(len(keys_df), dtype=int)

    if df_ts is None or df_ts.empty:
        # sin datos de time-series, dejamos df_small vacío pero con columnas esperadas
        df_small = pd.DataFrame(columns=["fecha", "h", "rid", *metrics_needed])
    else:
        # igual que el heatmap: filtra solo por fecha y luego mergea por las 4 claves
        df_small = df_ts.loc[
            df_ts["fecha"].astype(str).isin([yday, today])
        ].merge(
            keys_df,
            on=["technology", "vendor", "noc_cluster", "network"],
            how="inner",
            validate="many_to_one"
        )

        if df_small.empty or "hora" not in df_small.columns:
            df_small = pd.DataFrame(columns=["fecha", "h", "rid", *metrics_needed])
        else:
            # hora 0..23 (más robusto, sin expand=True)
            hh = df_small["hora"].astype(str).str.split(":", n=1).str[0]
            df_small["h"] = pd.to_numeric(hh, errors="coerce").where(
                lambda s: (s >= 0) & (s <= 23)
            ).astype("Int64")

            keep_cols = {"fecha", "h", "rid"} | set(metrics_needed)
            df_small = (
                df_small[[c for c in keep_cols if c in df_small.columns]]
                .dropna(subset=["h"])
            )
            df_small["h"] = df_small["h"].astype(int)

    # --- map fila -> rid (igual que heatmap) ---
    rows_page = rows_page.merge(
        keys_df,
        on=["technology", "vendor", "noc_cluster", "network"],
        how="left",
        validate="many_to_one"
    )

    # --- construir matrices
    x_dt = [f"{yday}T{h:02d}:00:00" for h in range(24)] + [f"{today}T{h:02d}:00:00" for h in range(24)]
    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []
    y_labels, row_detail = [], []
    traffic_rows_raw = []
    traffic_by_key = {}

    for _, r in rows_page.iterrows():
        tech = r["technology"]; vend = r["vendor"]; clus = r["noc_cluster"]; net = r["network"]; valores = r["valores"]
        pm, um = VALORES_MAP.get(valores, (None, None))
        rid = int(r.get("rid", -1))

        # Etiqueta Y (ponemos cluster adelante para reforzar el grupo)
        y_labels.append(str(clus))
        row_detail.append(f"{tech}/{vend}/{clus}/{net}/{valores}")

        def _row48_raw(metric):
            if metric is None or df_small.empty or rid < 0 or metric not in df_small.columns:
                return [None] * 48

            sub_y = df_small.loc[
                (df_small["rid"] == rid) & (df_small["fecha"].astype(str) == yday),
                ["h", metric]
            ]
            sub_t = df_small.loc[
                (df_small["rid"] == rid) & (df_small["fecha"].astype(str) == today),
                ["h", metric]
            ]
            arr_y = [None] * 24
            arr_t = [None] * 24
            if not sub_y.empty:
                for _, rr in sub_y.iterrows():
                    v = rr[metric]
                    arr_y[int(rr["h"])] = float(v) if pd.notna(v) else None
            if not sub_t.empty:
                for _, rr in sub_t.iterrows():
                    v = rr[metric]
                    arr_t[int(rr["h"])] = float(v) if pd.notna(v) else None
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
        if traffic_metric:
            traffic_rows_raw.append(_row48_raw(traffic_metric))
        else:
            traffic_rows_raw.append([None] * 48)
    # --- payloads
    if traffic_metric:
        for k, tr in zip(row_detail, traffic_rows_raw):
            traffic_by_key[k] = tr
    pct_payload = {
        "z": z_pct, "z_raw": z_pct_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "severity",
        "zmin": -0.5, "zmax": 3.5,
        "title": "% IA / % DC (color por umbral)",
        "row_detail": row_detail,
        "traffic_metric": traffic_metric,
        "traffic_raw": traffic_rows_raw,
        "traffic_by_key": traffic_by_key,
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
    mode: str = "severity",       # "severity" | "progress"
    height: int = 460,
    smooth_win: int = 3,
    opacity: float = 0.28,        # ahora se usa solo como referencia de intensidad visual
    line_width: float = 1.25,
    decimals: int = 2,
    title_text: str | None = None,
    show_yaxis_ticks: bool = True,
    selected_wave: str | None = None,
    selected_x: str | None = None,

    # tráfico
    show_traffic_bars: bool = False,
    traffic_agg: str = "mean",    # "mean" | "sum"
    traffic_decimals: int = 1,
):
    """
    Dibuja waves con valores CRUDOS en Y a partir de `payload` (z_raw, x_dt, y, row_detail).

    CAMBIOS clave:
      - Líneas ONLY (sin fill).
      - Sin marcadores de pico.
      - Color fijo por familia de 'valores' (RRC/RAB/S1/DROP).
      - Barras de tráfico opcionales en y2 (eje derecho).
    """
    if not payload:
        return go.Figure()

    x        = payload.get("x_dt") or payload.get("x") or []
    y_labels = payload.get("y") or []
    detail   = payload.get("row_detail") or y_labels
    z_raw    = payload.get("z_raw") or payload.get("z") or []

    n = len(y_labels)
    if n == 0 or not x:
        return go.Figure()
    # --- Normaliza X a datetime real para evitar offsets visuales ---
    try:
        # si viene como string ISO
        x = pd.to_datetime(x).to_pydatetime().tolist()
    except Exception:
        pass

    HOUR_MS = 3600 * 1000
    BAR_WIDTH = HOUR_MS * 0.92  # ancho ~92% de la hora
    # --------------------------
    # Colores fijos por familia
    # --------------------------
    VALOR_COLORS = {
        "RRC":  "#3b82f6",  # azul
        "RAB":  "#ef4444",  # rojo
        "S1":   "#facc15",  # amarillo
        "DROP": "#a855f7",  # morado
    }

    def _color_for_valores(valores: str) -> str:
        v = (valores or "").upper()
        if "RRC" in v:
            return VALOR_COLORS["RRC"]
        if "RAB" in v:
            return VALOR_COLORS["RAB"]
        if "S1" in v:
            return VALOR_COLORS["S1"]
        if "DROP" in v:
            return VALOR_COLORS["DROP"]
        return "#9aa0a6"

    # Normaliza selected_x a ISO
    if isinstance(selected_x, str):
        selected_x = selected_x.replace(" ", "T")[:19]

    indices = list(range(n))

    # ---------- 1) Rango Y global ----------
    sev_min = np.inf
    sev_max = -np.inf
    unit_data_min = np.inf
    unit_data_max = -np.inf
    unit_cfg_min  = np.inf
    unit_cfg_max  = -np.inf

    for i in indices:
        row_vals = z_raw[i] if i < len(z_raw) else []
        raw = _fill_missing_with_zero(row_vals)

        parts   = (detail[i] if i < len(detail) else "").split("/", 4)
        net     = parts[3] if len(parts) > 3 else ""
        valores = parts[4] if len(parts) > 4 else ""

        if raw.size:
            vmin = np.nanmin(raw)
            vmax = np.nanmax(raw)

            if mode == "severity":
                if np.isfinite(vmin):
                    sev_min = min(sev_min, float(vmin))
                if np.isfinite(vmax):
                    sev_max = max(sev_max, float(vmax))
            else:
                if np.isfinite(vmin):
                    unit_data_min = min(unit_data_min, float(vmin))
                if np.isfinite(vmax):
                    unit_data_max = max(unit_data_max, float(vmax))
                _pm, um = VALORES_MAP.get(valores, (None, None))
                if um:
                    mn, mx = _prog_cfg(um, net, UMBRAL_CFG)
                    unit_cfg_min = min(unit_cfg_min, float(mn))
                    unit_cfg_max = max(unit_cfg_max, float(mx))

    def _pad(lo, hi, frac=0.05):
        if not np.isfinite(lo) or not np.isfinite(hi):
            return 0.0, 1.0
        if hi <= lo:
            return lo, lo + 1.0
        span = hi - lo
        return lo - span * frac, hi + span * frac

    if mode == "severity":
        ylo, yhi = _pad(
            sev_min if np.isfinite(sev_min) else 0.0,
            sev_max if np.isfinite(sev_max) else 1.0
        )
    else:
        lo_cand = (
            min(c for c in [unit_data_min, unit_cfg_min] if np.isfinite(c))
            if (np.isfinite(unit_data_min) or np.isfinite(unit_cfg_min))
            else 0.0
        )
        hi_cand = (
            max(c for c in [unit_data_max, unit_cfg_max] if np.isfinite(c))
            if (np.isfinite(unit_data_max) or np.isfinite(unit_cfg_max))
            else 1.0
        )
        ylo, yhi = _pad(lo_cand, hi_cand)

    # ---------- 2) Figura base ----------
    fig = go.Figure()
    val_fmt = f",.{decimals}f" if decimals > 0 else ",.0f"
    seen_legend = set()
    is_any_selected = bool(selected_wave)

    # ---------- 2.1) Barras de tráfico (opcional en y2) ----------
    # ---------- 2.1) Barras de tráfico (opcional en y2) ----------
    traffic_max = None

    if show_traffic_bars:
        traffic_by_key = payload.get("traffic_by_key") or {}
        traffic_rows = payload.get("traffic_raw") or []

        traffic_series = None

        # 1) si hay selección, usa SOLO esa serie
        if selected_wave and selected_wave in traffic_by_key:
            traffic_series = traffic_by_key[selected_wave]

        # 2) si no hay selección, agrega todas las visibles
        elif traffic_rows:
            traffic_series = []
            for j in range(len(x)):
                vals = []
                for row in traffic_rows:
                    if row and j < len(row):
                        v = row[j]
                        if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                            vals.append(float(v))
                if not vals:
                    traffic_series.append(None)
                else:
                    traffic_series.append(
                        sum(vals) if traffic_agg == "sum" else (sum(vals) / len(vals))
                    )

        # Asegura longitud igual a x
        if traffic_series and len(traffic_series) != len(x):
            traffic_series = (
                    traffic_series[:len(x)]
                    + [None] * max(0, len(x) - len(traffic_series))
            )

        if traffic_series:
            arr_t = np.array(
                [np.nan if v is None else float(v) for v in traffic_series],
                dtype=float
            )
            if np.isfinite(arr_t).any():
                traffic_max = float(np.nanmax(arr_t))

            # Barras
            fig.add_trace(go.Bar(
                x=x,
                y=traffic_series,
                yaxis="y2",
                width=[BAR_WIDTH] * len(x),
                marker=dict(color="rgba(180,180,180,0.14)"),
                showlegend=False,
                hoverinfo="skip",
                offsetgroup="traffic",
                alignmentgroup="traffic",
            ))

            # Puntos en cada barra (hover aquí)
            fig.add_trace(go.Scatter(
                x=x,
                y=traffic_series,
                yaxis="y2",
                mode="markers",
                marker=dict(size=5, color="rgba(220,220,220,0.35)"),
                showlegend=False,
                hovertemplate=(
                    "<b>Tráfico</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    f"%{{y:,.{traffic_decimals}f}}<extra></extra>"
                ),
            ))

    # ---------- 2.2) Waves ----------
    for i in indices:
        row_vals = z_raw[i] if i < len(z_raw) else []
        raw = _fill_missing_with_zero(row_vals)

        parts   = (detail[i] if i < len(detail) else "").split("/", 4)
        tech    = parts[0] if len(parts) > 0 else ""
        vendor  = parts[1] if len(parts) > 1 else ""
        cluster = parts[2] if len(parts) > 2 else ""
        net     = parts[3] if len(parts) > 3 else ""
        valores = parts[4] if len(parts) > 4 else ""

        series_key = detail[i]

        # suavizado visual
        raw_plot = (
            _smooth_1d(raw, win=smooth_win)
            if (smooth_win and smooth_win > 1 and raw.size)
            else raw
        )

        # bucket solo para hover (no para color)
        if raw.size and np.isfinite(raw).any():
            pk = int(np.nanargmax(raw))
            val_pk = raw[pk]
        else:
            val_pk = 0.0

        if mode == "severity":
            pm, _um = VALORES_MAP.get(valores, (None, None))
            orient, thr = _sev_cfg(pm or "", net, UMBRAL_CFG)
        else:
            # para hover, intentamos usar thresholds del % si existe
            pm, um = VALORES_MAP.get(valores, (None, None))
            orient, thr = _sev_cfg(pm or (um or ""), net, UMBRAL_CFG)

        bucket = _bucket_for_value(val_pk if np.isfinite(val_pk) else 0.0, orient, thr)
        base_color_hex = _color_for_valores(valores)

        # selección
        is_sel = bool(selected_wave and series_key == selected_wave)

        if is_any_selected:
            line_w        = (line_width * 2.2) if is_sel else (line_width * 0.9)
            overall_alpha = 1.0 if is_sel else 0.22
            line_color    = base_color_hex if is_sel else "rgba(160,160,160,0.55)"
        else:
            line_w        = line_width * 1.25
            overall_alpha = 1.0
            line_color    = base_color_hex

        # leyenda por cluster
        legend_key = f"{cluster}__{valores}"
        showlegend = legend_key not in seen_legend
        seen_legend.add(legend_key)

        # halo para seleccionada
        if is_sel:
            fig.add_trace(go.Scatter(
                x=x, y=raw_plot,
                mode="lines",
                line=dict(width=line_w * 1.7, color="rgba(255,255,255,0.30)"),
                hoverinfo="skip",
                showlegend=False,
                opacity=1.0
            ))

        # serie principal (LINES ONLY)
        fig.add_trace(go.Scatter(
            x=x, y=raw_plot,
            mode="lines",
            line=dict(width=line_w, color=line_color),
            name=f"{cluster} • {valores}",
            legendgroup=legend_key,
            showlegend=showlegend,
            opacity=overall_alpha,
            customdata=np.column_stack([
                np.full(len(x), series_key, dtype=object),
                raw if raw.size else np.zeros(len(x)),
                np.full(len(x), tech, dtype=object),
                np.full(len(x), vendor, dtype=object),
                np.full(len(x), cluster, dtype=object),
                np.full(len(x), net, dtype=object),
                np.full(len(x), valores, dtype=object),
                np.full(len(x), bucket, dtype=object),
            ]),
            hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    + ("Valor: " if mode == "severity" else "Unidad: ")
                    + f"%{{customdata[1]:{val_fmt}}}"
                      "<br>──────────<br>"
                      "<span style='opacity:0.85'>Tech:</span> %{customdata[2]}<br>"
                      "<span style='opacity:0.85'>Vendor:</span> %{customdata[3]}<br>"
                      "<span style='opacity:0.85'>Cluster:</span> %{customdata[4]}<br>"
                      "<span style='opacity:0.85'>Net:</span> %{customdata[5]}<br>"
                      "<span style='opacity:0.85'>Valor:</span> %{customdata[6]}"
                      "<extra></extra>"
            ),
        ))

    # ---------- 3) Ejes y partición AYER|HOY ----------
    fig.update_xaxes(
        type="date",
        dtick=3 * 3600 * 1000,
        tickformat="%b %d %H:%M",
        tickangle=-45,
        ticks="outside",
        ticklen=5,
        fixedrange=True,
        showgrid=False,
    )

    # y principal
    fig.update_yaxes(
        visible=True if show_yaxis_ticks else False,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255,255,255,0.12)",
        zeroline=False,
        tickfont=dict(size=10),
        rangemode="tozero",
        range=[ylo, yhi],
    )

    # línea separadora entre días
    if isinstance(x, (list, tuple)) and len(x) >= 25:
        try:
            fig.add_vline(
                x=x[24],
                line_dash="dot",
                line_color="rgba(255,255,255,0.45)",
                line_width=1
            )
        except Exception:
            pass

    # selected_x solo vline (sin marcador)
    if selected_x:
        try:
            fig.add_vline(
                x=selected_x,
                line_width=2,
                line_color="rgba(255,255,255,0.85)"
            )
        except Exception:
            pass

    # ---------- 4) Layout + eje derecho ----------
    layout_kwargs = dict(
        title=title_text or None,
        height=height,
        margin=dict(l=14, r=14, t=30 if title_text else 8, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10)
        ),
        uirevision="keep",
        bargap=0.0,
    )

    if show_traffic_bars:
        # y2 overlay
        max_y2 = (traffic_max * 1.08) if (traffic_max and traffic_max > 0) else 1
        layout_kwargs["yaxis2"] = dict(
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10),
            rangemode="tozero",
            range=[0, max_y2],
        )

    fig.update_layout(**layout_kwargs)

    return fig








