# components/Heatmaps/topoff_histograma.py
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict

from components.topoff.heatmap import (
    META_COLS_TOPOFF, VALORES_MAP_TOPOFF
)
from components.topoff.heatmap import (
    _max_date_str, _day_str, _sev_cfg, _prog_cfg, _normalize, SEV_COLORS
)

# Reutilizamos helpers del histograma main
from components.main.histograma import (
    _interp_nan, _smooth_1d, _bucket_for_value, _hex_to_rgba
)

METRIC_COLOR_MAP_TOPOFF = {
    "PS_RRC":  "#0d6efd",  # azul
    "CS_RRC":  "#0d6efd",
    "PS_RAB":  "#dc3545",  # rojo
    "CS_RAB":  "#dc3545",
    "PS_S1":   "#ffc107",  # amarillo
    "PS_DROP": "#6f42c1",  # morado
    "CS_DROP": "#6f42c1",
}
# =========================================================
# PAYLOADS HISTO TOPOFF (AYER/Hoy, 48 columnas)
# =========================================================

def build_histo_payloads_topoff(
    df_meta: pd.DataFrame,
    df_ts: pd.DataFrame,
    *,
    UMBRAL_CFG: dict,
    valores_order=("PS_RRC","CS_RRC","PS_DROP","CS_DROP","PS_RAB","CS_RAB"),
    today: Optional[str] = None,
    yday: Optional[str] = None,
    alarm_keys: Optional[set] = None,
    alarm_only: bool = False,
    offset: int = 0,
    limit: int = 20,
    domain: str = "PS",  # 👈 NUEVO
) -> Tuple[Optional[dict], Optional[dict], dict]:
    """
    TopOff histo con muestras cada 15 min:
      - waves a 15m (ayer 96 + hoy 96 = 192)
      - eje X solo tickea por hora (se controla en la figura)
    """

    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # fechas
    if today is None:
        if df_ts is not None and not df_ts.empty and "fecha" in df_ts.columns:
            today = _max_date_str(df_ts["fecha"]) or _day_str(datetime.now())
        else:
            today = _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # métricas necesarias
    metrics_needed = set()
    for v in valores_order:
        pm, um = VALORES_MAP_TOPOFF.get(v, (None, None))
        if pm: metrics_needed.add(pm)
        if um: metrics_needed.add(um)

    domain = (domain or "PS").upper()
    traffic_metric = None
    if df_ts is not None and not df_ts.empty:
        if domain == "PS" and "ps_traff_gb" in df_ts.columns:
            traffic_metric = "ps_traff_gb"
            metrics_needed.add("ps_traff_gb")
        elif domain == "CS" and "cs_traff_erl" in df_ts.columns:
            traffic_metric = "cs_traff_erl"
            metrics_needed.add("cs_traff_erl")

    # base meta (ahora incluye cluster vía META_COLS_TOPOFF)
    base = df_meta.drop_duplicates(subset=META_COLS_TOPOFF)[META_COLS_TOPOFF].reset_index(drop=True)

    # expand por métricas
    rows_all_list = []
    for v in valores_order:
        rf = base.copy()
        rf["valores"] = v

        if alarm_only and alarm_keys is not None:
            keys_ok = set(alarm_keys)
            # key completa con cluster
            mask = list(zip(
                rf["technology"], rf["vendor"], rf["region"], rf["province"],
                rf["municipality"], rf["cluster"], rf["site_att"], rf["rnc"], rf["nodeb"]
            ))
            rf = rf[[m in keys_ok for m in mask]]

        rows_all_list.append(rf)

    if not rows_all_list:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    rows_all = pd.concat(rows_all_list, ignore_index=True)

    # ---- Max UNIT en ayer/hoy (para ordenar filas) ----
    if df_ts is not None and not df_ts.empty:
        um_cols = [um for _, um in VALORES_MAP_TOPOFF.values() if um and um in df_ts.columns]
        if um_cols:
            df_long = df_ts.loc[
                df_ts["fecha"].astype(str).isin([yday, today]),
                META_COLS_TOPOFF + um_cols
            ].melt(
                id_vars=META_COLS_TOPOFF,
                value_vars=um_cols,
                var_name="metric", value_name="value"
            )

            UM_TO_VAL = {um: name for name, (_, um) in VALORES_MAP_TOPOFF.items() if um}
            df_long["valores"] = df_long["metric"].map(UM_TO_VAL)

            df_maxu = (
                df_long.dropna(subset=["valores"])
                .groupby(META_COLS_TOPOFF + ["valores"], as_index=False)["value"]
                .max()
                .rename(columns={"value": "max_unit"})
            )
            rows_all = rows_all.merge(
                df_maxu, on=META_COLS_TOPOFF + ["valores"], how="left"
            )
        else:
            rows_all["max_unit"] = np.nan
    else:
        rows_all["max_unit"] = np.nan

    # ordenar igual que heatmap topoff
    rows_all["__ord_max_unit__"] = rows_all["max_unit"].astype(float).fillna(float("-inf"))
    rows_all = rows_all.sort_values("__ord_max_unit__", ascending=False, kind="stable").reset_index(drop=True)

    # paginado
    total_rows = len(rows_all)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # keys visibles (incluye cluster)
    keys_df = rows_page[META_COLS_TOPOFF].drop_duplicates().reset_index(drop=True)
    keys_df["rid"] = np.arange(len(keys_df), dtype=int)

    # ---------- helpers 15 min ----------
    def _safe_q15_to_idx(hhmmss):
        """'10:15:00' -> 10*4 + 1 = 41. Devuelve 0..95 o None."""
        try:
            s = str(hhmmss)
            parts = s.split(":")
            hh = int(parts[0])
            mm = int(parts[1]) if len(parts) > 1 else 0
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                q = mm // 15
                return hh * 4 + q
        except Exception:
            pass
        return None

    def _build_x_dt_15m(day_str):
        return [
            f"{day_str}T{h:02d}:{m:02d}:00"
            for h in range(24)
            for m in (0, 15, 30, 45)
        ]

    # TS reducido a visibles
    if df_ts is None or df_ts.empty:
        df_small = pd.DataFrame()
    else:
        df_small = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today])].merge(
            keys_df, on=META_COLS_TOPOFF, how="inner", validate="many_to_one"
        )

        df_small["q15"] = df_small["hora"].apply(_safe_q15_to_idx)
        df_small["offset192"] = df_small["q15"] + np.where(
            df_small["fecha"].astype(str) == today, 96, 0
        )
        df_small = df_small.dropna(subset=["offset192"])
        df_small["offset192"] = df_small["offset192"].astype(int)

    # maps (rid, offset192)->value
    metric_maps: Dict[str, dict] = {}
    if not df_small.empty:
        for m in metrics_needed:
            if m in df_small.columns:
                sub = df_small[["rid", "offset192", m]].dropna().sort_values("offset192")
                metric_maps[m] = dict(zip(zip(sub["rid"], sub["offset192"]), sub[m]))
            else:
                metric_maps[m] = {}
    else:
        metric_maps = {m: {} for m in metrics_needed}

    def _row192_raw(metric, rid, zero_after_last: bool = True):
        mp = metric_maps.get(metric) or {}
        row = [mp.get((rid, off)) for off in range(192)]

        if not zero_after_last:
            return row

        # Índice del PRIMER valor real
        first_idx = -1
        last_idx = -1
        for i, v in enumerate(row):
            if v is not None:
                if first_idx == -1:
                    first_idx = i
                last_idx = i

        # Si no hubo ningún valor real, deja que _interp_nan lo maneje
        if first_idx == -1:
            return row

        # 👉 Antes del primer valor, pon 0.0
        if first_idx > 0:
            for j in range(0, first_idx):
                row[j] = 0.0

        # 👉 Después del último valor, pon 0.0 (como ya hacías)
        if last_idx < len(row) - 1:
            for j in range(last_idx + 1, len(row)):
                row[j] = 0.0

        return row

    # amarrar rid real por fila expandida
    rows_page = rows_page.merge(
        keys_df, on=META_COLS_TOPOFF, how="left", validate="many_to_one"
    )

    # ejes 15m (192 pts)
    x_dt = _build_x_dt_15m(yday) + _build_x_dt_15m(today)

    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []

    y_labels, row_detail = [], []
    traffic_rows_raw = []
    traffic_by_key = {}

    for r in rows_page.itertuples(index=False):
        rid = int(getattr(r, "rid"))

        tech = r.technology
        vend = r.vendor
        region = r.region
        province = r.province
        municipality = r.municipality
        cluster = getattr(r, "cluster", "") or ""
        site_att = r.site_att
        rnc = r.rnc
        nodeb = r.nodeb
        valores = r.valores

        pm, um = VALORES_MAP_TOPOFF.get(valores, (None, None))

        y_labels.append(str(nodeb))

        # detail ahora: tech/vendor/region/prov/mun/cluster/site/rnc/nodeb/valores
        row_detail.append(
            f"{tech}/{vend}/{region}/{province}/{municipality}/{cluster}/{site_att}/{rnc}/{nodeb}/{valores}"
        )
        series_key = row_detail[-1]
        # --- Tráfico raw por serie ---
        if traffic_metric:
            row_traffic = _row192_raw(traffic_metric, rid, zero_after_last=True)
        else:
            row_traffic = [None] * 192

        traffic_rows_raw.append(row_traffic)
        traffic_by_key[series_key] = row_traffic
        # % raw + buckets
        if pm:
            row_raw = _row192_raw(pm, rid)
            orient, thr = _sev_cfg(pm, None, UMBRAL_CFG)
            row_color = [
                (0 if v is None else
                 0 if v <= thr["excelente"] else
                 1 if v <= thr["bueno"] else
                 2 if v <= thr["regular"] else
                 3)
                for v in row_raw
            ]
            z_pct.append(row_color)
            z_pct_raw.append(row_raw)
        else:
            z_pct.append([None]*192)
            z_pct_raw.append([None]*192)

        # UNIT raw + normalizado
        if um:
            row_raw_u = _row192_raw(um, rid)
            mn, mx = _prog_cfg(um, None, UMBRAL_CFG)
            row_norm = [_normalize(v, mn, mx) if v is not None else None for v in row_raw_u]
            z_unit.append(row_norm)
            z_unit_raw.append(row_raw_u)
        else:
            z_unit.append([None]*192)
            z_unit_raw.append([None]*192)

    pct_payload = {
        "z": z_pct, "z_raw": z_pct_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "severity",
        "zmin": -0.5, "zmax": 3.5,
        "title": "% IA / % DC (TopOff)",
        "row_detail": row_detail,
        # 👇 NUEVO
        "traffic_raw": traffic_rows_raw,
        "traffic_by_key": traffic_by_key,
        "traffic_metric": traffic_metric,
        "domain": domain,
    }
    unit_payload = {
        "z": z_unit, "z_raw": z_unit_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "progress",
        "zmin": 0.0, "zmax": 1.0,
        "title": "Unidades (TopOff)",
        "row_detail": row_detail,
        # 👇 opcional pero útil que también lo lleve
        "traffic_raw": traffic_rows_raw,
        "traffic_by_key": traffic_by_key,
        "traffic_metric": traffic_metric,
        "domain": domain,
    }

    page_info = {
        "total_rows": total_rows,
        "offset": start,
        "limit": limit,
        "showing": len(rows_page),
    }
    return pct_payload, unit_payload, page_info

# =========================================================
# FIGURA HISTO TOPOFF (overlay waves)
# =========================================================

def build_overlay_waves_figure_topoff(
    payload,
    *,
    UMBRAL_CFG: dict,
    mode: str = "severity",    # "severity" (% crudos) | "progress" (units crudos)
    height: int = 420,
    smooth_win: int = 3,
    opacity: float = 0.9,      # ya no hay fill, solo líneas
    line_width: float = 1.2,
    decimals: int = 2,
    title_text: str | None = None,
    show_yaxis_ticks: bool = True,
    selected_wave: str | None = None,
    selected_x: str | None = None,

    # NUEVO: tráfico de fondo
    show_traffic_bars: bool = False,
    traffic_agg: str = "mean",     # "mean" | "sum"
    traffic_decimals: int = 1,
):
    """
    Overlay TopOff:
      - waves crudas en 15m (192 puntos)
      - sin fill (solo línea)
      - colores fijos por valores (RRC, RAB, S1, DROP)
      - barras grises de tráfico en y2 si show_traffic_bars=True
    """

    if not payload:
        return go.Figure()

    x        = payload.get("x_dt") or payload.get("x") or []
    y_labels = payload.get("y") or []
    detail   = payload.get("row_detail") or y_labels
    z_raw    = payload.get("z_raw") or payload.get("z") or []
    n = len(y_labels)
    if n == 0:
        return go.Figure()

    domain = (payload.get("domain") or "PS").upper()
    traffic_metric = payload.get("traffic_metric")
    traffic_raw = payload.get("traffic_raw") or []
    traffic_by_key = payload.get("traffic_by_key") or {}

    if isinstance(selected_x, str):
        selected_x = selected_x.replace(" ", "T")[:19]

    # ---------- 1) Rango Y global ----------
    sev_min = np.inf
    sev_max = -np.inf
    unit_data_min = np.inf
    unit_data_max = -np.inf
    unit_cfg_min  = np.inf
    unit_cfg_max  = -np.inf

    for i in range(n):
        row_vals = z_raw[i] if i < len(z_raw) else []
        raw = _interp_nan(row_vals)
        parts   = (detail[i] if i < len(detail) else "").split("/", 9)
        tech    = parts[0] if len(parts) > 0 else ""
        vendor  = parts[1] if len(parts) > 1 else ""
        region  = parts[2] if len(parts) > 2 else ""
        prov    = parts[3] if len(parts) > 3 else ""
        mun     = parts[4] if len(parts) > 4 else ""
        cluster = parts[5] if len(parts) > 5 else ""
        site    = parts[6] if len(parts) > 6 else ""
        rnc     = parts[7] if len(parts) > 7 else ""
        nodeb   = parts[8] if len(parts) > 8 else ""
        valores = parts[9] if len(parts) > 9 else ""

        if raw.size and np.isfinite(raw).any():
            vmin = float(np.nanmin(raw))
            vmax = float(np.nanmax(raw))
            if mode == "severity":
                sev_min = min(sev_min, vmin)
                sev_max = max(sev_max, vmax)
            else:
                unit_data_min = min(unit_data_min, vmin)
                unit_data_max = max(unit_data_max, vmax)
                _pm, um = VALORES_MAP_TOPOFF.get(valores, (None, None))
                if um:
                    mn, mx = _prog_cfg(um, None, UMBRAL_CFG)
                    unit_cfg_min = min(unit_cfg_min, float(mn))
                    unit_cfg_max = max(unit_cfg_max, float(mx))

    def _pad(lo, hi, frac=0.05):
        if not np.isfinite(lo) or not np.isfinite(hi):
            return 0.0, 1.0
        if hi <= lo:
            return lo, lo + 1.0
        span = hi - lo
        return lo - span*frac, hi + span*frac

    if mode == "severity":
        ylo, yhi = _pad(
            sev_min if np.isfinite(sev_min) else 0.0,
            sev_max if np.isfinite(sev_max) else 1.0,
        )
    else:
        lo_cand = min(
            c for c in [unit_data_min, unit_cfg_min] if np.isfinite(c)
        ) if (np.isfinite(unit_data_min) or np.isfinite(unit_cfg_min)) else 0.0
        hi_cand = max(
            c for c in [unit_data_max, unit_cfg_max] if np.isfinite(c)
        ) if (np.isfinite(unit_data_max) or np.isfinite(unit_cfg_max)) else 1.0
        ylo, yhi = _pad(lo_cand, hi_cand)

    # ---------- 2) Figure base ----------
    fig = go.Figure()
    val_fmt = f",.{decimals}f" if decimals > 0 else ",.0f"

    seen_legend = set()
    is_any_selected = bool(selected_wave)

    # ---------- 2.A) Barras de tráfico en Y2 ----------
    if show_traffic_bars and x and (traffic_raw or traffic_by_key):
        traffic_series = None

        # Si hay selección -> solo esa serie
        if selected_wave and selected_wave in traffic_by_key:
            traffic_series = traffic_by_key[selected_wave]

        # Si no, agregamos todas las visibles
        elif traffic_raw:
            traffic_series = []
            for j in range(len(x)):
                vals = []
                for row in traffic_raw:
                    if row and j < len(row):
                        v = row[j]
                        if isinstance(v, (int,float,np.floating)) and np.isfinite(v):
                            vals.append(float(v))
                if not vals:
                    traffic_series.append(None)
                else:
                    traffic_series.append(
                        sum(vals) if traffic_agg == "sum" else (sum(vals)/len(vals))
                    )

        # asegurar longitud consistente
        if traffic_series and len(traffic_series) != len(x):
            traffic_series = (
                traffic_series[:len(x)]
                + [None] * max(0, len(x) - len(traffic_series))
            )

        if traffic_series:
            hovertemplate_traf = (
                "%{x|%Y-%m-%d %H:%M}<br>"
                f"Tráfico: %{{y:,.{traffic_decimals}f}}<extra></extra>"
            )

            fig.add_trace(go.Bar(
                x=x,
                y=traffic_series,
                yaxis="y2",
                marker=dict(color="rgba(180,180,180,0.18)"),
                showlegend=False,
                hovertemplate=hovertemplate_traf,
            ))

    # ---------- 2.B) Waves ----------
    for i in range(n):
        row_vals = z_raw[i] if i < len(z_raw) else []
        raw = _interp_nan(row_vals)

        parts = (detail[i] if i < len(detail) else "").split("/", 9)
        tech    = parts[0] if len(parts) > 0 else ""
        vendor  = parts[1] if len(parts) > 1 else ""
        region  = parts[2] if len(parts) > 2 else ""
        prov    = parts[3] if len(parts) > 3 else ""
        mun     = parts[4] if len(parts) > 4 else ""
        cluster = parts[5] if len(parts) > 5 else ""
        site    = parts[6] if len(parts) > 6 else ""
        rnc     = parts[7] if len(parts) > 7 else ""
        nodeb   = parts[8] if len(parts) > 8 else ""
        valores = parts[9] if len(parts) > 9 else ""

        series_key = detail[i]

        raw_plot = _smooth_1d(raw, win=smooth_win) if (smooth_win and smooth_win > 1 and raw.size) else raw

        # pico (si lo quieres usar luego)
        if raw.size and np.isfinite(raw).any():
            pk = int(np.nanargmax(raw))
            val_pk = float(raw[pk])
        else:
            pk = 0
            val_pk = 0.0

        # Color base por "valores"
        base_hex = METRIC_COLOR_MAP_TOPOFF.get(valores, "#999999")

        # Selección
        is_sel = bool(selected_wave and series_key == selected_wave)

        if is_any_selected:
            line_w        = (line_width * 2.4) if is_sel else (line_width * 0.9)
            overall_alpha = 1.0 if is_sel else 0.22
            line_color    = base_hex if is_sel else "rgba(150,150,150,0.55)"
        else:
            line_w        = line_width * 1.3
            overall_alpha = 1.0
            line_color    = base_hex

        # Leyenda: una por NodeB · Cluster
        display_name = f"{tech} · {vendor} · {cluster} · {site} · {valores}"
        legend_key = f"{tech}__{vendor}__{cluster}__{site}__{valores}"

        showlegend = legend_key not in seen_legend
        seen_legend.add(legend_key)

        # Marcadores para selected_x
        sel_sizes = [0]*len(x)
        sel_colors = [base_hex]*len(x)
        mode_line = "lines"
        if selected_x and selected_x in x:
            j = x.index(selected_x)
            if 0 <= j < len(sel_sizes):
                sel_sizes[j]  = 9
                sel_colors[j] = "#ffffff"
            mode_line = "lines+markers"

        # Hover limpio, en cascada y sin región/prov/mun ni concatenado largo
        hovertemplate = (
            "%{x|%Y-%m-%d %H:%M}<br>"
            + ("Valor: " if mode=="severity" else "Unidad: ")
            + f"%{{customdata[1]:{val_fmt}}}"
            + "<br><span style='opacity:0.85'>Tech:</span> %{customdata[2]}"
            + "<br><span style='opacity:0.85'>Vendor:</span> %{customdata[3]}"
            + "<br><span style='opacity:0.85'>Cluster:</span> %{customdata[7]}"
            + "<br><span style='opacity:0.85'>Site:</span> %{customdata[8]}"
            + "<br><span style='opacity:0.85'>RNC:</span> %{customdata[9]}"
            + "<br><span style='opacity:0.85'>NodeB:</span> %{customdata[10]}"
            + "<br><span style='opacity:0.85'>Valor:</span> %{customdata[11]}"
            + "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=x,
            y=raw_plot,
            mode=mode_line,
            line=dict(width=line_w, color=line_color),
            marker=dict(
                size=sel_sizes,
                color=sel_colors,
                symbol="diamond"
            ),
            name=display_name,
            legendgroup=legend_key,
            showlegend=showlegend,
            opacity=overall_alpha,
            fill=None,  # sin área, solo línea
            customdata=np.column_stack([
                np.full(len(x), series_key, dtype=object),   # 0 (ya no lo usamos en hover)
                raw if raw.size else np.zeros(len(x)),       # 1
                np.full(len(x), tech, dtype=object),         # 2
                np.full(len(x), vendor, dtype=object),       # 3
                np.full(len(x), region, dtype=object),       # 4
                np.full(len(x), prov, dtype=object),         # 5
                np.full(len(x), mun, dtype=object),          # 6
                np.full(len(x), cluster, dtype=object),      # 7
                np.full(len(x), site, dtype=object),         # 8
                np.full(len(x), rnc, dtype=object),          # 9
                np.full(len(x), nodeb, dtype=object),        # 10
                np.full(len(x), valores, dtype=object),      # 11
            ]),
            hovertemplate=hovertemplate,
        ))

    # ---------- 3) Ejes ----------
    ONE_HOUR_MS = 3600 * 1000
    fig.update_xaxes(
        type="date",
        dtick=ONE_HOUR_MS,
        tickformat="%b %d %H:%M",
        tickangle=-45,
        ticks="outside",
        ticklen=5,
        fixedrange=True,
        showgrid=False,
    )
    fig.update_yaxes(
        visible=show_yaxis_ticks,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255,255,255,0.12)",
        zeroline=False,
        tickfont=dict(size=10),
        rangemode="tozero",
        range=[ylo, yhi],
    )

    # Y2 para tráfico
    traffic_title = None
    if show_traffic_bars and traffic_metric:
        if "gb" in traffic_metric.lower():
            traffic_title = "Tráfico (GB)"
        elif "erl" in traffic_metric.lower():
            traffic_title = "Tráfico (Erlangs)"
        else:
            traffic_title = "Tráfico"

    fig.update_layout(
        title=title_text or None,
        height=height,
        margin=dict(l=14, r=40, t=30 if title_text else 8, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(
            bgcolor="#222",
            bordercolor="#444",
            font=dict(color="#fff")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        uirevision="keep",
        yaxis2=dict(
            title=traffic_title,
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10),
        ) if show_traffic_bars and traffic_metric else None,
    )

    # Línea divisoria ayer/hoy (96 puntos = 1 día)
    if isinstance(x, (list, tuple)) and len(x) >= 97:
        try:
            fig.add_vline(
                x=x[96],
                line_dash="dot",
                line_color="rgba(255,255,255,0.45)",
                line_width=1
            )
        except Exception:
            pass

    if selected_x:
        try:
            fig.add_vline(x=selected_x, line_width=2, line_color="rgba(255,255,255,0.85)")
        except Exception:
            pass

    return fig



