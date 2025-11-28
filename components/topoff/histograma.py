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
    if not metrics_needed:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # base meta
    base = df_meta.drop_duplicates(subset=META_COLS_TOPOFF)[META_COLS_TOPOFF].reset_index(drop=True)

    # expand por métricas
    rows_all_list = []
    for v in valores_order:
        rf = base.copy()
        rf["valores"] = v

        if alarm_only and alarm_keys is not None:
            keys_ok = set(alarm_keys)
            mask = list(zip(
                rf["technology"], rf["vendor"], rf["region"], rf["province"],
                rf["municipality"], rf["site_att"], rf["rnc"], rf["nodeb"]
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

    # keys visibles
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

    def _row192_raw(metric, rid):
        mp = metric_maps.get(metric) or {}
        return [mp.get((rid, off)) for off in range(192)]

    # amarrar rid real por fila expandida
    rows_page = rows_page.merge(
        keys_df, on=META_COLS_TOPOFF, how="left", validate="many_to_one"
    )

    # ejes 15m (192 pts)
    x_dt = _build_x_dt_15m(yday) + _build_x_dt_15m(today)

    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []

    y_labels, row_detail = [], []

    for r in rows_page.itertuples(index=False):
        rid = int(getattr(r, "rid"))

        tech = r.technology
        vend = r.vendor
        region = r.region
        province = r.province
        municipality = r.municipality
        site_att = r.site_att
        rnc = r.rnc
        nodeb = r.nodeb
        valores = r.valores

        pm, um = VALORES_MAP_TOPOFF.get(valores, (None, None))

        y_labels.append(str(nodeb))

        row_detail.append(
            f"{tech}/{vend}/{region}/{province}/{municipality}/{site_att}/{rnc}/{nodeb}/{valores}"
        )

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
    }
    unit_payload = {
        "z": z_unit, "z_raw": z_unit_raw, "x_dt": x_dt, "y": y_labels,
        "color_mode": "progress",
        "zmin": 0.0, "zmax": 1.0,
        "title": "Unidades (TopOff)",
        "row_detail": row_detail,
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
    opacity: float = 0.28,
    line_width: float = 1.2,
    decimals: int = 2,
    title_text: str | None = None,
    show_yaxis_ticks: bool = True,
    selected_wave: str | None = None,  # series_key = payload["row_detail"][i]
    selected_x: str | None = None,
    show_peak: bool = True,
):
    """
    Igual a build_overlay_waves_figure del main,
    pero parsea detail TopOff: tech/vendor/region/prov/mun/site/rnc/nodeb/valores
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

    if isinstance(selected_x, str):
        selected_x = selected_x.replace(" ", "T")[:19]

    # ---------- rango Y global ----------
    y_min = np.inf
    y_max = -np.inf

    for i in range(n):
        row_vals = z_raw[i] if i < len(z_raw) else []
        raw = _interp_nan(row_vals)
        if raw.size and np.isfinite(raw).any():
            y_min = min(y_min, float(np.nanmin(raw)))
            y_max = max(y_max, float(np.nanmax(raw)))

    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    pad = (y_max - y_min) * 0.05
    ylo, yhi = y_min - pad, y_max + pad

    # ---------- trazado ----------
    fig = go.Figure()
    val_fmt = f",.{decimals}f" if decimals > 0 else ",.0f"

    is_any_selected = bool(selected_wave)
    seen_legend = set()

    for i in range(n):
        row_vals = z_raw[i] if i < len(z_raw) else []
        raw = _interp_nan(row_vals)

        # parse TopOff detail
        parts = (detail[i] if i < len(detail) else "").split("/", 8)
        tech   = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        region = parts[2] if len(parts) > 2 else ""
        prov   = parts[3] if len(parts) > 3 else ""
        mun    = parts[4] if len(parts) > 4 else ""
        site   = parts[5] if len(parts) > 5 else ""
        rnc    = parts[6] if len(parts) > 6 else ""
        nodeb  = parts[7] if len(parts) > 7 else ""
        valores= parts[8] if len(parts) > 8 else ""

        series_key = detail[i]

        raw_plot = _smooth_1d(raw, win=smooth_win) if (smooth_win and smooth_win > 1 and raw.size) else raw

        # pico
        if raw.size and np.isfinite(raw).any():
            pk = int(np.nanargmax(raw))
            val_pk = raw[pk]
        else:
            pk = 0
            val_pk = 0.0

        # color base por bucket
        if mode == "severity":
            pm, _um = VALORES_MAP_TOPOFF.get(valores, (None, None))
            orient, thr = _sev_cfg(pm or "", None, UMBRAL_CFG)
            bucket = _bucket_for_value(val_pk if np.isfinite(val_pk) else 0.0, orient, thr)
            base_hex = SEV_COLORS.get(bucket, "#999")
        else:
            base_hex = "#0d6efd"
            bucket = ""

        is_sel = bool(selected_wave and series_key == selected_wave)

        if is_any_selected:
            line_w        = (line_width * 2.6) if is_sel else (line_width * 0.9)
            fill_opacity  = (opacity * 1.25) if is_sel else (opacity * 0.12)
            overall_alpha = 1.0 if is_sel else 0.25
            line_color    = base_hex if is_sel else "rgba(160,160,160,0.55)"
            fill_color    = (_hex_to_rgba(base_hex, fill_opacity) if mode=="severity"
                             else ("rgba(13,110,253,0.35)" if is_sel else "rgba(160,160,160,0.08)"))
        else:
            line_w        = line_width * 1.25
            overall_alpha = 1.0
            line_color    = base_hex
            fill_color    = (_hex_to_rgba(base_hex, opacity) if mode=="severity"
                             else "rgba(13,110,253,0.25)")

        # leyenda una sola por nodeb
        showlegend = nodeb not in seen_legend
        seen_legend.add(nodeb)

        # halo si seleccionado
        if is_sel:
            fig.add_trace(go.Scatter(
                x=x, y=raw_plot, mode="lines",
                line=dict(width=line_w*1.8, color="rgba(255,255,255,0.35)"),
                hoverinfo="skip", showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=x, y=raw_plot,
            mode="lines",
            line=dict(width=line_w, color=line_color),
            fill="tozeroy",
            fillcolor=fill_color,
            name=str(nodeb),
            legendgroup=str(nodeb),
            showlegend=showlegend,
            opacity=overall_alpha,
            customdata=np.column_stack([
                np.full(len(x), series_key, dtype=object),
                raw if raw.size else np.zeros(len(x)),
                np.full(len(x), tech, dtype=object),
                np.full(len(x), vendor, dtype=object),
                np.full(len(x), region, dtype=object),
                np.full(len(x), prov, dtype=object),
                np.full(len(x), mun, dtype=object),
                np.full(len(x), site, dtype=object),
                np.full(len(x), rnc, dtype=object),
                np.full(len(x), nodeb, dtype=object),
                np.full(len(x), valores, dtype=object),
                np.full(len(x), bucket, dtype=object),
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{x|%Y-%m-%d %H:%M}<br>"
                + ("Valor: " if mode=="severity" else "Unidad: ")
                + f"%{{customdata[1]:{val_fmt}}}"
                + "<br><span style='opacity:0.85'>Bucket:</span> %{customdata[11]}"
                + "<br><span style='opacity:0.85'>Tech:</span> %{customdata[2]} | "
                  "<span style='opacity:0.85'>Vendor:</span> %{customdata[3]}"
                + "<br><span style='opacity:0.85'>Region:</span> %{customdata[4]} | "
                  "<span style='opacity:0.85'>Prov:</span> %{customdata[5]} | "
                  "<span style='opacity:0.85'>Mun:</span> %{customdata[6]}"
                + "<br><span style='opacity:0.85'>Site:</span> %{customdata[7]} | "
                  "<span style='opacity:0.85'>RNC:</span> %{customdata[8]} | "
                  "<span style='opacity:0.85'>NodeB:</span> %{customdata[9]}"
                + "<br><span style='opacity:0.85'>Valor:</span> %{customdata[10]}"
                + "<extra></extra>"
            )
        ))

        # marcador pico
        if show_peak and raw.size and np.isfinite(val_pk) and pk < len(x):
            fig.add_trace(go.Scatter(
                x=[x[pk]], y=[raw_plot[pk] if pk < len(raw_plot) else None],
                mode="markers",
                marker=dict(size=10, color=base_hex, symbol="x"),
                showlegend=False, hoverinfo="skip",
                opacity=1.0 if is_sel or not is_any_selected else 0.35
            ))

    # ejes y partición ayer/hoy
    ONE_HOUR_MS = 3600 * 1000
    fig.update_xaxes(
        type="date",
        dtick=ONE_HOUR_MS,  # 👈 ticks por hora aunque haya 15m abajo
        tickformat="%b %d %H:%M",
        tickangle=-45,
        ticks="outside",
        ticklen=5,
        fixedrange=True,
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

    fig.update_layout(
        title=title_text or None,
        height=height,
        margin=dict(l=14, r=14, t=30 if title_text else 8, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
        uirevision="keep",
    )
    return fig
