# components/Heatmaps/topoff_heatmap.py
import math
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc

from components.main.heatmap import (
    _max_date_str, _day_str, _sev_cfg, _sev_score_continuo, _prog_cfg, _normalize,
    _hm_height, _last_numeric, _only_time, _fmt, _vendor_initial
)

# =========================================================
# CONFIG TOPOFF
# =========================================================

VALORES_MAP_TOPOFF = {
    "PS_RRC":  ("ps_rrc_ia_percent", "ps_rrc_fail"),
    "CS_RRC":  ("cs_rrc_ia_percent", "cs_rrc_fail"),
    "PS_RAB":  ("ps_rab_ia_percent", "ps_rab_fail"),
    "CS_RAB":  ("cs_rab_ia_percent", "cs_rab_fail"),
    "PS_DROP": ("ps_drop_dc_percent", "ps_drop_abnrel"),
    "CS_DROP": ("cs_drop_dc_percent", "cs_drop_abnrel"),
    "PS_S1":   ("ps_s1_ia_percent", "ps_s1_fail"),
    "RTX_TNL": ("rtx_tnl_tx_percent", "tnl_abn"),
}

SEV_COLORS = {
    "excelente": "#2ecc71",
    "bueno":     "#f1c40f",
    "regular":   "#e67e22",
    "critico":   "#e74c3c",
}
SEV_ORDER = ["excelente", "bueno", "regular", "critico"]

# Meta que define UNA FILA única en TopOff (ahora con cluster)
META_COLS_TOPOFF = [
    "technology", "vendor",
    "region", "province", "municipality",
    "cluster",          # 👈 NOC_CLUSTER
    "site_att", "rnc", "nodeb",
]

# Alturas para alinear fila-a-fila (idéntico a main)
ROW_H = 26
MARG_TOP = 0
MARG_BOTTOM = 124
EXTRA = 0

# =========================================================
# Helpers
# =========================================================
def _build_x_dt_15m(day_str):
    # 96 bins por día (24*4)
    return [
        f"{day_str}T{h:02d}:{m:02d}:00"
        for h in range(24)
        for m in (0, 15, 30, 45)
    ]


def _safe_q15_to_idx(hhmmss):
    """
    hhmmss: '10:15:00' -> 10*4 + 1 = 41
    Devuelve 0..95 o None.
    """
    try:
        s = str(hhmmss)
        parts = s.split(":")
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            q = mm // 15  # 0,1,2,3
            return hh * 4 + q
    except Exception:
        pass
    return None


# =========================================================
# PAYLOADS TOPOFF (AYER/Hoy, 48 columnas)
# =========================================================

def build_heatmap_payloads_topoff(
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
    Igual a build_heatmap_payloads_fast del main pero adaptado a TopOff:
    - SIN network
    - CON noc_cluster (cluster) a nivel sitio/nodo
    - meta por sitio/nodo
    """
    if df_meta is None or df_meta.empty:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # --- fechas hoy/ayer ---
    if today is None:
        if df_ts is not None and not df_ts.empty and "fecha" in df_ts.columns:
            today = _max_date_str(df_ts["fecha"]) or _day_str(datetime.now())
        else:
            today = _day_str(datetime.now())
    if yday is None:
        yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # --- métricas requeridas ---
    metrics_needed = {
        m for v in valores_order for m in VALORES_MAP_TOPOFF.get(v, (None, None)) if m
    }
    if not metrics_needed:
        return None, None, {"total_rows": 0, "offset": 0, "limit": limit, "showing": 0}

    # --- base meta ---
    base = df_meta.drop_duplicates(subset=META_COLS_TOPOFF)[META_COLS_TOPOFF].reset_index(drop=True)

    # --- expand por valores_order ---
    rows_all_list = []
    for v in valores_order:
        rf = base.copy()
        rf["valores"] = v

        if alarm_only and alarm_keys is not None:
            keys_ok = set(alarm_keys)
            # ahora la llave incluye también cluster
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
            ]
            df_long = df_long.melt(
                id_vars=META_COLS_TOPOFF,
                value_vars=um_cols,
                var_name="metric",
                value_name="value",
            )
            UM_TO_VAL = {um: name for name, (_, um) in VALORES_MAP_TOPOFF.items() if um}
            df_long["valores"] = df_long["metric"].map(UM_TO_VAL)

            df_maxu = (
                df_long
                .dropna(subset=["valores"])
                .groupby(META_COLS_TOPOFF + ["valores"], as_index=False)["value"]
                .max()
                .rename(columns={"value":"max_unit"})
            )
            rows_all = rows_all.merge(
                df_maxu,
                on=META_COLS_TOPOFF + ["valores"],
                how="left"
            )
        else:
            rows_all["max_unit"] = np.nan
    else:
        rows_all["max_unit"] = np.nan

    rows_all["__ord_max_unit__"] = rows_all["max_unit"].astype(float).fillna(float("-inf"))
    rows_all = rows_all.sort_values("__ord_max_unit__", ascending=False, kind="stable")

    # --- paginado ---
    total_rows = len(rows_all)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    rows_page = rows_all.iloc[start:end].reset_index(drop=True)

    # --- keys visibles y df_small TS ---
    keys_df = rows_page[META_COLS_TOPOFF].drop_duplicates().reset_index(drop=True)
    keys_df["rid"] = np.arange(len(keys_df))

    if df_ts is None or df_ts.empty:
        df_small = pd.DataFrame()
    else:
        df_small = df_ts.loc[df_ts["fecha"].astype(str).isin([yday, today])].merge(
            keys_df, on=META_COLS_TOPOFF
        )

        # índice 15m dentro del día
        df_small["q15"] = df_small["hora"].apply(_safe_q15_to_idx)

        # offset 0..191 (ayer 0..95, hoy 96..191)
        df_small["offset192"] = df_small["q15"] + np.where(
            df_small["fecha"].astype(str) == today, 96, 0
        )

        df_small = df_small.dropna(subset=["offset192"])
        df_small["offset192"] = df_small["offset192"].astype(int)

    # --- maps por métrica ---
    metric_maps = {}
    if not df_small.empty:
        for m in metrics_needed:
            if m in df_small.columns:
                sub = df_small[["rid", "offset192", m]].dropna()
                # si hay múltiples muestras en el mismo bin, decide agregación:
                # MAX, LAST, MEAN... aquí pongo LAST por timestamp natural
                sub = sub.sort_values("offset192")
                metric_maps[m] = dict(zip(zip(sub["rid"], sub["offset192"]), sub[m]))
            else:
                metric_maps[m] = {}
    else:
        metric_maps = {m: {} for m in metrics_needed}

    def _row192_raw(metric, rid):
        mp = metric_maps.get(metric) or {}
        return [mp.get((rid, off)) for off in range(192)]

    # amarrar rid real a cada fila (aunque se repita por valores_order)
    rows_page = rows_page.merge(
        keys_df,
        on=META_COLS_TOPOFF,
        how="left",
        validate="many_to_one"
    )

    # --- ejes ---

    x_dt = _build_x_dt_15m(yday) + _build_x_dt_15m(today)
    z_pct, z_unit = [], []
    z_pct_raw, z_unit_raw = [], []

    y_labels, row_detail = [], []
    row_last_ts, row_max_pct, row_max_unit = [], [], []

    all_scores_pct, all_scores_unit = [], []

    for r in rows_page.itertuples(index=False):
        rid = int(getattr(r, "rid"))

        valores = r.valores
        pm, um = VALORES_MAP_TOPOFF.get(valores, (None, None))

        nodeb = getattr(r, "nodeb", "") or getattr(r, "NODEB", "") or ""
        tech = r.technology
        vend = r.vendor

        region = getattr(r, "region", "") or ""
        province = getattr(r, "province", "") or ""
        municipality = getattr(r, "municipality", "") or ""
        site_att = getattr(r, "site_att", "") or ""
        rnc = getattr(r, "rnc", "") or ""

        # etiqueta visual (no incluimos cluster aquí, pero sí participa en la llave/meta)
        y_labels.append(f"{nodeb} | {tech}/{vend}/{valores}")

        # detail consistente con parser del hover:
        # tech/vendor/region/province/municipality/site/rnc/nodeb/valores
        # (cluster no se incluye en este string para no romper los parsers actuales)
        row_detail.append(
            f"{tech}/{vend}/{region}/{province}/{municipality}/{site_att}/{rnc}/{nodeb}/{valores}"
        )

        row_raw = _row192_raw(pm, rid) if pm else [None] * 192
        row_raw_u = _row192_raw(um, rid) if um else [None] * 192

        # % -> score continuo
        if pm:
            orient, thr = _sev_cfg(pm, None, UMBRAL_CFG)
            row_color = [
                _sev_score_continuo(v, orient, thr, max_ratio=2.0) if v is not None else None
                for v in row_raw
            ]
            z_pct.append(row_color)
            z_pct_raw.append(row_raw)
            all_scores_pct += [s for s in row_color if s is not None]
        else:
            z_pct.append([None]*48)
            z_pct_raw.append(row_raw)

        # UNIT -> normalizado
        if um:
            mn, mx = _prog_cfg(um, None, UMBRAL_CFG)
            row_norm = [
                _normalize(v, mn, mx) if v is not None else None
                for v in row_raw_u
            ]
            z_unit.append(row_norm)
            z_unit_raw.append(row_raw_u)
            all_scores_unit += [s for s in row_norm if s is not None]
        else:
            z_unit.append([None]*48)
            z_unit_raw.append(row_raw_u)

        # stats por fila
        arr_u = np.array([v if isinstance(v,(int,float)) else np.nan for v in row_raw_u], float)
        arr_p = np.array([v if isinstance(v,(int,float)) else np.nan for v in row_raw], float)
        valid_idx = np.where(np.isfinite(arr_u if np.isfinite(arr_u).any() else arr_p))[0]
        last_label = str(x_dt[int(valid_idx[-1])]).replace("T"," ")[:16] if valid_idx.size else ""

        row_last_ts.append(last_label)
        row_max_pct.append(np.nanmax(arr_p) if np.isfinite(arr_p).any() else np.nan)
        row_max_unit.append(np.nanmax(arr_u) if np.isfinite(arr_u).any() else np.nan)

    # rangos dinámicos
    zmin_pct, zmax_pct = (min(all_scores_pct), max(all_scores_pct)) if all_scores_pct else (0.0, 1.0)
    zmin_unit, zmax_unit = (min(all_scores_unit), max(all_scores_unit)) if all_scores_unit else (0.0, 1.0)

    pct_payload = {
        "z": z_pct,
        "z_raw": z_pct_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "severity",
        "zmin": zmin_pct,
        "zmax": zmax_pct,
        "title": "% IA / % DC",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
    }

    unit_payload = {
        "z": z_unit,
        "z_raw": z_unit_raw,
        "x_dt": x_dt,
        "y": y_labels,
        "color_mode": "progress",
        "zmin": zmin_unit,
        "zmax": zmax_unit,
        "title": "Unidades",
        "row_detail": row_detail,
        "row_last_ts": row_last_ts,
        "row_max_pct": row_max_pct,
        "row_max_unit": row_max_unit,
    }

    page_info = {
        "total_rows": total_rows,
        "offset": start,
        "limit": limit,
        "showing": len(rows_page),
        "height": _hm_height(len(rows_page)),
    }

    return pct_payload, unit_payload, page_info


# =========================================================
# FIGURA HEATMAP (Plotly) TOPOFF
# =========================================================

def build_heatmap_figure_topoff(payload, *, height=750, decimals=2):
    if not payload:
        return go.Figure()

    z       = payload["z"]
    z_raw   = payload.get("z_raw") or z
    x       = payload.get("x_dt") or payload.get("x")
    y       = payload["y"]
    zmin    = payload["zmin"]
    zmax    = payload["zmax"]
    mode    = payload.get("color_mode", "severity")
    detail  = payload.get("row_detail") or y

    # colores
    if mode == "severity":
        colorscale = [
            [0/3, SEV_COLORS["excelente"]],
            [1/3, SEV_COLORS["bueno"]],
            [2/3, SEV_COLORS["regular"]],
            [3/3, SEV_COLORS["critico"]],
        ]
    else:
        colorscale = [
            [0.0, "#f8f9fa"],
            [1.0, "#0d6efd"],
        ]

    # customdata
    customdata = []
    for i, row in enumerate(z_raw):
        arr = np.array([v if isinstance(v,(int,float)) else np.nan for v in row], dtype=float)
        if np.isfinite(arr).any():
            valid_idx = np.where(np.isfinite(arr))[0]
            last_idx = int(valid_idx[-1])
            last_label = str(x[last_idx]).replace("T"," ")[:16]
        else:
            last_label = "—"

        # detail: tech/vendor/region/province/mun/site/rnc/nodeb/valores
        parts = (detail[i] if i < len(detail) else str(y[i])).split("/", 8)
        tech   = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        region = parts[2] if len(parts) > 2 else ""
        prov   = parts[3] if len(parts) > 3 else ""
        mun    = parts[4] if len(parts) > 4 else ""
        site   = parts[5] if len(parts) > 5 else ""
        rnc    = parts[6] if len(parts) > 6 else ""
        nodeb  = parts[7] if len(parts) > 7 else ""
        valor  = parts[8] if len(parts) > 8 else ""

        def _fmt_cell(v):
            if not np.isfinite(v):
                return ""
            return f"{v:,.{decimals}f}" if decimals > 0 else f"{v:,.0f}"

        row_cd = []
        for j in range(len(x)):
            raw_cell = arr[j] if j < len(arr) else np.nan
            raw_s = _fmt_cell(raw_cell)
            row_cd.append([tech, vendor, region, prov, mun, site, rnc, nodeb, raw_s, last_label, valor])
        customdata.append(row_cd)

    hover_tmpl = (
        "<span style='font-size:120%; font-weight:700'>%{customdata[8]}</span><br>"
        "<span style='opacity:0.85'>%{x|%Y-%m-%d %H:%M}</span><br>"
        "──────────<br>"
        "DETALLE<br>"
        "<b>Site:</b> %{customdata[5]}<br>"
        "<b>RNC:</b> %{customdata[6]}<br>"
        "<b>NodeB:</b> %{customdata[7]}<br>"
        "<b>Última hora con registro:</b> %{customdata[9]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        showscale=False,
        customdata=customdata,
        hovertemplate=hover_tmpl,
        hoverongaps=False,
        xgap=0.5, ygap=0.5,
    ))

    ONE_HOUR_MS = 3600 * 1000
    fig.update_xaxes(
        type="date",
        dtick=ONE_HOUR_MS,  # ticks cada hora aunque haya bins de 15m
        showticklabels=False,
        ticks="",
        fixedrange=True,
    )

    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        showgrid=False,
        zeroline=False,
        title="",
        automargin=False,
        fixedrange=True,
        categoryorder="array",
        categoryarray=y,
        autorange="reversed",
    )

    if isinstance(x, (list, tuple)) and len(x) >= 97:
        fig.add_vline(
            x=x[96],
            line_width=3,
            line_color="rgba(0,0,0,0.75)",
            line_dash="solid",
            layer="above"
        )

    fig.update_layout(
        autosize=False,
        height=height,
        margin=dict(l=4, r=4, t=MARG_TOP, b=MARG_BOTTOM),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea"),
        hoverlabel=dict(bgcolor="#222", bordercolor="#444", font=dict(color="#fff", size=13)),
        uirevision="keep",
    )
    return fig

# =========================================================
# SUMMARY TABLE TOPOFF
# =========================================================

def render_heatmap_summary_table_topoff(
    pct_payload,
    unit_payload,
    *,
    pct_decimals=2,
    unit_decimals=0,
    active_y=None,
):
    src = pct_payload or unit_payload
    if not src:
        return dbc.Alert("Sin filas para mostrar.", color="secondary", className="mb-0")

    y = src.get("y") or []
    detail = src.get("row_detail") or y
    row_last_ts = (unit_payload or pct_payload).get("row_last_ts") or []
    z_raw_pct = (pct_payload or {}).get("z_raw")
    z_raw_unit = (unit_payload or {}).get("z_raw")

    cols = [
        ("NodeB", "w-nodeb"),
        ("Tech", "w-tech"),
        ("Vendor", "w-vendor"),
        ("Valor", "w-valor"),
        ("Última hora", "w-ultima"),
        ("Valor de la última muestra (%)", "w-num"),
        ("Valor de la última muestra (UNIT)", "w-num"),
    ]
    thead = html.Thead(
        html.Tr([html.Th(n, className=c) for n, c in cols]),
        className="table-dark"
    )

    def _fmt_full(v):
        try:
            f = float(v)
            if not np.isfinite(f):
                return ""
            return f"{f:,.2f}"
        except Exception:
            return ""

    body_rows = []
    for i in range(len(y)):
        # detail ahora es tech/vendor/region/prov/mun/site/rnc/nodeb/valores
        parts = (detail[i] if i < len(detail) else str(y[i])).split("/", 8)
        tech   = parts[0] if len(parts) > 0 else ""
        vendor = parts[1] if len(parts) > 1 else ""
        nodeb  = parts[7] if len(parts) > 7 else ""
        valor  = parts[8] if len(parts) > 8 else ""

        last_pct  = _last_numeric(z_raw_pct[i]) if z_raw_pct and i < len(z_raw_pct) else None
        last_unit = _last_numeric(z_raw_unit[i]) if z_raw_unit and i < len(z_raw_unit) else None

        ultima_txt = _only_time(row_last_ts[i] if i < len(row_last_ts) else "")
        pct_txt    = _fmt(last_pct, pct_decimals)
        unit_txt   = _fmt(last_unit, unit_decimals)

        body_rows.append(
            html.Tr([
                html.Td(
                    html.Span(
                        html.Span(nodeb, className="unflip"),
                        className="ellipsis-left"
                    ),
                    className="w-nodeb",
                    title=nodeb
                ),
                html.Td(tech, className="w-tech", title=tech),
                html.Td(
                    html.Span(_vendor_initial(vendor), className="vendor-initial"),
                    className="td-vendor w-vendor",
                    title=vendor
                ),
                html.Td(valor, className="w-valor", title=valor),
                html.Td(ultima_txt, className="w-ultima ta-center", title=ultima_txt),
                html.Td(pct_txt, className="w-num ta-right", title=_fmt_full(last_pct)),
                html.Td(unit_txt, className="w-num ta-right", title=_fmt_full(last_unit)),
            ])
        )

    return dbc.Table(
        [thead, html.Tbody(body_rows)],
        striped=True, bordered=False, hover=True, size="sm",
        className="mb-0 table-dark kpi-table topoff-table compact"
    )


# =========================================================
# TIME HEADERS
# =========================================================

def build_time_header_children_by_dates(fecha_str: str):
    try:
        today_dt = datetime.strptime(fecha_str, "%Y-%m-%d") if fecha_str else datetime.utcnow()
    except Exception:
        today_dt = datetime.utcnow()
    yday_dt = today_dt - timedelta(days=1)
    today_s = today_dt.strftime("%Y-%m-%d")
    yday_s  = yday_dt.strftime("%Y-%m-%d")

    dates_children = [
        html.Div(yday_s,  className="cell"),
        html.Div(today_s, className="cell sep"),
    ]

    hours_children = []
    for i in range(48):
        hh = i if i < 24 else i - 24
        cls = ["cell"]
        if i == 24:
            cls.append("sep")
        if hh % 6 == 0:
            cls.append("tick6")
        label = html.Span(f"{hh:02d}", className="lbl") if (hh % 3 == 0) else None
        hours_children.append(html.Div(label, className=" ".join(cls)))

    return dates_children, hours_children


