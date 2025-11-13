from dash import html
import dash_bootstrap_components as dbc
import math
import pandas as pd

# Reutilizamos tus utilidades de umbrales
from src.Utils.umbrales.utils_umbrales import cell_severity, progress_cfg

# =============== Configuración visual ===============

ROW_KEYS = ["fecha", "hora", "tech", "region", "province", "municipality", "site_att", "rnc", "nodeb"]

# SIN META (como pediste)
BASE_GROUPS = [
    ("PS_TRAFF", ["ps_traff_gb"]),
    ("PS_RRC", ["ps_rrc_ia_percent", "ps_rrc_fail"]),
    ("PS_RAB", ["ps_rab_ia_percent", "ps_rab_fail"]),
    ("PS_S1", ["ps_s1_ia_percent", "ps_s1_fail"]),
    ("PS_DROP", ["ps_drop_dc_percent", "ps_drop_abnrel"]),
    ("CS_TRAFF", ["cs_traff_erl"]),
    ("CS_RRC", ["cs_rrc_ia_percent", "cs_rrc_fail"]),
    ("CS_RAB", ["cs_rab_ia_percent", "cs_rab_fail"]),
    ("CS_DROP", ["cs_drop_dc_percent", "cs_drop_abnrel"]),
    ("3G_RTX/4G_TNL%", ["unav", "rtx_tnl_tx_percent", "tnl_abn"]),
    ("", ["tnl_fail"])
]

DISPLAY = {
    # keys
    "fecha": "Fecha", "hora": "Hora", "tech": "Tech", "region": "Region",
    "province": "Province", "municipality": "Municipality", "site_att": "Site ATT",
    "rnc": "RNC", "nodeb": "NodeB",
    # PS
    "ps_traff_gb": "GB", "ps_rrc_ia_percent": "%IA", "ps_rrc_fail": "FAIL",
    "ps_rab_ia_percent": "%IA", "ps_rab_fail": "FAIL",
    "ps_s1_ia_percent": "%IA", "ps_s1_fail": "FAIL",
    "ps_drop_dc_percent": "%DC", "ps_drop_abnrel": "ABNREL",
    # CS
    "cs_traff_erl": "ERL", "cs_rrc_ia_percent": "%IA", "cs_rrc_fail": "FAIL",
    "cs_rab_ia_percent": "%IA", "cs_rab_fail": "FAIL",
    "cs_drop_dc_percent": "%DC", "cs_drop_abnrel": "ABNREL",
    # DISP/TNL
    "unav": "Unav", "rtx_tnl_tx_percent": "%Tx", "tnl_fail": "TNL FAIL", "tnl_abn": "TNL ABN",

}

# Columnas que llevan PROGRESS BAR (fails y abnrel)
PROGRESS_COLS = {
    "ps_rrc_fail", "ps_rab_fail", "ps_s1_fail", "ps_drop_abnrel",
    "cs_rrc_fail", "cs_rab_fail", "cs_drop_abnrel",
    # Si quieres que TNL también use barra, descomenta:
    # "tnl_fail", "tnl_abn",
}

# Columnas que pintan severidad (porcentaje)
SEVERITY_COLS = {
    "ps_rrc_ia_percent", "ps_rab_ia_percent", "ps_s1_ia_percent", "ps_drop_dc_percent",
    "cs_rrc_ia_percent", "cs_rab_ia_percent", "cs_drop_dc_percent",
    "rtx_tnl_tx_percent",  # ← también es porcentaje
}

# =============== Helpers ===============

def _fmt(v, col=None):
    if v is None:
        return ""
    if isinstance(v, float):
        if pd.isna(v) or math.isinf(v):
            return ""
        # casos particulares
        if col in ("ps_traff_gb",):
            return f"{int(v):,}"
        return f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)

def _progress_cell(value, *, vmin=0.0, vmax=100.0, label_tpl="{value:.1f}",
                   striped=True, animated=True, decimals=1, width_px=140, color=None, show_value_right=False):
    # robustez
    try:
        real = float(value)
    except (TypeError, ValueError):
        real = None

    if real is None or pd.isna(real) or math.isinf(real):
        return html.Div("", className="kb kb--empty", style={"--kb-width": f"{width_px}px"})

    if vmax <= vmin:
        vmax = vmin + 1.0

    pct = max(0.0, min((real - vmin) / (vmax - vmin) * 100.0, 100.0))
    label = label_tpl.format(value=real) if label_tpl else f"{real:.{decimals}f}"

    classes = ["kb", "kb--primary"]
    if striped:
        classes.append("is-striped")
    if animated:
        classes.append("is-animated")

    container_style = {"--kb-width": f"{width_px}px"}
    if color:
        container_style["--kb-fill"] = color

    bar = html.Div(
        html.Div(label, className="kb__fill", style={"width": f"{pct:.2f}%"}),
        className=" ".join(classes),
        style=container_style,
        role="progressbar",
        **{
            "aria-valuemin": f"{vmin:.0f}",
            "aria-valuemax": f"{vmax:.0f}",
            "aria-valuenow": f"{real:.0f}",
        },
    )
    if show_value_right:
        return html.Div([bar, html.Div(label, className="kb-value")], className="kb-wrap")
    return bar

# =============== Header 2 niveles ===============

def build_header(sort_state=None):
    sort_col = (sort_state or {}).get("column")
    ascending = (sort_state or {}).get("ascending", True)

    row1 = [
        html.Th(DISPLAY.get(k, k).title(), rowSpan=2, className=f"th-left th-{k}")
        for k in ROW_KEYS
    ]
    for grp, cols in BASE_GROUPS:
        row1.append(html.Th(grp, colSpan=len(cols), className="th-group"))

    row2 = []
    for _, cols in BASE_GROUPS:
        for c in cols:
            is_sorted = (sort_col == c)
            arrow = "▲" if (is_sorted and ascending) else ("▼" if is_sorted else "↕")
            inner = html.Div([
                html.Span(DISPLAY.get(c, c), className="th-label"),
                html.Button(
                    arrow,
                    id={"type": "topoff-sort-btn", "col": c},
                    n_clicks=0,
                    className="sort-btn",
                    **{"aria-label": f"Ordenar {c}"}
                )
            ], className="th-sort-wrap")
            cls = "th-sub" + (" th-sorted" if is_sorted else "")
            row2.append(html.Th(inner, className=cls))
    return html.Thead([html.Tr(row1), html.Tr(row2)])

# =============== Render principal ===============

def render_topoff_table(df: pd.DataFrame, sort_state=None):
    if df is None or df.empty:
        return dbc.Alert("Sin datos para los filtros seleccionados.", color="warning", className="my-3")

    metric_cols = [c for _, cols in BASE_GROUPS for c in cols]
    visible = [c for c in ROW_KEYS + metric_cols if c in df.columns]

    thead = build_header(sort_state=sort_state)

    # body
    body_rows = []
    for _, r in df[visible].iterrows():
        tds = []
        # keys
        for k in ROW_KEYS:
            val = r.get(k, None)
            tds.append(html.Td(html.Div(_fmt(val, k), className="cell-key"), className=f"td-key td-{k}"))

        # métricas
        for col in metric_cols:
            if col not in df.columns:
                continue
            val = r[col]

            if col in PROGRESS_COLS:
                cfg = progress_cfg(col, network=None)  # no hay 'network' en esta tabla
                cell = _progress_cell(
                    val,
                    vmin=cfg.get("min", 0.0),
                    vmax=cfg.get("max", 100.0),
                    label_tpl=cfg.get("label", "{value:.1f}"),
                    decimals=cfg.get("decimals", 1),
                    width_px=140,
                    show_value_right=False,
                )
                td = html.Td(cell, className="td-cell")
            else:
                # severidad si aplica
                if col in SEVERITY_COLS and isinstance(val, (int, float)) and not pd.isna(val):
                    sev = cell_severity(col, float(val), network=None)
                    cls = f"cell-{sev}"
                else:
                    cls = "cell-neutral"
                td = html.Td(html.Div(_fmt(val, col), className=cls), className="td-cell")

            tds.append(td)

        body_rows.append(html.Tr(tds))

    table = dbc.Table(
        [thead, html.Tbody(body_rows)],
        bordered=False, hover=True, responsive=True, striped=True, size="sm",
        className="kpi-table compact"
    )
    return dbc.Card(dbc.CardBody([html.H4("TopOff - Tabla", className="mb-3"), table]), className="shadow-sm")
