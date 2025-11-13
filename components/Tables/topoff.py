# components/Tables/topoff_table.py
from dash import html
import dash_bootstrap_components as dbc
import math
import pandas as pd

ROW_KEYS = ["fecha", "hora", "tech", "region", "province", "municipality", "site_att", "rnc", "nodeb"]

# Grupos por temática (similar a la principal)
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
    # META
    "archivo_fuente": "Archivo", "fecha_ejecucion": "Ejecución",
}

def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if pd.isna(v) or math.isinf(v):
            return ""
        return f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)

def build_header():
    # Fila 1: keys + grupos
    row1 = [
        html.Th(DISPLAY.get(k, k).title(), rowSpan=2, className=f"th-left th-{k}")
        for k in ROW_KEYS
    ]
    for grp, cols in BASE_GROUPS:
        row1.append(html.Th(grp, colSpan=len(cols), className="th-group"))
    # Fila 2: subheaders
    row2 = []
    for _, cols in BASE_GROUPS:
        for c in cols:
            row2.append(html.Th(DISPLAY.get(c, c), className="th-sub"))
    return html.Thead([html.Tr(row1), html.Tr(row2)])

def render_topoff_table(df: pd.DataFrame):
    if df is None or df.empty:
        return dbc.Alert("Sin datos para los filtros seleccionados.", color="warning", className="my-3")

    # Asegura columnas para este orden visible
    metric_cols = [c for _, cols in BASE_GROUPS for c in cols]
    visible = [c for c in ROW_KEYS + metric_cols if c in df.columns]

    thead = build_header()
    body_rows = []
    for row in df[visible].itertuples(index=False, name=None):
        tds = [html.Td(_fmt(v), className=f"td-key td-{col}") for v, col in zip(row[:len(ROW_KEYS)], ROW_KEYS)]
        # métricas
        for v, col in zip(row[len(ROW_KEYS):], visible[len(ROW_KEYS):]):
            tds.append(html.Td(html.Div(_fmt(v), className="cell-neutral"), className="td-cell"))
        body_rows.append(html.Tr(tds))

    table = dbc.Table(
        [thead, html.Tbody(body_rows)],
        bordered=False, hover=True, responsive=True, striped=True, size="sm",
        className="kpi-table compact"
    )
    return dbc.Card(dbc.CardBody([html.H4("TopOff - Tabla", className="mb-3"), table]), className="shadow-sm")
