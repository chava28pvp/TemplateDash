from dash import html
import dash_bootstrap_components as dbc
from src.utils import cell_severity

# Columnas con progress bar (0-100)
PROGRESS_COLS = [
    "lcs_ps_rate",
    "lcs_cs_rate"
]

# Columnas con semáforo
SEVERITY_COLS = [
    "ps_failure_rrc_percent","ps_failures_rab_percent",
    "cs_failures_rrc_percent","cs_failures_rab_percent",
]

# Grupos (header superior) → columnas (sub-headers)
GROUPS = [
    ("Fecha/Hora", ["fecha","hora"]),
    ("Identidad", ["vendor","noc_cluster"]),
    ("LCS Rate", ["lcs_ps_rate","lcs_cs_rate"]),
    ("PS RRC", ["ps_failure_rrc_percent","ps_failures_rab_percent"]),
    ("CS RRC", ["cs_failures_rrc_percent","cs_failures_rab_percent"]),
    ("PS_TRAFF", ["total_mbytes_nocperf","delta_total_mbytes_nocperf"]),
    ("CS_TRAFF", ["total_erlangs_nocperf","delta_total_erlangs_nocperf"]),
    ("Traffic ATT", ["traffic_gb_att","delta_traffic_gb_att","traffic_amr_att","delta_traffic_amr_att"]),
    ("Traffic PLMN2", ["traffic_gb_plmn2","delta_traffic_gb_plmn2","traffic_amr_plmn2","delta_traffic_amr_plmn2"]),
    ("Abnormal Releases", ["ps_abnormal_releases","cs_abnormal_releases"]),
]

# Orden visible = concatenación de todos los sub-headers
VISIBLE_ORDER = [col for _, cols in GROUPS for col in cols]

# Última columna de cada grupo (para marcar divisores verticales)
END_OF_GROUP = {cols[-1] for _, cols in GROUPS}

def _progress_cell(value: float):
    try:
        val = float(value or 0)
    except:
        val = 0.0
    val = max(0.0, min(val, 100.0))
    # compacto
    return dbc.Progress(value=val, striped=True, animated=True,
                        className="kpi-progress", style={"height":"14px"})

def _fmt_number(v):
    if v is None: return "-"
    if isinstance(v, float):
        return f"{v:,.3f}"
    return str(v)

def _build_header():
    # Fila 1: headers de grupo (usa colspan)
    top_cells = []
    for title, cols in GROUPS:
        top_cells.append(html.Th(title, colSpan=len(cols), className="th-group"))
    top_row = html.Tr(top_cells)

    # Fila 2: sub-headers (los nombres actuales de columna)
    sub_cells = []
    for col in VISIBLE_ORDER:
        cls = "th-sub"
        if col in END_OF_GROUP:
            cls += " th-end-of-group"
        sub_cells.append(html.Th(col, className=cls))
    sub_row = html.Tr(sub_cells)

    return html.Thead([top_row, sub_row])

def render_kpi_table(df):
    if df is None or df.empty:
        return dbc.Alert("Sin datos para los filtros seleccionados.", color="warning", className="my-3")

    header = _build_header()

    body_rows = []
    for _, row in df.iterrows():
        tds = []
        for col in VISIBLE_ORDER:
            val = row.get(col, None)
            # Contenido de celda
            if col in PROGRESS_COLS:
                cell = _progress_cell(val)
            else:
                sev = cell_severity(col, float(val) if isinstance(val, (int,float)) else None) \
                      if col in SEVERITY_COLS else "ok"
                cell = html.Div(_fmt_number(val), className=f"cell-{sev}")

            # Clase para trazar divisor fuerte al fin de grupo
            td_cls = "td-cell"
            if col in END_OF_GROUP:
                td_cls += " td-end-of-group"

            tds.append(html.Td(cell, className=td_cls))
        body_rows.append(html.Tr(tds))
    body = html.Tbody(body_rows)

    table = dbc.Table(
        [header, body],
        bordered=False,
        hover=True,
        responsive=True,
        striped=True,
        size="sm",                        # tabla compacta
        className="kpi-table compact"     # clase para CSS
    )
    card = dbc.Card(
        dbc.CardBody([html.H4("Tabla principal", className="mb-3"), table]),
        className="shadow-sm"
    )
    return card
