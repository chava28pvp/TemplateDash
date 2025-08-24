from dash import html
import dash_bootstrap_components as dbc
from src.utils import cell_severity


# 1) DICCIONARIO de etiquetas visibles por columna (puedes ajustar libremente)
DISPLAY_NAME = {
    "fecha": "Fecha",
    "hora": "Hora",
    "vendor": "Vendor",
    "noc_cluster": "Cluster",
    "lcs_ps_rate": "%DC",
    "lcs_cs_rate": "%DC",
    "ps_failure_rrc_percent": "%IA",
    "ps_failures_rab_percent": "%IA",
    "cs_failures_rrc_percent": "%IA",
    "cs_failures_rab_percent": "%IA",
    "total_mbytes_nocperf": "GB",
    "delta_total_mbytes_nocperf": "DELTA",
    "total_erlangs_nocperf": "ERL",
    "delta_total_erlangs_nocperf": "DELTA",
    "traffic_gb_att": "GB",
    "delta_traffic_gb_att": "DELTA",
    "traffic_amr_att": "ERL",
    "delta_traffic_amr_att": "DELTA",
    "traffic_gb_plmn2": "GB",
    "delta_traffic_gb_plmn2": "DELTA",
    "traffic_amr_plmn2": "ERL",
    "delta_traffic_amr_plmn2": "DELTA",
    "ps_abnormal_releases": "ABNREL",
    "cs_abnormal_releases": "ABNREL",
}

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

def _label(col: str) -> str:
    return DISPLAY_NAME.get(col, col)  # fallback al nombre real si no está mapeado

def _build_header():
    top_cells = [html.Th(title, colSpan=len(cols), className="th-group") for title, cols in GROUPS]
    sub_cells = []
    for col in VISIBLE_ORDER:
        cls = "th-sub"
        if col in END_OF_GROUP:
            cls += " th-end-of-group"
        sub_cells.append(html.Th(_label(col), className=cls))  # <<< usa etiqueta visible
    return html.Thead([html.Tr(top_cells), html.Tr(sub_cells)])

def _progress_cell(value, color=None, striped=True, animated=True, decimals=1):
    # Asegura 0–100
    try:
        val = float(value or 0.0)
    except:
        val = 0.0
    # Si tus datos vienen 0–1, descomenta:
    # if 0 <= val <= 1: val *= 100.0

    val = max(0.0, min(val, 100.0))
    label = f"{val:.{decimals}f}%"

    classes = ["kb", "kb--primary"]
    if striped:  classes.append("is-striped")
    if animated: classes.append("is-animated")

    # Puedes cambiar color por celda con CSS var:
    container_style = {}
    if color:
        container_style["--kb-fill"] = color  # ej: "#1e88e5"

    return html.Div(
        html.Div(label, className="kb__fill", style={"width": f"{val}%"}),
        className=" ".join(classes),
        style=container_style,
        role="progressbar",                 # accesible
        **{"aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": f"{val:.0f}"}
    )


def _fmt_number(v):
    if v is None: return "-"
    if isinstance(v, float):
        return f"{v:,.3f}"
    return str(v)

def render_kpi_table(df):
    if df is None or df.empty:
        return dbc.Alert("Sin datos para los filtros seleccionados.", color="warning", className="my-3")

    header = _build_header()

    body_rows = []
    for _, row in df.iterrows():
        tds = []
        for col in VISIBLE_ORDER:
            val = row.get(col, None)
            # dentro de render_kpi_table, en el loop de columnas:
            if col in PROGRESS_COLS:
                cell = _progress_cell(val)
            else:
                if col in SEVERITY_COLS:
                    # SOLO estas columnas llevan semáforo
                    sev = cell_severity(col, float(val) if isinstance(val, (int, float)) else None)
                    cls = f"cell-{sev}"  # cell-ok | cell-warn | cell-bad
                else:
                    cls = "cell-neutral"  # sin color de fondo

                cell = html.Div(_fmt_number(val), className=cls)

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



