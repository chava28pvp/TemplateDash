from dash import html
import dash_bootstrap_components as dbc
from src.Utils.utils_umbrales import cell_severity, progress_cfg


# 1) DICCIONARIO de etiquetas visibles por columna
DISPLAY_NAME = {
    "fecha": "Fecha",
    "hora": "Hora",
    "vendor": "Vendor",
    "integrity": "Integrity",
    "noc_cluster": "Cluster",
    "lcs_ps_rate": "%DC",
    "lcs_cs_rate": "%DC",
    "ps_failure_rrc_percent": "%IA",
    "ps_failure_rrc": "FAIL",
    "ps_failures_rab_percent": "%IA",
    "ps_failures_rab": "FAIL",
    "cs_failures_rrc_percent": "%IA",
    "cs_failures_rrc": "FAIL",
    "cs_failures_rab_percent": "%IA",
    "cs_failures_rab": "FAIL",
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
    (">2%", ["vendor"]),
    (">5%", ["fecha"]),
    (">10%", ["hora"]),
    ("      ", ["noc_cluster"]),
    ("      ", ["integrity"]),
    ("CS_TRAFF", ["delta_total_erlangs_nocperf","total_erlangs_nocperf"]),
    ("CS_RRC", ["cs_failures_rrc_percent", "cs_failures_rrc"]),
    ("CS_RAB", ["cs_failures_rab_percent", "ps_failures_rab"]),
    ("CS_DROP", ["lcs_cs_rate", "cs_abnormal_releases"]),
    ("PS_TRAFF", ["delta_total_mbytes_nocperf", "total_mbytes_nocperf"]),
    ("PS_RRC", ["ps_failure_rrc_percent", "ps_failure_rrc"]),
    ("PS_RAB", ["ps_failures_rab_percent", "ps_failures_rab"]),
    ("PS_DROP", ["lcs_ps_rate", "ps_abnormal_releases"]),

]

# Orden visible = concatenación de todos los sub-headers
VISIBLE_ORDER = [col for _, cols in GROUPS for col in cols]

# Última columna de cada grupo (para marcar divisores verticales)
END_OF_GROUP = {cols[-1] for _, cols in GROUPS}

# Columnas con progress bar (0-100)
# Sustituye PROGRESS_COLS por una configuración más rica:
PROGRESS_COLS = [
    "cs_failures_rrc", "ps_failures_rab", "cs_abnormal_releases",
    "ps_failure_rrc", "ps_failures_rab", "ps_abnormal_releases"
]


# Columnas con semáforo
SEVERITY_COLS = [
    "cs_failures_rrc_percent","lcs_cs_rate",
    "ps_failure_rrc_percent"
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

def _progress_cell(value, *, vmin=0.0, vmax=100.0, label_tpl="{value:.1f}",
                   color=None, striped=True, animated=True, decimals=1):
    try:
        real = float(value if value is not None else 0.0)
    except:
        real = 0.0

    if vmax <= vmin:
        vmax = vmin + 1.0
    pct = (real - vmin) / (vmax - vmin) * 100.0
    pct = max(0.0, min(pct, 100.0))

    label = label_tpl.format(value=real) if label_tpl else f"{real:.{decimals}f}"

    classes = ["kb", "kb--primary"]
    if striped:  classes.append("is-striped")
    if animated: classes.append("is-animated")

    container_style = {}
    if color:
        container_style["--kb-fill"] = color

    return html.Div(
        html.Div(label, className="kb__fill", style={"width": f"{pct:.2f}%"}),
        className=" ".join(classes),
        style=container_style,
        role="progressbar",
        **{
            "aria-valuemin": f"{vmin:.0f}",
            "aria-valuemax": f"{vmax:.0f}",
            "aria-valuenow": f"{real:.0f}"
        }
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
            # dentro del loop de columnas en render_kpi_table:
            if col in PROGRESS_COLS:
                cfg = progress_cfg(col)
                cell = _progress_cell(
                    val,
                    vmin=cfg["min"],
                    vmax=cfg["max"],
                    label_tpl=cfg["label"],
                    decimals=cfg["decimals"],
                    # Opcional: color según severidad
                    # color={"ok":"#4caf50","warn":"#ffb300","bad":"#e53935"}.get(
                    #     cell_severity(col, float(val) if isinstance(val,(int,float)) else None)
                    # )
                )

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



