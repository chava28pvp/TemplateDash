from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
import math

# ===============================================
# Configuración base (igual estructura que la tabla previa)
# ===============================================

ROW_KEYS = ["fecha", "hora", "vendor", "noc_cluster", "technology"]

BASE_GROUPS = [
    ("INTEG", ["integrity"]),
    ("PS_TRAFF", ["ps_traff_delta", "ps_traff_gb"]),
    ("PS_RRC",   ["ps_rrc_ia_percent", "ps_rrc_fail"]),
    ("PS_RAB",   ["ps_rab_ia_percent", "ps_rab_fail"]),
    ("PS_S1",    ["ps_s1_ia_percent", "ps_s1_fail"]),
    ("PS_DROP",  ["ps_drop_dc_percent", "ps_drop_abnrel"]),
    ("CS_TRAFF", ["cs_traff_delta", "cs_traff_erl"]),
    ("CS_RRC",   ["cs_rrc_ia_percent", "cs_rrc_fail"]),
    ("CS_RAB",   ["cs_rab_ia_percent", "cs_rab_fail"]),
    ("CS_DROP",  ["cs_drop_dc_percent", "cs_drop_abnrel"]),
]

DISPLAY_NAME_BASE = {
    # Keys
    "fecha": "Fecha",
    "hora": "Hora",
    "vendor": "Vendor",
    "technology": "Tech",
    "noc_cluster": "Cluster",

    # Métricas
    "integrity": "Integrity",

    "ps_traff_delta": "DELTA",
    "ps_traff_gb": "GB",
    "ps_rrc_ia_percent": "%IA",
    "ps_rrc_fail": "FAIL",
    "ps_rab_ia_percent": "%IA",
    "ps_rab_fail": "FAIL",
    "ps_s1_ia_percent": "%IA",
    "ps_s1_fail": "FAIL",
    "ps_drop_dc_percent": "%DC",
    "ps_drop_abnrel": "ABNREL",

    "cs_traff_delta": "DELTA",
    "cs_traff_erl": "ERL",
    "cs_rrc_ia_percent": "%IA",
    "cs_rrc_fail": "FAIL",
    "cs_rab_ia_percent": "%IA",
    "cs_rab_fail": "FAIL",
    "cs_drop_dc_percent": "%DC",
    "cs_drop_abnrel": "ABNREL",
}

VALUE_COLS = sorted({c for _, cols in BASE_GROUPS for c in cols} | {"integrity"})

# ===============================================
# Helpers
# ===============================================

def strip_net(colname: str) -> str:
    """ATT__ps_rrc_fail -> ps_rrc_fail; si no hay prefijo, retorna igual."""
    if "__" in colname:
        return colname.split("__", 1)[1]
    return colname


def _fmt_number(v, colname=None):
    if v is None:
        return ""
    if isinstance(v, float):
        if pd.isna(v) or math.isinf(v):
            return ""
        if colname == "ps_traff_gb":
            try:
                return f"{int(v):,}"
            except Exception:
                return f"{v:,.0f}"
        # por defecto una decimal
        return f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def expand_groups_for_networks(networks: list[str]):
    groups_3lvl, visible_order, end_of_group = [], [], set()
    for net in networks:
        for grp_title, base_cols in BASE_GROUPS:
            cols = [f"{net}__{c}" for c in base_cols]
            groups_3lvl.append((net, grp_title, cols))
            visible_order.extend(cols)
            end_of_group.add(cols[-1])
    return groups_3lvl, visible_order, end_of_group


def pivot_by_network(df_long: pd.DataFrame, networks=None) -> pd.DataFrame:
    if networks is None:
        networks = sorted(df_long.get("network", pd.Series(dtype=str)).dropna().unique().tolist())
    df = df_long[df_long["network"].isin(networks)].copy()
    if df.empty:
        return df
    wide = df.pivot_table(index=ROW_KEYS, columns="network", values=VALUE_COLS, aggfunc="first")
    # columnas como "NET__metric"
    wide.columns = [f"{net}__{val}" for (val, net) in wide.columns]
    wide = wide.reset_index()
    return wide

# ===============================================
# CSS (colócalo en assets/ si prefieres; aquí lo inyectamos inline)
# ===============================================

GRID_CSS = """
.kpi-grid { display: grid; gap: 2px; }
.kpi-grid-header { display: grid; align-items: stretch; }
.kpi-grid-row { display: grid; align-items: center; border-bottom: 1px solid #e9ecef; }

.grid-h1, .grid-h2, .grid-h3 { font-weight: 600; text-align: center; padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; }
.grid-h1 { font-size: 0.95rem; }
.grid-h2 { font-size: 0.9rem; }
.grid-h3 { font-size: 0.85rem; }

.grid-key { padding: 6px 8px; font-weight: 500; background: #ffffff; border: 1px solid #eef0f2; }
.grid-metric { padding: 6px 8px; text-align: right; border: 1px solid #f1f3f5; background: #fff; font-variant-numeric: tabular-nums; }

.grid-row-even .grid-key, .grid-row-even .grid-metric { background: #fcfcfd; }

.sticky-top { position: sticky; top: 0; z-index: 5; }
"""

# ===============================================
# Header con CSS Grid (3 niveles) + Body sin umbrales
# ===============================================

def _build_header_grid(groups_3lvl, key_col_widths, metric_col_width):
    """
    Devuelve (header_div, col_index_map, template_str)
      - header_div: contenedor con 3 filas de header usando CSS Grid, incluyendo spans
      - col_index_map: mapa {col_name -> índice de columna base-1} para el body
      - template_str: grid-template-columns
    """
    # 1) Índices de columna
    #   Keys: 1..len(ROW_KEYS)
    #   Métricas: len(ROW_KEYS)+1 .. len(ROW_KEYS)+len(METRIC_ORDER)
    visible_order = []
    for _, _, cols in groups_3lvl:
        visible_order.extend(cols)

    col_index_map = {c: i + 1 + len(ROW_KEYS) for i, c in enumerate(visible_order)}  # base-1

    # 2) Template de columnas
    key_cols = [f"{w}px" for w in key_col_widths]
    metric_cols = [f"{metric_col_width}px" for _ in visible_order]
    template = " ".join(key_cols + metric_cols)

    # 3) Fila 1: keys (con row-span 3) + networks
    cells_row1 = []
    # keys con row-span
    for i, key in enumerate(ROW_KEYS):
        cells_row1.append(
            html.Div(
                DISPLAY_NAME_BASE.get(key, key).title(),
                className="grid-h1",
                style={
                    "gridColumn": f"{i+1} / {i+2}",
                    "gridRow": "1 / span 3",
                },
            )
        )

    # span por network
    # calcular cuántas columnas de métricas tiene cada network
    from collections import OrderedDict
    net_span = OrderedDict()
    for net, grp, cols in groups_3lvl:
        net_span[net] = net_span.get(net, 0) + len(cols)

    start = len(ROW_KEYS) + 1
    for net, span in net_span.items():
        cells_row1.append(
            html.Div(
                net,
                className="grid-h1",
                style={
                    "gridColumn": f"{start} / span {span}",
                    "gridRow": "1 / 2",
                },
            )
        )
        start += span

    # 4) Fila 2: grupos por network
    cells_row2 = []
    start = len(ROW_KEYS) + 1
    for net, grp, cols in groups_3lvl:
        span = len(cols)
        cells_row2.append(
            html.Div(
                grp,
                className="grid-h2",
                style={
                    "gridColumn": f"{start} / span {span}",
                    "gridRow": "2 / 3",
                },
            )
        )
        start += span

    # 5) Fila 3: subheaders (métrica por métrica)
    cells_row3 = []
    # posición de la primera métrica
    cur = len(ROW_KEYS) + 1
    for _, _, cols in groups_3lvl:
        for c in cols:
            base = strip_net(c)
            cells_row3.append(
                html.Div(
                    DISPLAY_NAME_BASE.get(base, base),
                    className="grid-h3",
                    style={
                        "gridColumn": f"{cur} / {cur+1}",
                        "gridRow": "3 / 4",
                    },
                )
            )
            cur += 1

    header = html.Div(
        cells_row1 + cells_row2 + cells_row3,  # 👈 sin html.Style()
        className="kpi-grid-header sticky-top",
        style={"display": "grid", "gridTemplateColumns": template},
    )
    return header, col_index_map, template


# ===============================================
# Render principal (GRID)
# ===============================================

def render_kpi_grid_multinet(
    df_in: pd.DataFrame,
    networks=None,
    sort_state=None,  # reservado por si luego quieres replicar orden
    key_col_widths=(110, 70, 70, 160, 90),  # fecha, hora, vendor, cluster, tech
    metric_col_width=90,
    title="Tabla principal (Grid)",
):
    """
    Renderiza una "tabla" con CSS Grid, conservando la estructura de 3 niveles en el header.
    No usa umbrales ni progress bars; solo números formateados.
    """
    if df_in is None or df_in.empty:
        return dbc.Alert("Sin datos para los filtros seleccionados.", color="warning", className="my-3")

    is_long = "network" in df_in.columns

    # Derivar networks si no vienen
    if networks is None:
        if is_long:
            networks = sorted(df_in["network"].dropna().unique().tolist())
        else:
            nets = set()
            for c in df_in.columns:
                if "__" in c:
                    nets.add(c.split("__", 1)[0])
            networks = sorted(nets)

    # Pivot si es long
    if is_long:
        df_wide = pivot_by_network(df_in, networks=networks)
    else:
        df_wide = df_in.copy()

    if df_wide is None or df_wide.empty:
        return dbc.Alert("Sin datos para las redes seleccionadas.", color="warning", className="my-3")

    # Construcción de grupos y header
    groups_3lvl, METRIC_ORDER, END_OF_GROUP = expand_groups_for_networks(networks)
    header, col_index_map, template = _build_header_grid(groups_3lvl, key_col_widths, metric_col_width)

    # Body: una fila = un grid con el mismo template
    rows = []
    for rix, row in enumerate(df_wide.itertuples(index=False, name=None)):
        # acceso por nombre
        cols = list(df_wide.columns)
        idx = {c: i for i, c in enumerate(cols)}

        cells = []
        # Keys (primero N columnas)
        for i, key in enumerate(ROW_KEYS):
            val = row[idx[key]] if key in idx else None
            # vendor: inicial + tooltip
            if key == "vendor":
                txt = (str(val)[0]).upper() if val not in (None, "") else ""
                content = html.Span(txt, title=str(val) if val not in (None, "") else "")
            else:
                content = _fmt_number(val, key)
            cells.append(
                html.Div(
                    content,
                    className="grid-key",
                    style={"gridColumn": f"{i+1} / {i+2}"},
                )
            )

        # Métricas
        for col, j1 in col_index_map.items():
            val = row[idx[col]] if col in idx else None
            base = strip_net(col)
            cells.append(
                html.Div(
                    _fmt_number(val, base),
                    className="grid-metric",
                    style={"gridColumn": f"{j1} / {j1+1}"},
                )
            )

        rows.append(
            html.Div(
                cells,
                className=f"kpi-grid-row {'grid-row-even' if (rix % 2)==1 else ''}",
                style={"gridTemplateColumns": template},
            )
        )


    grid_inner = html.Div([header] + rows, className="kpi-grid compact")

    return dbc.Card(
        dbc.CardBody([
            html.H4(title, className="mb-3"),
            html.Div(grid_inner, className="kpi-grid-viewport"),
        ]),
        className="shadow-sm"
    )


