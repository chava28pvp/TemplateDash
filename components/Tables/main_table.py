from dash import html
import math
import dash_bootstrap_components as dbc
import pandas as pd
from src.Utils.umbrales.utils_umbrales import cell_severity, progress_cfg

# =========================
# Configuración / Constantes
# =========================

# Keys que identifican una fila (NO se prefijan por network, se muestran una sola vez a la izquierda)
ROW_KEYS = ["fecha", "hora", "vendor", "noc_cluster", "technology"]

# Grupos SOLO de métricas (sin fecha/hora/vendor/cluster/tech)
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

# Columnas base que llevan progress bar y severidad (sin prefijo de red)
BASE_PROGRESS_COLS = [
    "ps_rrc_fail", "ps_rab_fail", "ps_s1_fail", "ps_drop_abnrel",
    "cs_rrc_fail", "cs_rab_fail", "cs_drop_abnrel"
]
BASE_SEVERITY_COLS = [
    "ps_rrc_ia_percent", "ps_rab_ia_percent", "ps_s1_ia_percent", "ps_drop_dc_percent",
    "cs_rrc_ia_percent", "cs_rab_ia_percent", "cs_drop_dc_percent"
]

# Etiquetas visibles para alias (sin prefijo)
DISPLAY_NAME_BASE = {
    # ROW_KEYS
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
VENDOR_ICON = {
    "ericsson": "ericsson.svg",
    "nokia":    "nokia.svg",
    "huawei":   "huawei.svg",
    "samsung":  "samsung.svg",
}

# Derivados
INDEX_KEYS = ROW_KEYS
VALUE_COLS = sorted({c for _, cols in BASE_GROUPS for c in cols} | {"integrity"})


# =========================
# Helpers
# =========================

def strip_net(colname: str) -> str:
    """ATT__ps_rrc_fail -> ps_rrc_fail; si no hay prefijo, retorna igual."""
    if "__" in colname:
        return colname.split("__", 1)[1]
    return colname

def _resolve_sort_col(df: pd.DataFrame, metric_order: list[str], sort_col: str | None):
    if not sort_col:
        return None
    if sort_col in df.columns:
        return sort_col
    base = strip_net(sort_col)
    for c in metric_order:
        if strip_net(c) == base and c in df.columns:
            return c
    cand = [c for c in df.columns if c.endswith(f"__{base}")]
    return cand[0] if cand else None

def _label_base(base: str) -> str:
    return DISPLAY_NAME_BASE.get(base, base)

def _fmt_number(v, colname=None):
    if v is None:
        return ""
    if isinstance(v, float):
        if pd.isna(v) or math.isinf(v):
            return ""
        # Caso especial para ps_traff_gb → sin decimales
        if colname == "ps_traff_gb":
            return f"{int(v):,}"
        return f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _progress_cell(value, *, vmin=0.0, vmax=100.0, label_tpl="{value:.1f}",
                   color=None, striped=True, animated=True, decimals=1,
                   width_px=140, show_value_right=False):
    # Detecta faltantes/inválidos
    try:
        real = float(value)
    except (TypeError, ValueError):
        real = None

    if real is None or pd.isna(real) or math.isinf(real):
        # Celda vacía (sin barra ni label)
        return html.Div("", className="kb kb--empty", style={"--kb-width": f"{width_px}px"})

    if vmax <= vmin:
        vmax = vmin + 1.0

    pct = (real - vmin) / (vmax - vmin) * 100.0
    pct = max(0.0, min(pct, 100.0))

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

def vendor_badge(val):
    """
    Devuelve Img con el logo del vendor (si existe),
    si no, cae a la inicial (fallback).
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return html.Span("", className="vendor-empty")

    raw = str(val).strip()
    key = raw.lower()

    fname = VENDOR_ICON.get(key)
    if fname:
        # Ojo: si usas Dash, todo en /assets se sirve automáticamente
        src = f"/assets/vendor/{fname}"
        return html.Div(
            html.Img(
                src=src,
                alt=raw,
                title=raw,  # tooltip nativo
                className="vendor-icon"
            ),
            className="cell-key vendor-badge",
            **{"aria-label": f"Vendor {raw}"}
        )

    # Fallback: inicial como antes
    return html.Span(
        (raw[0]).upper(),
        title=raw,
        className="cell-key vendor-initial",
        **{"aria-label": f"Vendor {raw}"}
    )

# =========================
# Lógica multi-network
# =========================

def expand_groups_for_networks(networks: list[str]):
    groups_3lvl, visible_order, end_of_group = [], [], set()
    for net in networks:
        for grp_title, base_cols in BASE_GROUPS:
            cols = [f"{net}__{c}" for c in base_cols]
            groups_3lvl.append((net, grp_title, cols))
            visible_order.extend(cols)
            end_of_group.add(cols[-1])
    return groups_3lvl, visible_order, end_of_group


def prefixed_progress_cols(networks: list[str]):
    return {f"{net}__{c}" for net in networks for c in BASE_PROGRESS_COLS}


def prefixed_severity_cols(networks: list[str]):
    return {f"{net}__{c}" for net in networks for c in BASE_SEVERITY_COLS}

def pivot_by_network(df_long: pd.DataFrame, networks=None) -> pd.DataFrame:
    if networks is None:
        networks = sorted(df_long["network"].dropna().unique().tolist())
    df = df_long[df_long["network"].isin(networks)].copy()
    if df.empty:
       return df

    wide = df.pivot_table(index=INDEX_KEYS, columns="network", values=VALUE_COLS, aggfunc="first")
    wide.columns = [f"{net}__{val}" for (val, net) in wide.columns]
    wide = wide.reset_index()
    return wide



# =========================
# Header 3 niveles (keys + grupos por network)
# =========================

def build_header_3lvl(groups_3lvl, end_of_group_set, sort_state=None):
    sort_col = (sort_state or {}).get("column")
    ascending = (sort_state or {}).get("ascending", True)

    # Nivel 1: keys fijos (igual que antes)
    left = [
        html.Th(DISPLAY_NAME_BASE.get(k, k).title(),
                rowSpan=3,
                className=f"th-left th-{k}")  # 👈 clase específica por columna
        for k in ROW_KEYS
    ]

    # Nivel 1: Networks
    net_to_span = {}
    for net, _, cols in groups_3lvl:
        net_to_span[net] = net_to_span.get(net, 0) + len(cols)
    row1 = left + [html.Th(net, colSpan=span, className="th-network")
                   for net, span in net_to_span.items()]

    # Nivel 2: Grupos
    row2 = [html.Th(grp, colSpan=len(cols), className="th-group")
            for (_, grp, cols) in groups_3lvl]

    # Nivel 3: Subheaders con botón de sort
    row3 = []
    for _, _, cols in groups_3lvl:
        for c in cols:
            base = strip_net(c)
            # ¿esta columna es la actualmente ordenada?
            is_sorted = (sort_col == c) or (sort_col == base)
            arrow = "▲" if (is_sorted and ascending) else ("▼" if is_sorted else "↕")

            inner = html.Div([
                html.Span(_label_base(base), className="th-label"),
                html.Button(
                    arrow,
                    id={"type": "sort-btn", "col": c},  # ← pattern-matching ID
                    n_clicks=0,
                    className="sort-btn",
                    **{"aria-label": f"Ordenar {c}"}
                )
            ], className="th-sort-wrap")

            cls = "th-sub"
            if c in end_of_group_set:
                cls += " th-end-of-group"
            if is_sorted:
                cls += " th-sorted"
            row3.append(html.Th(inner, className=cls))

    return html.Thead([html.Tr(row1), html.Tr(row2), html.Tr(row3)])



# =========================
# Render principal
# =========================

# --- firma nueva (nota: agrega sort_state) ---
def render_kpi_table_multinet(df_in: pd.DataFrame, networks=None, sort_state=None):
    if df_in is None or df_in.empty:
        return dbc.Alert("Sin datos para los filtros seleccionados.", color="warning", className="my-3")

    # 1) Detectar si es long o wide
    is_long = "network" in df_in.columns  # long si tiene esta columna
    df_long = df_in.copy()

    # 2) Derivar networks si no se especifican explícitamente
    if networks is None:
        if is_long:
            networks = sorted(df_long["network"].dropna().unique().tolist())
        else:
            # inferir de columnas wide: prefijo antes de "__"
            nets = set()
            for c in df_in.columns:
                if "__" in c:
                    nets.add(c.split("__", 1)[0])
            networks = sorted(nets)

    # 3) Conseguir wide
    if is_long:
        df_wide = pivot_by_network(df_long, networks=networks)
    else:
        df_wide = df_in.copy()

    if df_wide is None or df_wide.empty:
        return dbc.Alert("Sin datos para las redes seleccionadas.", color="warning", className="my-3")

    # 4) Construir grupos/orden y aplicar sort si procede
    groups_3lvl, METRIC_ORDER, END_OF_GROUP = expand_groups_for_networks(networks)
    PROGRESS_COLS = prefixed_progress_cols(networks)
    SEVERITY_COLS = prefixed_severity_cols(networks)

    # ↓↓↓ APLICAR ORDEN ↓↓↓
    if sort_state:
        sort_col_req = (sort_state or {}).get("column")
        resolved = _resolve_sort_col(df_wide, METRIC_ORDER, sort_col_req)
        if resolved and resolved in df_wide.columns:
            asc = bool((sort_state or {}).get("ascending", True))
            df_wide = df_wide.sort_values(by=resolved, ascending=asc, na_position="last")

    # 5) Header (pasa sort_state para flecha)
    header = build_header_3lvl(groups_3lvl, END_OF_GROUP, sort_state=sort_state)
    VISIBLE_ORDER = ROW_KEYS + METRIC_ORDER
    idx_map = {c: i for i, c in enumerate(df_wide.columns)}

    def _safe_get(row_tuple, col):
        i = idx_map.get(col)
        if i is None:
            return None
        try:
            return row_tuple[i]
        except Exception:
            return None

    # 6) Body
    body_rows = []
    for row in df_wide.itertuples(index=False, name=None):
        tds = []

        # keys a la izquierda (si no existen en wide, intenta buscarlas en df_in)
        for key in ROW_KEYS:
            val = _safe_get(row, key)

            # Mostrar solo inicial en 'vendor' (y dejar el valor completo como tooltip)
            if key == "vendor":
                cell_content = vendor_badge(val)  # 👈 nuevo
                tds.append(
                    html.Td(cell_content, className=f"td-key td-{key}")
                )
            else:
                content = _fmt_number(val, key)
                tds.append(
                    html.Td(
                        html.Div(content, className="cell-key"),
                        className=f"td-key td-{key}"
                    )
                )


        # métricas
        for col in METRIC_ORDER:
            val = _safe_get(row, col)
            base_name = strip_net(col)
            net = col.split("__", 1)[0] if "__" in col else None

            if col in PROGRESS_COLS:
                cfg = progress_cfg(base_name, network=net)
                cell = _progress_cell(
                    val,
                    vmin=cfg.get("min", 0.0),
                    vmax=cfg.get("max", 100.0),
                    label_tpl=cfg.get("label", "{value:.1f}"),
                    decimals=cfg.get("decimals", 1),
                    width_px=140,
                    show_value_right=False,
                )
            else:
                num_val = None if (val is None or (isinstance(val, float) and pd.isna(val))) else val
                if (col in SEVERITY_COLS) and isinstance(num_val, (int, float)):
                    cls = f"cell-{cell_severity(base_name, float(num_val), network=net)}"
                else:
                    cls = "cell-neutral"
                cell = html.Div(_fmt_number(val, base_name), className=cls)

            td_cls = "td-cell" + (" td-end-of-group" if col in END_OF_GROUP else "")
            tds.append(html.Td(cell, className=td_cls))

        body_rows.append(html.Tr(tds))

    body = html.Tbody(body_rows)
    table = dbc.Table([header, body], bordered=False, hover=True, responsive=True, striped=True, size="sm",
                      className="kpi-table compact")
    return dbc.Card(dbc.CardBody([html.H4("Tabla principal", className="mb-3"), table]), className="shadow-sm")

