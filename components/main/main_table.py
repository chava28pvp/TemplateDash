from dash import html
import math
import dash_bootstrap_components as dbc
import pandas as pd
from src.Utils.umbrales.utils_umbrales import cell_severity, progress_cfg
import numpy as np

# =========================
# Configuración / Constantes
# =========================

# Keys que identifican una fila (NO se prefijan por network, se muestran una sola vez a la izquierda)
ROW_KEYS = ["fecha", "hora", "vendor", "noc_cluster", "technology"]

# Grupos SOLO de métricas (sin fecha/hora/vendor/cluster/tech)
BASE_GROUPS = [
    ("INTEG", ["alarmas", "integrity", "integrity_deg_pct"]),
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
    "integrity": "Integ",
    "integrity_deg_pct": "%",

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
    "alarmas": "Ocurr",
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

    # Caso especial: hora 'HH:MM:SS' -> 'HH:MM'
    if colname == "hora":
        if isinstance(v, str):
            # Si viene como '06:00:00' o '06:00'
            return v[:5]
        # Por si algún día viene como datetime/time:
        try:
            from datetime import time, datetime
            if isinstance(v, (time, datetime)):
                return v.strftime("%H:%M")
        except Exception:
            pass

    if isinstance(v, float):
        if pd.isna(v) or math.isinf(v):
            return ""
        if colname == "ps_traff_gb":
            return f"{int(v):,}"
        return f"{v:,.1f}"

    if isinstance(v, int):
        return f"{v:,}"

    return str(v)


def _progress_cell(
    value,
    *,
    vmin=0.0,
    vmax=100.0,
    label_tpl="{value:.1f}",
    color=None,
    striped=True,
    animated=True,
    decimals=1,
    width_px=140,
    show_value_right=False,
    scale="linear",          # "linear" o "log"
):
    # Detecta faltantes/inválidos -> solo pista gris, sin número
    try:
        real = float(value)
    except (TypeError, ValueError):
        real = None

    if real is None or pd.isna(real) or math.isinf(real):
        return html.Div("", className="kb kb--empty", style={"--kb-width": f"{width_px}px"})

    if vmax <= vmin:
        vmax = vmin + 1.0

    # --- normalización según escala ---
    if scale == "log":
        def _log(x: float) -> float:
            return math.log10(max(x, 0.0) + 1.0)  # evita log(0)

        vmin_n = _log(vmin)
        vmax_n = _log(vmax)
        real_n = _log(real)
    else:
        vmin_n = vmin
        vmax_n = vmax
        real_n = real

    if vmax_n <= vmin_n:
        vmax_n = vmin_n + 1.0

    pct = (real_n - vmin_n) / (vmax_n - vmin_n) * 100.0
    pct = max(0.0, min(pct, 100.0))

    # --- etiqueta (valor real) ---
    if label_tpl:
        try:
            label = label_tpl.format(value=real)
        except Exception:
            label = f"{real:.{decimals}f}"
    else:
        label = f"{real:.{decimals}f}"

    # ============================
    # Caso especial: pct == 0  → valor 0 (o igual a min)
    # ============================
    if pct <= 0.0:
        # Pista gris, SIN barra de color, pero con el texto centrado y más oscuro
        inner = html.Div(
            label,
            className="kb-zero-label",
            style={
                "width": "100%",
                "height": "100%",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontSize": "0.70rem",
                "fontWeight": "700",
                "color": "#343a40",  # texto más oscuro para que no se pierda
                "lineHeight": "1",
            },
        )
        bar = html.Div(
            inner,
            className="kb kb--empty kb--zero",
            style={"--kb-width": f"{width_px}px"},
        )
        # Para 0 ignoramos show_value_right: siempre adentro del track
        return bar

    # --- caso normal (pct > 0): barra coloreada con el label dentro ---
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
    Muestra solo la inicial del vendor (E, N, H, S...), con tooltip.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return html.Span("", className="vendor-empty")

    raw = str(val).strip()
    initial = (raw[0]).upper() if raw else "?"

    return html.Div(
        html.Span(
            initial,
            title=raw,  # tooltip nativo
            className="vendor-initial"
        ),
        className="cell-key",
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

def pivot_by_network(df_long: pd.DataFrame, networks=None, order_map=None) -> pd.DataFrame:
    if networks is None:
        networks = sorted(df_long["network"].dropna().unique().tolist())

    df = df_long[df_long["network"].isin(networks)].copy()
    if df.empty:
        return df

    value_cols = [c for c in VALUE_COLS if c in df.columns]
    if not value_cols:
        return df[INDEX_KEYS].drop_duplicates().reset_index(drop=True)

    wide = df.pivot_table(
        index=INDEX_KEYS,
        columns="network",
        values=value_cols,
        aggfunc="first",
        sort=False,  # 👈 importante
    )

    wide.columns = [f"{net}__{val}" for (val, net) in wide.columns]
    wide = wide.reset_index()

    # 👇 aplica orden estable si te lo pasan
    if order_map:
        wide["_ord"] = wide[INDEX_KEYS].apply(
            lambda r: order_map.get(tuple(r.values.tolist()), 10**9),
            axis=1
        )
        wide = wide.sort_values("_ord").drop(columns=["_ord"])

    return wide



# =========================
# Header 3 niveles (keys + grupos por network)
# =========================

def build_header_3lvl(groups_3lvl, end_of_group_set, sort_state=None):
    sort_col = (sort_state or {}).get("column")
    ascending = (sort_state or {}).get("ascending", True)

    # Nivel 1: keys fijos (igual que antes)
    left = []
    for k in ROW_KEYS:
        label = DISPLAY_NAME_BASE.get(k, k).title()

        if k == "noc_cluster":
            # 👇 El <th> es clickeable, pero visualmente sigue siendo un th normal
            left.append(
                html.Th(
                    label,
                    id="main-cluster-header-reset",  # ID para el callback
                    n_clicks=0,  # habilita clicks
                    rowSpan=3,
                    className=f"th-left th-{k} th-clickable",
                )
            )
        else:
            left.append(
                html.Th(
                    label,
                    rowSpan=3,
                    className=f"th-left th-{k}",
                )
            )

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
            is_sorted = (sort_col == c) or (sort_col == base)
            arrow = "▲" if (is_sorted and ascending) else ("▼" if is_sorted else "↕")

            inner = html.Div([
                html.Span(_label_base(base), className="th-label"),
                html.Button(
                    arrow,
                    id={"type": "sort-btn", "col": c},
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
def render_kpi_table_multinet(
    df_in: pd.DataFrame,
    networks=None,
    sort_state=None,
    progress_max_by_col=None,
    integrity_baseline_map=None,
):
    if df_in is None or df_in.empty:
        return dbc.Alert("Sin datos para los filtros seleccionados.", color="warning", className="my-3")

    # 1) Detectar si es long o wide
    is_long = "network" in df_in.columns
    df_long = df_in.copy()

    # 2) Derivar networks si no se especifican explícitamente
    if networks is None:
        if is_long:
            networks = sorted(df_long["network"].dropna().unique().tolist())
        else:
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

    # --- PROGRESS_MAX_BY_COL: usar dict externo o calcular localmente ---
    if progress_max_by_col is not None:
        PROGRESS_MAX_BY_COL = progress_max_by_col
    else:
        PROGRESS_MAX_BY_COL = {}
        for col in PROGRESS_COLS:
            if col in df_wide.columns:
                serie = df_wide[col]
                # ignorar NaN / inf
                valid = serie.replace([np.inf, -np.inf], np.nan).dropna()
                if not valid.empty:
                    PROGRESS_MAX_BY_COL[col] = float(valid.max())
                else:
                    PROGRESS_MAX_BY_COL[col] = None
            else:
                PROGRESS_MAX_BY_COL[col] = None

    if sort_state:
        sort_col_req = (sort_state or {}).get("column")
        resolved = _resolve_sort_col(df_wide, METRIC_ORDER, sort_col_req)
        if resolved and resolved in df_wide.columns:
            asc = bool((sort_state or {}).get("ascending", True))
            df_wide = df_wide.sort_values(by=resolved, ascending=asc, na_position="last")

    # 5) Header (pasa sort_state para flecha)
    header = build_header_3lvl(groups_3lvl, END_OF_GROUP, sort_state=sort_state)
    VISIBLE_ORDER = ROW_KEYS + METRIC_ORDER  # por si lo usas luego
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

        # keys a la izquierda (con vendor inicial y tooltips en todas)
        for key in ROW_KEYS:
            val = _safe_get(row, key)

            if key == "vendor":
                cell_content = vendor_badge(val)
                tds.append(html.Td(cell_content, className=f"td-key td-{key}"))

            elif key == "noc_cluster":
                content = _fmt_number(val, key)
                # sacamos también vendor y tech de la fila para el link
                vendor_val = _safe_get(row, "vendor")
                tech_val = _safe_get(row, "technology")

                btn = html.Button(
                    content or "",
                    id={
                        "type": "main-cluster-link",
                        "cluster": val,
                        "vendor": vendor_val,
                        "technology": tech_val,
                    },
                    n_clicks=0,
                    className="cluster-link-btn cell-key",  # puedes estilizarla en CSS
                    title=str(content) if content else "",
                )
                tds.append(
                    html.Td(
                        btn,
                        className=f"td-key td-{key}",
                    )
                )

            else:
                content = _fmt_number(val, key)
                tds.append(
                    html.Td(
                        html.Div(content, className="cell-key", title=str(content) if content else ""),
                        className=f"td-key td-{key}",
                    )
                )

        # métricas (centradas por CSS)
        for col in METRIC_ORDER:
            val = _safe_get(row, col)
            base_name = strip_net(col)
            net = col.split("__", 1)[0] if "__" in col else None

            # ======================================================
            # PROGRESS BARS
            # ======================================================
            if col in PROGRESS_COLS:
                cfg = progress_cfg(base_name, network=net, profile="main")

                col_max = PROGRESS_MAX_BY_COL.get(col)
                vmin = cfg.get("min", 0.0)

                # --- vmax: siempre el máximo real cuando exista, para que solo el mayor llegue al 100% ---
                if col_max is not None and col_max > vmin:
                    vmax = col_max
                else:
                    vmax = cfg.get("max", 100.0)

                # --- decidir si usamos escala log o lineal ---
                use_log = False
                if col in df_wide.columns:
                    serie = df_wide[col].replace([math.inf, -math.inf], math.nan).dropna()
                    min_pos = None
                    if not serie.empty:
                        serie_pos = serie[serie > 0]
                        if not serie_pos.empty:
                            min_pos = float(serie_pos.min())

                    if min_pos is not None and col_max and col_max > 0:
                        ratio = col_max / max(min_pos, 1.0)
                        # ajusta el umbral a tu gusto (10, 20, 50, etc.)
                        use_log = ratio >= 20

                cell = _progress_cell(
                    val,
                    vmin=vmin,
                    vmax=vmax,
                    label_tpl=cfg.get("label", "{value:.1f}"),
                    decimals=cfg.get("decimals", 1),
                    width_px=80,  # más compacto
                    show_value_right=False,
                    scale="log" if use_log else "linear",
                )
                td_type_cls = "td-progress"

            # ======================================================
            # CELDAS NORMALES
            # ======================================================
            else:
                num_val = None if (val is None or (isinstance(val, float) and pd.isna(val))) else val

                # --- NUEVO: % de degrade vs baseline (solo texto, sin colores) ---
                if base_name == "integrity_deg_pct":
                    vendor_val = _safe_get(row, "vendor")
                    cluster_val = _safe_get(row, "noc_cluster")
                    tech_val = _safe_get(row, "technology")

                    key = (net, vendor_val, cluster_val, tech_val)
                    baseline = integrity_baseline_map.get(key) if integrity_baseline_map else None

                    integ_col = col.replace("integrity_deg_pct", "integrity")
                    integ_val = _safe_get(row, integ_col)

                    txt = ""
                    if (
                            baseline is not None
                            and isinstance(integ_val, (int, float))
                            and not pd.isna(integ_val)
                            and baseline > 0
                    ):
                        ratio = float(integ_val) / float(baseline)
                        # porcentaje RESTANTE respecto a la media:
                        # baseline=100, actual=80 → 80.0
                        # baseline=100, actual=50 → 50.0
                        health_pct = max(0.0, min(100.0, ratio * 100.0))
                        txt = f"{health_pct:.1f}"

                    # aquí la idea es NO pintarla, solo mostrar el número
                    cell = html.Div(txt, className="cell-neutral")

                # --- LÓGICA ESPECIAL PARA INTEGRITY (esta sí se pinta) ---
                elif base_name == "integrity" and isinstance(num_val, (int, float)):
                    vendor_val = _safe_get(row, "vendor")
                    cluster_val = _safe_get(row, "noc_cluster")
                    tech_val = _safe_get(row, "technology")
                    key = (net, vendor_val, cluster_val, tech_val)

                    baseline = None
                    if integrity_baseline_map is not None:
                        baseline = integrity_baseline_map.get(key)

                    if baseline is not None and baseline > 0:
                        ratio = float(num_val) / float(baseline)
                        # aquí ya decides si usas 0.2, 0.8, etc.
                        if ratio <= 0.799:
                            cls = "cell-integrity-degraded"
                        else:
                            cls = "cell-neutral"
                    else:
                        cls = "cell-neutral"

                    cell = html.Div(_fmt_number(val, base_name), className=cls)

                # --- SEVERITY normal ---
                elif (col in SEVERITY_COLS) and isinstance(num_val, (int, float)):
                    cls = f"cell-{cell_severity(base_name, float(num_val), network=net, profile='main')}"
                    cell = html.Div(_fmt_number(val, base_name), className=cls)

                # --- neutro ---
                else:
                    cls = "cell-neutral"
                    cell = html.Div(_fmt_number(val, base_name), className=cls)

                td_type_cls = "td-plain"

            # clases comunes de la celda KPI
            td_cls = "td-cell " + td_type_cls + (" td-end-of-group" if col in END_OF_GROUP else "")
            tds.append(html.Td(cell, className=td_cls))

        body_rows.append(html.Tr(tds))

    body = html.Tbody(body_rows)
    table = dbc.Table(
        [header, body],
        bordered=False,
        hover=True,
        responsive=True,
        striped=True,
        size="sm",
        className="kpi-table compact",
    )
    return dbc.Card(dbc.CardBody([html.H4("Tabla principal", className="mb-3"), table]), className="shadow-sm")


