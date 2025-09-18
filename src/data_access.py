# src/data_access.py
from datetime import datetime, timedelta

import pandas as pd
from functools import lru_cache
from sqlalchemy import create_engine, text, bindparam
from typing import Dict, Tuple, Optional, List

from components.Tables.histograma import VALORES_MAP
from .Utils.alarmados import alarm_threshold_for, load_threshold_cfg, excess_base_for
from .config import SQLALCHEMY_URL

# =========================================================
# Engine / Config
# =========================================================

_engine = None
_TABLE_NAME = "Dashboard_Master"

def get_engine():
    """Singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True, pool_recycle=1800)
    return _engine


# =========================================================
# Column mapping (friendly -> real DB column)
# OJO: columnas con % requieren backticks en SQL.
# =========================================================

COLMAP = {
    # Identificadores
    "network": "Network",
    "technology": "Technology",
    "vendor": "Vendor",
    "noc_cluster": "Noc_Cluster",
    "fecha": "Date",
    "hora": "Time",

    # Integridad / PS
    "integrity": "INTEGRITY",
    "ps_traff_delta": "PS_TRAFF_DELTA",
    "ps_traff_gb": "PS_TRAFF_GB",
    "ps_rrc_ia_percent": "PS_RRC_%IA",
    "ps_rrc_fail": "PS_RRC_FAIL",
    "ps_rab_ia_percent": "PS_RAB_%IA",
    "ps_rab_fail": "PS_RAB_FAIL",
    "ps_s1_ia_percent": "PS_S1_%IA",
    "ps_s1_fail": "PS_S1_FAIL",
    "ps_drop_dc_percent": "PS_DROP_%DC",
    "ps_drop_abnrel": "PS_DROP_ABNREL",

    # CS
    "cs_traff_delta": "CS_TRAFF_DELTA",
    "cs_traff_erl": "CS_TRAFF_ERL",
    "cs_rrc_ia_percent": "CS_RRC_%IA",
    "cs_rrc_fail": "CS_RRC_FAIL",
    "cs_rab_ia_percent": "CS_RAB_%IA",
    "cs_rab_fail": "CS_RAB_FAIL",
    "cs_drop_dc_percent": "CS_DROP_%DC",
    "cs_drop_abnrel": "CS_DROP_ABNREL",
}

# Orden “deseado” para el DataFrame expuesto al resto de la app.
BASE_COLUMNS = [
    "fecha", "hora", "vendor", "noc_cluster", "network", "technology",
    "integrity",
    "ps_traff_delta", "ps_traff_gb",
    "ps_rrc_ia_percent", "ps_rrc_fail",
    "ps_rab_ia_percent", "ps_rab_fail",
    "ps_s1_ia_percent", "ps_s1_fail",
    "ps_drop_dc_percent", "ps_drop_abnrel",
    "cs_traff_delta", "cs_traff_erl",
    "cs_rrc_ia_percent", "cs_rrc_fail",
    "cs_rab_ia_percent", "cs_rab_fail",
    "cs_drop_dc_percent", "cs_drop_abnrel",
]

# Mínimo seguro si ninguna columna mapea (evita SELECT *)
_MIN_SAFE_COLUMNS = ["fecha", "hora", "vendor", "noc_cluster", "network", "technology"]

_ALARM_KPIS = [
    "ps_rrc_fail",
    "ps_rab_fail",
    "ps_s1_fail",
    "ps_drop_abnrel",
    "cs_rrc_fail",
    "cs_rab_fail",
]
_ALARM_KPIS_heatmap = [
    "ps_rrc_fail",
    "ps_rab_fail",
    "ps_drop_abnrel",
    "cs_rrc_fail",
    "cs_rab_fail",
]
# =========================================================
# Helpers internos
# =========================================================

def _quote(colname: str) -> str:
    """Backticks para columnas (soporta %, palabras reservadas, etc.)."""
    return f"`{colname}`"

def _quote_table(name: str) -> str:
    return f"`{name}`"

def _prepare_stmt_with_expanding(
    sql,
    use_vendors=False,
    use_clusters=False,
    use_networks=False,
    use_technologies=False,
):
    """Agrega bindparams expanding=True para listas."""
    stmt = text(sql)
    if use_vendors:
        stmt = stmt.bindparams(bindparam("vendors", expanding=True))
    if use_clusters:
        stmt = stmt.bindparams(bindparam("clusters", expanding=True))
    if use_networks:
        stmt = stmt.bindparams(bindparam("networks", expanding=True))
    if use_technologies:
        stmt = stmt.bindparams(bindparam("technologies", expanding=True))
    return stmt

def _as_list(x):
    """Convierte a lista; strings se tratan como un solo valor; filtra vacíos."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [v for v in x if v not in (None, "")]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [x]

@lru_cache(maxsize=1)
def _existing_columns():
    """
    Devuelve set de columnas reales existentes en la tabla.
    Cacheado para evitar golpear INFORMATION_SCHEMA en cada request.
    """
    eng = get_engine()
    sql = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = :tbl
    """
    with eng.connect() as conn:
        rows = conn.execute(text(sql), {"tbl": _TABLE_NAME}).fetchall()
    return {r[0] for r in rows}

def _resolve_columns(requested_friendly_cols):
    """
    Filtra columnas amigables a solo las que existen en BD (vía COLMAP y INFORMATION_SCHEMA).
    Retorna lista de columnas amigables válidas, o un fallback mínimo si ninguna mapea.
    """
    existing_real = _existing_columns()
    cols = []
    for friendly in requested_friendly_cols:
        real = COLMAP.get(friendly)
        if real and real in existing_real:
            cols.append(friendly)

    if cols:
        return cols

    # Fallback mínimo seguro
    fallback = [c for c in _MIN_SAFE_COLUMNS if COLMAP.get(c) in existing_real]
    return fallback

def _select_list_with_aliases(friendly_cols):
    """
    Construye la lista de SELECT con backticks y alias amigables.
    Aplica DATE_FORMAT a 'hora' → 'HH:MM:SS'.
    """
    if not friendly_cols:
        # Por el fallback de _resolve_columns no deberíamos llegar aquí.
        return ["*"]

    select_parts = []
    for friendly in friendly_cols:
        real = COLMAP[friendly]
        if friendly == "hora":
            select_parts.append(f"DATE_FORMAT({_quote(real)}, '%H:%i:%s') AS {friendly}")
        else:
            select_parts.append(f"{_quote(real)} AS {friendly}")
    return select_parts

def _filters_where_and_params(
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    """
    Genera cláusula WHERE y params compartidos por SELECT/COUNT.
    Devuelve: (where_sql, params, flags expanding)
    """
    where = ["1=1"]
    params = {}

    if fecha:
        where.append(f"{_quote(COLMAP['fecha'])} = :fecha")
        params["fecha"] = fecha

    if hora and str(hora).lower() != "todas":
        where.append(f"{_quote(COLMAP['hora'])} = :hora")
        params["hora"] = hora

    vendors = _as_list(vendors)
    clusters = _as_list(clusters)
    networks = _as_list(networks)
    technologies = _as_list(technologies)

    use_vendors = use_clusters = use_networks = use_technologies = False

    if vendors:
        where.append(f"{_quote(COLMAP['vendor'])} IN :vendors")
        params["vendors"] = vendors
        use_vendors = True
    if clusters:
        where.append(f"{_quote(COLMAP['noc_cluster'])} IN :clusters")
        params["clusters"] = clusters
        use_clusters = True
    if networks:
        where.append(f"{_quote(COLMAP['network'])} IN :networks")
        params["networks"] = networks
        use_networks = True
    if technologies:
        where.append(f"{_quote(COLMAP['technology'])} IN :technologies")
        params["technologies"] = technologies
        use_technologies = True

    return (
        " AND ".join(where),
        params,
        use_vendors,
        use_clusters,
        use_networks,
        use_technologies,
    )

def _build_flag_and_excess_cases(
    kpi: str,
    networks: Optional[List[str]],
    cfg
) -> Tuple[str, str, Dict[str, object]]:
    """
    Devuelve (flag_thr_expr, excess_thr_expr, params) donde:
      - flag_thr_expr   = CASE por red usando alarm_threshold_for
      - excess_thr_expr = CASE por red usando excess_base_for
    Todos con fallback a default/max, sin hardcode.
    """
    params: Dict[str, object] = {}

    # redes a considerar dentro del CASE
    per_net_cfg = (cfg.get("progress", {}).get(kpi, {}).get("per_network") or {})
    nets = networks or list(per_net_cfg.keys())

    # defaults
    flag_def = alarm_threshold_for(kpi, "", cfg)
    exc_def  = excess_base_for(kpi, "", cfg)
    if flag_def is None:
        flag_def = 0.0
    if exc_def is None:
        # si no hay excess_base ni max en default, cae a flag_def como último recurso
        exc_def = flag_def

    params[f"{kpi}_flag_thr_def"] = flag_def
    params[f"{kpi}_exc_thr_def"]  = exc_def

    flag_parts, exc_parts = [], []
    for i, net in enumerate(nets):
        flag_thr = alarm_threshold_for(kpi, net or "", cfg)
        exc_thr  = excess_base_for(kpi, net or "", cfg)

        # si no hay específicos, omite rama; caerá al ELSE default
        if flag_thr is not None:
            p_net = f"{kpi}_net_{i}"
            p_thr = f"{kpi}_flag_thr_{i}"
            params[p_net] = net
            params[p_thr] = flag_thr
            flag_parts.append(f"WHEN {_quote(COLMAP['network'])} = :{p_net} THEN :{p_thr}")

        if exc_thr is not None:
            p_net_e = f"{kpi}_enet_{i}"
            p_thr_e = f"{kpi}_exc_thr_{i}"
            params[p_net_e] = net
            params[p_thr_e] = exc_thr
            exc_parts.append(f"WHEN {_quote(COLMAP['network'])} = :{p_net_e} THEN :{p_thr_e}")

    if flag_parts:
        flag_expr = f"(CASE {' '.join(flag_parts)} ELSE :{kpi}_flag_thr_def END)"
    else:
        flag_expr = f":{kpi}_flag_thr_def"

    if exc_parts:
        exc_expr = f"(CASE {' '.join(exc_parts)} ELSE :{kpi}_exc_thr_def END)"
    else:
        exc_expr = f":{kpi}_exc_thr_def"

    return flag_expr, exc_expr, params
# =========================================================
# API pública
# =========================================================

def fetch_kpis(
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    limit=None,
    na_as_empty=False,

):
    """
    Consulta no paginada (útil para casos pequeños o descargas).
    Usa filtros y devuelve DataFrame con alias amigables y orden base.
    """
    # 1) columnas amigables válidas
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    # 2) where + params compartidos
    where_sql, params, uv, uc, un, ut = _filters_where_and_params(
        fecha, hora, vendors, clusters, networks, technologies
    )

    # 3) SELECT
    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
        ORDER BY {_quote(COLMAP['fecha'])} DESC, {_quote(COLMAP['hora'])} DESC
    """
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)

    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(sql, uv, uc, un, ut)
        df = pd.read_sql(stmt, conn, params=params)

    # 4) Asegura orden de columnas amigables
    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])

    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df


def fetch_kpis_paginated(
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page=1,
    page_size=50,
    na_as_empty=False,
):
    """
    Consulta paginada (server-side): retorna (df, total_rows).
    - Aplica COUNT(*) con los mismos filtros.
    - Devuelve solo la página solicitada con LIMIT/OFFSET.
    """
    # 1) columnas amigables válidas
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    # 2) where + params compartidos
    where_sql, params, uv, uc, un, ut = _filters_where_and_params(
        fecha, hora, vendors, clusters, networks, technologies
    )

    # 3) COUNT total
    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    # 4) SELECT paginado
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    sel_sql = f"""
        SELECT {", ".join(select_cols)}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
        ORDER BY {_quote(COLMAP['fecha'])} DESC, {_quote(COLMAP['hora'])} DESC
        LIMIT :limit OFFSET :offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        # COUNT
        stmt_count = _prepare_stmt_with_expanding(count_sql, uv, uc, un, ut)
        total = conn.execute(stmt_count, params).scalar() or 0

        # SELECT page
        sel_params = dict(params)
        sel_params.update({"limit": page_size, "offset": offset})
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, uv, uc, un, ut)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    # 5) Orden de columnas amigables
    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])

    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df, int(total)

def fetch_kpis_paginated_global_sort(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page=1,
    page_size=50,
    sort_by_friendly=None,   # ej. "ps_rrc_fail"
    sort_net=None,           # ej. "ATT" si viene de "ATT__ps_rrc_fail"
    ascending=True,
    na_as_empty=False,
):
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    where_sql, base_params, uv, uc, un, ut = _filters_where_and_params(
        fecha, hora, vendors, clusters, networks, technologies
    )

    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    order_dir = "ASC" if ascending else "DESC"

    if sort_by_friendly and sort_by_friendly in COLMAP:
        real_metric = _quote(COLMAP[sort_by_friendly])
    else:
        real_metric = f"{_quote(COLMAP['fecha'])}"

    if sort_net:
        metric_expr = f"MAX(CASE WHEN {_quote(COLMAP['network'])} = :_sort_net THEN {real_metric} END)"
    else:
        metric_expr = f"MAX({real_metric})"

    key_cols_real = [
        COLMAP["fecha"], COLMAP["hora"], COLMAP["vendor"], COLMAP["noc_cluster"], COLMAP["technology"]
    ]
    key_cols_sel = ", ".join(_quote(c) for c in key_cols_real)

    # 👇 Reemplazo de NULLS LAST por (expr IS NULL)
    # Esto empuja NULL al final tanto en ASC como en DESC
    nulls_last_prefix = f"({metric_expr} IS NULL), "

    base_params_page = dict(base_params)
    if sort_net:
        base_params_page["_sort_net"] = sort_net
    base_params_page.update({"_limit": page_size, "_offset": offset})

    keys_sql = f"""
        SELECT {key_cols_sel}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
        GROUP BY {key_cols_sel}
        ORDER BY
          {nulls_last_prefix}{metric_expr} {order_dir},
          {_quote(COLMAP['fecha'])} DESC,
          {_quote(COLMAP['hora'])} DESC
        LIMIT :_limit OFFSET :_offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        total = conn.execute(text(count_sql), base_params).scalar() or 0
        stmt_keys = _prepare_stmt_with_expanding(keys_sql, uv, uc, un, ut)
        key_rows = conn.execute(stmt_keys, base_params_page).fetchall()

    if not key_rows:
        return pd.DataFrame(columns=BASE_COLUMNS), int(total)

    params_b = {}
    or_parts = []
    for i, (d, t, v, c, tech) in enumerate(key_rows, start=1):
        or_parts.append(
            f"({_quote(COLMAP['fecha'])} = :d{i} AND {_quote(COLMAP['hora'])} = :t{i} "
            f"AND {_quote(COLMAP['vendor'])} = :v{i} AND {_quote(COLMAP['noc_cluster'])} = :c{i} "
            f"AND {_quote(COLMAP['technology'])} = :tech{i})"
        )
        params_b[f"d{i}"] = d
        params_b[f"t{i}"] = t
        params_b[f"v{i}"] = v
        params_b[f"c{i}"] = c
        params_b[f"tech{i}"] = tech

    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    sql_b = f"""
        SELECT {", ".join(select_cols)}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {" OR ".join(or_parts)}
    """

    with eng.connect() as conn:
        df_page = pd.read_sql(text(sql_b), conn, params=params_b)

    df_page = df_page.reindex(columns=[c for c in friendly_cols if c in df_page.columns])
    if na_as_empty and not df_page.empty:
        df_page = df_page.where(pd.notna(df_page), "")

    return df_page, int(total)

#ALARMADOS DATA ACCESS

def fetch_kpis_paginated_alarm_sort(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,        # puede ser 1 o varias; aplica CASE por fila
    technologies=None,
    page=1,
    page_size=50,
    na_as_empty=False,
):
    """
    Ordena por:
      1) número de KPIs en alarma (>= alarm del JSON por red),
      2) exceso total sobre umbrales (col - excess_base del JSON por red),
      3) Noc_Cluster (desempate).
    Incluye solo filas con >=1 KPI en alarma.
    """
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    cfg = load_threshold_cfg()  # cacheado

    # WHERE base y params
    where_sql, params, uv, uc, un, ut = _filters_where_and_params(
        fecha, hora, vendors, clusters, networks, technologies
    )

    nets_list = _as_list(networks) or []

    flag_terms: List[str] = []   # (col >= flag_thr_expr)
    excess_terms: List[str] = [] # GREATEST(COALESCE(col,0) - exc_thr_expr, 0)
    thr_params_all: Dict[str, object] = {}

    for kpi in _ALARM_KPIS:
        if kpi not in COLMAP:
            continue
        col_sql = _quote(COLMAP[kpi])

        # Si hay posibilidad de texto, fuerza CAST:
        num_col = f"CAST({col_sql} AS DECIMAL(20,6))"

        flag_thr_expr, exc_thr_expr, p = _build_flag_and_excess_cases(kpi, nets_list, cfg)
        thr_params_all.update(p)

        flag_terms.append(
            f"(CASE WHEN COALESCE({num_col}, 0) >= {flag_thr_expr} THEN 1 ELSE 0 END)"
        )

        # 2) exceso truncado a 0
        excess_terms.append(
            f"GREATEST(COALESCE({num_col}, 0) - {exc_thr_expr}, 0)"
        )

    # si no hay KPIs válidos, cae al orden normal
    if not flag_terms:
        return fetch_kpis_paginated(
            fecha=fecha, hora=hora, vendors=vendors, clusters=clusters,
            networks=networks, technologies=technologies,
            page=page, page_size=page_size, na_as_empty=na_as_empty
        )

    flags_sum  = " + ".join(flag_terms)
    excess_sum = " + ".join(excess_terms)

    # COUNT con al menos 1 alarma
    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
          AND ( ({flags_sum}) >= 1 )
    """

    # SELECT paginado aplicando el ORDER deseado
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    select_cols_debug = select_cols + [
        f"({flags_sum}) AS kpis_alarmados",
        f"({excess_sum}) AS exceso_total"
    ]

    sel_sql = f"""
        SELECT {", ".join(select_cols_debug)}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
          AND ( ({flags_sum}) >= 1 )
        ORDER BY
          ({flags_sum}) DESC,
          ({excess_sum}) DESC,
          {_quote(COLMAP['noc_cluster'])} ASC
        LIMIT :_limit OFFSET :_offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        # COUNT
        stmt_count = _prepare_stmt_with_expanding(count_sql, uv, uc, un, ut)
        total = conn.execute(stmt_count, {**params, **thr_params_all}).scalar() or 0

        # PAGE
        sel_params = {**params, **thr_params_all, "_limit": page_size, "_offset": offset}
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, uv, uc, un, ut)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])
    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df, int(total)

#ALARMADOS HEADMAP
def fetch_alarm_meta_for_heatmap(
    *,
    fecha: str,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    """
    Devuelve:
      - df_meta_heat: (technology, vendor, noc_cluster) ORDENADO por #KPIs alarmados (desc) y flag_hits (desc)
      - alarm_keys_set: {(technology, vendor, noc_cluster, network)} con ≥1 KPI alarmado (ayer u hoy)
    """
    # Fechas
    try:
        base_dt = datetime.strptime(fecha, "%Y-%m-%d") if fecha else datetime.utcnow()
    except Exception:
        base_dt = datetime.utcnow()
    yday_dt = base_dt - timedelta(days=1)
    today_str = base_dt.strftime("%Y-%m-%d")
    yday_str  = yday_dt.strftime("%Y-%m-%d")

    cfg = load_threshold_cfg()  # cacheado
    nets_list = _as_list(networks) or []

    # WHERE sin fecha/hora; fechas via IN
    where_sql, params, uv, uc, un, ut = _filters_where_and_params(
        fecha=None, hora=None, vendors=vendors, clusters=clusters,
        networks=networks, technologies=technologies
    )
    where_sql = f"({where_sql}) AND {_quote(COLMAP['fecha'])} IN (:f1, :f2)"
    params = {**params, "f1": yday_str, "f2": today_str}

    # KPIs a evaluar como "alarmados"
    try:
        alarm_kpis = list(_ALARM_KPIS_heatmap)  # si ya tienes lista global
    except NameError:
        # Fallback: toma todos los percent de VALORES_MAP
        alarm_kpis = sorted({pm for (_name, (pm, _um)) in VALORES_MAP.items() if pm and pm in COLMAP})

    if not alarm_kpis:
        return pd.DataFrame(columns=["technology","vendor","noc_cluster"]), set()

    # Construye expresiones CASE por KPI y parámetros de umbrales
    flag_cols_sql = []
    thr_params_all = {}

    # Esta función devuelve un SQL que probablemente use el nombre físico de columna (p.ej. `Network`).
    # Para este nivel (SELECT desde CTE base) debemos comparar contra el ALIAS 'network'.
    def _flag_expr_for(kpi: str) -> str:
        col_sql = _quote(COLMAP[kpi])
        num_col = f"CAST({col_sql} AS DECIMAL(20,6))"
        flag_thr_expr, _exc_thr_expr, p = _build_flag_and_excess_cases(kpi, nets_list, cfg)
        thr_params_all.update(p)
        # Fuerza a usar el alias 'network' en el CASE, no el nombre físico:
        flag_thr_expr_alias = flag_thr_expr.replace(_quote(COLMAP['network']), "network")
        return f"(CASE WHEN COALESCE({num_col}, 0) >= {flag_thr_expr_alias} THEN 1 ELSE 0 END)"

    for kpi in alarm_kpis:
        if kpi not in COLMAP:
            continue
        alias = f"f_{kpi}"
        flag_cols_sql.append(f"{_flag_expr_for(kpi)} AS {alias}")

    if not flag_cols_sql:
        return pd.DataFrame(columns=["technology","vendor","noc_cluster"]), set()

    flags_list_sql = ",\n            ".join(flag_cols_sql)
    sum_row_flags = " + ".join([f"COALESCE(f_{k},0)" for k in alarm_kpis])

    # Nombres SQL
    tbl    = _quote_table(_TABLE_NAME)
    f_tech = _quote(COLMAP["technology"])
    f_vend = _quote(COLMAP["vendor"])
    f_clus = _quote(COLMAP["noc_cluster"])
    f_net  = _quote(COLMAP["network"])

    # Construcción con CTEs; nota el paso intermedio flags_raw → flags
    sql = f"""
    WITH base AS (
        SELECT
            {f_tech} AS technology,
            {f_vend} AS vendor,
            {f_clus} AS noc_cluster,
            {f_net}  AS network,
            {", ".join(_quote(COLMAP[k]) for k in alarm_kpis)}
        FROM {tbl}
        WHERE {where_sql}
    ),
    flags_raw AS (
        SELECT
            technology, vendor, noc_cluster, network,
            {flags_list_sql}
        FROM base
    ),
    flags AS (
        SELECT
            fr.*,
            ({sum_row_flags}) AS row_flag_sum
        FROM flags_raw fr
    ),
    agg_trio AS (
        SELECT
            noc_cluster, vendor, technology,
            {", ".join([f"MAX(f_{k}) AS mx_{k}" for k in alarm_kpis])},
            SUM(row_flag_sum) AS flag_hits
        FROM flags
        GROUP BY noc_cluster, vendor, technology
    ),
    ranked AS (
        SELECT
            noc_cluster, vendor, technology,
            ({' + '.join([f"COALESCE(mx_{k},0)" for k in alarm_kpis])}) AS alarm_score,
            flag_hits
        FROM agg_trio
    )
    SELECT
        noc_cluster AS noc_cluster,
        vendor      AS vendor,
        technology  AS technology,
        alarm_score,
        flag_hits
    FROM ranked
    ORDER BY alarm_score DESC, flag_hits DESC, vendor ASC, technology ASC
    """

    # Consulta principal (ordenado)
    eng = get_engine()
    with eng.connect() as conn:
        stmt1 = _prepare_stmt_with_expanding(sql, uv, uc, un, ut)
        df_meta_heat = pd.read_sql(stmt1, conn, params={**params, **thr_params_all})

        # Consulta de keys por network (reutiliza mismo patrón flags_raw → flags)
        sql_keys_full = f"""
        WITH base AS (
            SELECT
                {f_tech} AS technology,
                {f_vend} AS vendor,
                {f_clus} AS noc_cluster,
                {f_net}  AS network,
                {", ".join(_quote(COLMAP[k]) for k in alarm_kpis)}
            FROM {tbl}
            WHERE {where_sql}
        ),
        flags_raw AS (
            SELECT
                technology, vendor, noc_cluster, network,
                {flags_list_sql}
            FROM base
        ),
        flags AS (
            SELECT
                fr.*,
                ({sum_row_flags}) AS row_flag_sum
            FROM flags_raw fr
        )
        SELECT DISTINCT technology, vendor, noc_cluster, network
        FROM flags
        WHERE row_flag_sum >= 1
        """
        stmt2 = _prepare_stmt_with_expanding(sql_keys_full, uv, uc, un, ut)
        df_alarm_keys = pd.read_sql(stmt2, conn, params={**params, **thr_params_all})

    if df_meta_heat is None or df_meta_heat.empty:
        return pd.DataFrame(columns=["technology","vendor","noc_cluster"]), set()

    alarm_keys_set = set(
        tuple(x) for x in df_alarm_keys[["technology","vendor","noc_cluster","network"]].itertuples(index=False, name=None)
    )

    # Devuelve sólo columnas base (pero ya vienen en orden)
    df_out = df_meta_heat[["technology","vendor","noc_cluster"]].drop_duplicates().reset_index(drop=True)
    return df_out, alarm_keys_set


