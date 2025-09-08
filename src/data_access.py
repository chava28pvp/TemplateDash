# src/data_access.py
# =========================================================
# Versión unificada con compatibilidad total para callbacks existentes.
# - Exporta COLMAP y las 4 funciones con los mismos nombres/firmas de antes:
#     fetch_kpis
#     fetch_kpis_paginated
#     fetch_kpis_paginated_global_sort
#     fetch_kpis_paginated_alarm_sort
# - Añade una función única interna/pública: fetch_kpis_unified
# - Mantiene todos los helpers, orden por KPI (métrica) y orden por alarmas.
# =========================================================

import pandas as pd
from functools import lru_cache
from sqlalchemy import create_engine, text, bindparam
from typing import Dict, Tuple, Optional, List, Iterable, Union
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

# KPIs considerados en modo “alarmas”
_ALARM_KPIS = [
    "ps_rrc_fail",
    "ps_rab_fail",
    "ps_s1_fail",
    "ps_drop_abnrel",
    "cs_rrc_fail",
    "cs_rab_fail",
]

# =========================================================
# Helpers internos
# =========================================================

def _quote(colname: str) -> str:
    return f"`{colname}`"


def _quote_table(name: str) -> str:
    return f"`{name}`"


def _prepare_stmt_with_expanding(
    sql: str,
    use_vendors: bool = False,
    use_clusters: bool = False,
    use_networks: bool = False,
    use_technologies: bool = False,
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


def _as_list(x: Optional[Union[Iterable, str]]):
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
    """Devuelve set de columnas reales existentes en la tabla."""
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


def _resolve_columns(requested_friendly_cols: List[str]) -> List[str]:
    existing_real = _existing_columns()
    cols = []
    for friendly in requested_friendly_cols:
        real = COLMAP.get(friendly)
        if real and real in existing_real:
            cols.append(friendly)
    if cols:
        return cols
    # Fallback mínimo
    return [c for c in _MIN_SAFE_COLUMNS if COLMAP.get(c) in existing_real]


def _select_list_with_aliases(friendly_cols: List[str]) -> List[str]:
    if not friendly_cols:
        return ["*"]
    parts = []
    for friendly in friendly_cols:
        real = COLMAP[friendly]
        if friendly == "hora":
            parts.append(f"DATE_FORMAT({_quote(real)}, '%H:%i:%s') AS {friendly}")
        else:
            parts.append(f"{_quote(real)} AS {friendly}")
    return parts


def _filters_where_and_params(
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    where = ["1=1"]
    params: Dict[str, object] = {}

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
    params: Dict[str, object] = {}

    per_net_cfg = (cfg.get("progress", {}).get(kpi, {}).get("per_network") or {})
    nets = networks or list(per_net_cfg.keys())

    flag_def = alarm_threshold_for(kpi, "", cfg)
    exc_def = excess_base_for(kpi, "", cfg)
    if flag_def is None:
        flag_def = 0.0
    if exc_def is None:
        exc_def = flag_def

    params[f"{kpi}_flag_thr_def"] = flag_def
    params[f"{kpi}_exc_thr_def"] = exc_def

    flag_parts, exc_parts = [], []
    for i, net in enumerate(nets):
        flag_thr = alarm_threshold_for(kpi, net or "", cfg)
        exc_thr = excess_base_for(kpi, net or "", cfg)
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

    flag_expr = (
        f"(CASE {' '.join(flag_parts)} ELSE :{kpi}_flag_thr_def END)" if flag_parts
        else f":{kpi}_flag_thr_def"
    )
    exc_expr = (
        f"(CASE {' '.join(exc_parts)} ELSE :{kpi}_exc_thr_def END)" if exc_parts
        else f":{kpi}_exc_thr_def"
    )

    return flag_expr, exc_expr, params


# =========================================================
# Helpers de SELECT / COUNT reutilizables para unificación
# =========================================================

def _count_total(where_sql: str, params: Dict, uv: bool, uc: bool, un: bool, ut: bool) -> int:
    count_sql = f"SELECT COUNT(*) AS total FROM {_quote_table(_TABLE_NAME)} WHERE {where_sql}"
    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(count_sql, uv, uc, un, ut)
        return int(conn.execute(stmt, params).scalar() or 0)


def _count_total_alarm(
    *, where_sql: str, params: Dict, uv: bool, uc: bool, un: bool, ut: bool,
    networks: Optional[List[str]], cfg
) -> int:
    # Reconstruir términos de alarma para el COUNT
    nets_list = _as_list(networks) or []
    flag_terms: List[str] = []
    thr_params_all: Dict[str, object] = {}
    for kpi in _ALARM_KPIS:
        if kpi not in COLMAP:
            continue
        col_sql = _quote(COLMAP[kpi])
        flag_thr_expr, _exc_thr_expr, p = _build_flag_and_excess_cases(kpi, nets_list, cfg)
        thr_params_all.update(p)
        flag_terms.append(f"({col_sql} >= {flag_thr_expr})")

    if not flag_terms:
        return _count_total(where_sql, params, uv, uc, un, ut)

    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql} AND ( { ' OR '.join(flag_terms) } )
    """
    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(count_sql, uv, uc, un, ut)
        return int(conn.execute(stmt, {**params, **thr_params_all}).scalar() or 0)


def _fetch_default(
    select_cols: List[str], where_sql: str, params: Dict,
    uv: bool, uc: bool, un: bool, ut: bool,
    *, page: Optional[int], page_size: Optional[int], limit: Optional[int]
) -> pd.DataFrame:
    sql = f"""
        SELECT {', '.join(select_cols)}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
        ORDER BY {_quote(COLMAP['fecha'])} DESC, {_quote(COLMAP['hora'])} DESC
    """
    exec_params = dict(params)

    if page and page_size:
        page = max(1, int(page))
        page_size = max(1, int(page_size))
        offset = (page - 1) * page_size
        sql += " LIMIT :_limit OFFSET :_offset"
        exec_params.update({"_limit": page_size, "_offset": offset})
    elif limit:
        sql += " LIMIT :_limit"
        exec_params.update({"_limit": int(limit)})

    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(sql, uv, uc, un, ut)
        return pd.read_sql(stmt, conn, params=exec_params)


def _fetch_metric_global(
    *, where_sql: str, base_params: Dict, uv: bool, uc: bool, un: bool, ut: bool,
    select_cols: List[str], sort_metric_friendly: Optional[str], sort_net: Optional[str], ascending: bool,
    page: int, page_size: int
) -> pd.DataFrame:
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    order_dir = "ASC" if ascending else "DESC"
    if sort_metric_friendly and sort_metric_friendly in COLMAP:
        real_metric = _quote(COLMAP[sort_metric_friendly])
    else:
        real_metric = _quote(COLMAP['fecha'])

    if sort_net:
        metric_expr = f"MAX(CASE WHEN {_quote(COLMAP['network'])} = :_sort_net THEN {real_metric} END)"
    else:
        metric_expr = f"MAX({real_metric})"

    key_cols_real = [COLMAP['fecha'], COLMAP['hora'], COLMAP['vendor'], COLMAP['noc_cluster'], COLMAP['technology']]
    key_cols_sel = ", ".join(_quote(c) for c in key_cols_real)
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
        stmt_keys = _prepare_stmt_with_expanding(keys_sql, uv, uc, un, ut)
        key_rows = conn.execute(stmt_keys, base_params_page).fetchall()

    if not key_rows:
        return pd.DataFrame(columns=BASE_COLUMNS)

    params_b: Dict[str, object] = {}
    or_parts: List[str] = []
    for i, (d, t, v, c, tech) in enumerate(key_rows, start=1):
        or_parts.append(
            f"({_quote(COLMAP['fecha'])} = :d{i} AND {_quote(COLMAP['hora'])} = :t{i} "
            f"AND {_quote(COLMAP['vendor'])} = :v{i} AND {_quote(COLMAP['noc_cluster'])} = :c{i} "
            f"AND {_quote(COLMAP['technology'])} = :tech{i})"
        )
        params_b.update({f"d{i}": d, f"t{i}": t, f"v{i}": v, f"c{i}": c, f"tech{i}": tech})

    sql_b = f"SELECT {', '.join(select_cols)} FROM {_quote_table(_TABLE_NAME)} WHERE {' OR '.join(or_parts)}"
    with eng.connect() as conn:
        return pd.read_sql(text(sql_b), conn, params=params_b)


def _fetch_alarm(
    *, where_sql: str, params: Dict, uv: bool, uc: bool, un: bool, ut: bool,
    select_cols: List[str], networks: Optional[List[str]], page: int, page_size: int
) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    cfg = load_threshold_cfg()
    nets_list = _as_list(networks) or []

    flag_terms: List[str] = []
    excess_terms: List[str] = []
    thr_params_all: Dict[str, object] = {}

    for kpi in _ALARM_KPIS:
        if kpi not in COLMAP:
            continue
        col_sql = _quote(COLMAP[kpi])
        flag_thr_expr, exc_thr_expr, p = _build_flag_and_excess_cases(kpi, nets_list, cfg)
        thr_params_all.update(p)
        flag_terms.append(f"({col_sql} >= {flag_thr_expr})")
        excess_terms.append(f"GREATEST(COALESCE({col_sql},0) - {exc_thr_expr}, 0)")

    if not flag_terms:
        # sin KPIs válidos, delega a default
        df = _fetch_default(select_cols, where_sql, params, uv, uc, un, ut, page=page, page_size=page_size, limit=None)
        return df, {}, []

    flags_sum = " + ".join(flag_terms)
    excess_sum = " + ".join(excess_terms)

    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    sel_sql = f"""
        SELECT {', '.join(select_cols)}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
          AND ( { ' OR '.join(flag_terms) } )
        ORDER BY
          ({flags_sum}) DESC,
          ({excess_sum}) DESC,
          {_quote(COLMAP['noc_cluster'])} ASC
        LIMIT :_limit OFFSET :_offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        sel_params = {**params, **thr_params_all, "_limit": page_size, "_offset": offset}
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, uv, uc, un, ut)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    return df, thr_params_all, flag_terms


# =========================================================
# FUNCIÓN PÚBLICA UNIFICADA
# =========================================================

def fetch_kpis_unified(
    *,
    # Filtros
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    # Paginación o límite
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    limit: Optional[int] = None,
    # Ordenamiento
    sort_mode: str = "default",          # "default" | "metric" | "alarm"
    sort_metric: Optional[str] = None,    # friendly
    sort_net: Optional[str] = None,
    ascending: bool = True,
    # Salida
    return_total: bool = False,
    na_as_empty: bool = False,
):
    """Punto único para recuperar KPIs con todas las variantes de orden y paginación."""
    # 1) Columnas amigables válidas
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    # 2) WHERE y params base
    where_sql, params, uv, uc, un, ut = _filters_where_and_params(
        fecha, hora, vendors, clusters, networks, technologies
    )

    # 3) Estrategia según sort_mode
    if sort_mode == "default":
        df = _fetch_default(select_cols, where_sql, params, uv, uc, un, ut, page=page, page_size=page_size, limit=limit)
        total = _count_total(where_sql, params, uv, uc, un, ut) if (return_total and page and page_size) else None
    elif sort_mode == "metric":
        # Para el orden global por métrica se requiere paginación (2 pasos)
        if not (page and page_size):
            page, page_size = 1, (limit or 100)
        df = _fetch_metric_global(
            where_sql=where_sql, base_params=params, uv=uv, uc=uc, un=un, ut=ut,
            select_cols=select_cols, sort_metric_friendly=sort_metric, sort_net=sort_net, ascending=ascending,
            page=page, page_size=page_size
        )
        total = _count_total(where_sql, params, uv, uc, un, ut) if return_total else None
    elif sort_mode == "alarm":
        if not (page and page_size):
            page, page_size = 1, (limit or 50)
        df, thr_params_all, flag_terms = _fetch_alarm(
            where_sql=where_sql, params=params, uv=uv, uc=uc, un=un, ut=ut,
            select_cols=select_cols, networks=networks, page=page, page_size=page_size
        )
        # COUNT especial: sólo filas con >=1 alarma
        if return_total:
            cfg = load_threshold_cfg()
            total = _count_total_alarm(
                where_sql=where_sql, params=params, uv=uv, uc=uc, un=un, ut=ut,
                networks=networks, cfg=cfg
            )
        else:
            total = None
    else:
        raise ValueError("sort_mode inválido. Usa 'default' | 'metric' | 'alarm'.")

    # 4) Orden final de columnas amigables + NaN → ""
    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])
    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return (df, int(total)) if return_total else df


# =========================================================
# WRAPPERS PÚBLICOS (compatibilidad con callbacks existentes)
# =========================================================

def fetch_kpis(
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    limit=None,
    na_as_empty: bool = False,
):
    """Consulta no paginada (compatibilidad)."""
    return fetch_kpis_unified(
        fecha=fecha, hora=hora,
        vendors=vendors, clusters=clusters, networks=networks, technologies=technologies,
        limit=limit, sort_mode="default", na_as_empty=na_as_empty,
    )


def fetch_kpis_paginated(
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page: int = 1,
    page_size: int = 50,
    na_as_empty: bool = False,
):
    """Consulta paginada (compatibilidad). Devuelve (df, total_rows)."""
    return fetch_kpis_unified(
        fecha=fecha, hora=hora,
        vendors=vendors, clusters=clusters, networks=networks, technologies=technologies,
        page=page, page_size=page_size,
        sort_mode="default", na_as_empty=na_as_empty, return_total=True,
    )


def fetch_kpis_paginated_global_sort(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page: int = 1,
    page_size: int = 50,
    sort_by_friendly: Optional[str] = None,
    sort_net: Optional[str] = None,
    ascending: bool = True,
    na_as_empty: bool = False,
):
    """Orden global por KPI (compatibilidad). Devuelve (df, total_rows)."""
    return fetch_kpis_unified(
        fecha=fecha, hora=hora,
        vendors=vendors, clusters=clusters, networks=networks, technologies=technologies,
        page=page, page_size=page_size,
        sort_mode="metric", sort_metric=sort_by_friendly, sort_net=sort_net, ascending=ascending,
        na_as_empty=na_as_empty, return_total=True,
    )


def fetch_kpis_paginated_alarm_sort(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page: int = 1,
    page_size: int = 50,
    na_as_empty: bool = False,
):
    """Orden por alarmas (compatibilidad). Devuelve (df, total_rows)."""
    return fetch_kpis_unified(
        fecha=fecha, hora=hora,
        vendors=vendors, clusters=clusters, networks=networks, technologies=technologies,
        page=page, page_size=page_size,
        sort_mode="alarm", na_as_empty=na_as_empty, return_total=True,
    )
