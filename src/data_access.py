# src/data_access.py
import pandas as pd
from functools import lru_cache
from sqlalchemy import create_engine, text, bindparam
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
