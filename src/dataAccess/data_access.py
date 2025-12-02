# src/data_access.py
from datetime import datetime, timedelta

import pandas as pd
from functools import lru_cache
from sqlalchemy import create_engine, text, bindparam
from typing import Dict, Tuple, Optional, List

from src.Utils.alarmados import alarm_threshold_for, load_threshold_cfg, excess_base_for
from src.config import SQLALCHEMY_URL

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

_SEVERITY_KPIS = [
    "ps_rrc_ia_percent",
    "ps_rab_ia_percent",
    "ps_s1_ia_percent",
    "ps_drop_dc_percent",
    "cs_rrc_ia_percent",
    "cs_rab_ia_percent",
    "cs_drop_dc_percent",
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

def _build_severity_expressions_from_json(
    cfg=None,
    profile: str = "main",
):
    """
    Construye:
      - severity_sum_expr: expresión SQL que suma las severidades (0..4) de todos los KPIs
      - crit_count_expr: expresión SQL que cuenta cuántos KPIs están en nivel 'critico'
      - params: dict de parámetros para las expresiones (umbrales)
    Usa profiles[profile].severity del JSON.
    """
    if cfg is None:
        cfg = load_threshold_cfg()  # debe devolver el JSON completo que mostraste

    prof = (cfg.get("profiles") or {}).get(profile) or {}
    sev_cfg = prof.get("severity") or {}

    sev_terms = []
    crit_terms = []
    params = {}

    for kpi in _SEVERITY_KPIS:
        if kpi not in COLMAP:
            continue

        kcfg = sev_cfg.get(kpi) or {}
        # soporta estructuras:
        # - { "orientation": ..., "thresholds": {...} }
        # - { "default": {...}, "per_network": {...} }
        base = (kcfg.get("default") or kcfg)
        thresholds = (base.get("thresholds") or {})

        # Umbrales; caen a 0 si falta alguno
        exc = float(thresholds.get("excelente", 0.0))
        bue = float(thresholds.get("bueno", exc))
        reg = float(thresholds.get("regular", bue))
        cri = float(thresholds.get("critico", reg))

        # De momento asumimos orientation = lower_is_better
        # (es lo que tienes en el JSON)
        col_sql = _quote(COLMAP[kpi])
        num_col = f"CAST({col_sql} AS DECIMAL(20,6))"

        p_prefix = kpi  # p.ej. "ps_rrc_ia_percent"
        params[f"{p_prefix}_exc"] = exc
        params[f"{p_prefix}_bue"] = bue
        params[f"{p_prefix}_reg"] = reg
        params[f"{p_prefix}_cri"] = cri

        sev_expr = (
            f"CASE "
            f"WHEN COALESCE({num_col}, 0) >= :{p_prefix}_cri THEN 4 "
            f"WHEN COALESCE({num_col}, 0) >= :{p_prefix}_reg THEN 3 "
            f"WHEN COALESCE({num_col}, 0) >= :{p_prefix}_bue THEN 2 "
            f"WHEN COALESCE({num_col}, 0) >= :{p_prefix}_exc THEN 1 "
            f"ELSE 0 END"
        )

        crit_expr = (
            f"CASE WHEN COALESCE({num_col}, 0) >= :{p_prefix}_cri "
            f"THEN 1 ELSE 0 END"
        )

        sev_terms.append(sev_expr)
        crit_terms.append(crit_expr)

    if not sev_terms:
        # si nada mapea, regresamos expresiones neutras
        return "0", "0", {}

    severity_sum_expr = " + ".join(sev_terms)
    crit_count_expr = " + ".join(crit_terms)
    return severity_sum_expr, crit_count_expr, params

def _build_severity_expr_from_json(profile: str = "main"):
    """
    Construye una expresión SQL de severidad tipo:

        severity_score = SUM_kpi( severity_kpi )

    donde cada severidad de KPI viene de profiles[profile].severity en el JSON
    de umbrales (data/umbrales.json).

    Para cada KPI:
      - Usa 'orientation' (lower_is_better / higher_is_better).
      - Usa thresholds.excelente/bueno/regular/critico.
      - Soporta valores por red en per_network.

    Devuelve: (severity_expr_sql, params_dict)
    """
    cfg = load_threshold_cfg()  # JSON completo
    profiles = cfg.get("profiles") or {}
    prof = profiles.get(profile) or {}
    sev_cfg = prof.get("severity") or {}

    params = {}
    kpi_terms = []

    def _build_case(num_col: str, prefix: str, orientation: str) -> str:
        """
        Devuelve el CASE de severidad para un KPI y un set de thresholds ligado
        al prefijo de parámetros (prefix_*).
        """
        # lower_is_better: valores altos son malos
        if orientation == "higher_is_better":
            # invertido: valores bajos son malos
            return (
                f"CASE "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_cri THEN 4 "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_reg THEN 3 "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_bue THEN 2 "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_exc THEN 1 "
                f"ELSE 0 END"
            )
        else:
            # default: lower_is_better → valores altos son peores
            return (
                f"CASE "
                f"WHEN COALESCE({num_col}, 0) >= :{prefix}_cri THEN 4 "
                f"WHEN COALESCE({num_col}, 0) >= :{prefix}_reg THEN 3 "
                f"WHEN COALESCE({num_col}, 0) >= :{prefix}_bue THEN 2 "
                f"WHEN COALESCE({num_col}, 0) >= :{prefix}_exc THEN 1 "
                f"ELSE 0 END"
            )

    for kpi in _SEVERITY_KPIS:
        if kpi not in COLMAP:
            continue

        kcfg = sev_cfg.get(kpi) or {}
        # puede venir como:
        #   { "orientation":.., "thresholds":.., "per_network":.. }
        # o como:
        #   { "default": {...}, "per_network": {...} }
        default_block = (kcfg.get("default") or kcfg) or {}
        per_net_block = kcfg.get("per_network") or {}

        thresholds_def = (default_block.get("thresholds") or {})
        orientation_def = default_block.get("orientation", "lower_is_better")

        # thresholds default
        def_exc = float(thresholds_def.get("excelente", 0.0))
        def_bue = float(thresholds_def.get("bueno", def_exc))
        def_reg = float(thresholds_def.get("regular", def_bue))
        def_cri = float(thresholds_def.get("critico", def_reg))

        pfx_def = f"{kpi}_def"
        params[f"{pfx_def}_exc"] = def_exc
        params[f"{pfx_def}_bue"] = def_bue
        params[f"{pfx_def}_reg"] = def_reg
        params[f"{pfx_def}_cri"] = def_cri

        col_sql = _quote(COLMAP[kpi])
        num_col = f"CAST({col_sql} AS DECIMAL(20,6))"

        case_default = _build_case(num_col, pfx_def, orientation_def)

        # Si hay per_network, armamos un CASE grande por red
        if per_net_block:
            when_parts = []

            for idx, (net_name, net_cfg) in enumerate(per_net_block.items()):
                thresholds_net = (net_cfg.get("thresholds") or thresholds_def) or {}
                orientation_net = net_cfg.get("orientation", orientation_def)

                net_exc = float(thresholds_net.get("excelente", def_exc))
                net_bue = float(thresholds_net.get("bueno", def_bue))
                net_reg = float(thresholds_net.get("regular", def_reg))
                net_cri = float(thresholds_net.get("critico", def_cri))

                pfx_net = f"{kpi}_net{idx}"
                params[f"{pfx_net}_exc"] = net_exc
                params[f"{pfx_net}_bue"] = net_bue
                params[f"{pfx_net}_reg"] = net_reg
                params[f"{pfx_net}_cri"] = net_cri
                params[f"{pfx_net}_name"] = net_name

                case_net = _build_case(num_col, pfx_net, orientation_net)

                when_parts.append(
                    f"WHEN {_quote(COLMAP['network'])} = :{pfx_net}_name THEN ({case_net})"
                )

            kpi_expr = f"(CASE {' '.join(when_parts)} ELSE ({case_default}) END)"
        else:
            # sin per_network, usamos sólo el default
            kpi_expr = f"({case_default})"

        kpi_terms.append(kpi_expr)

    if not kpi_terms:
        return "0", {}

    severity_expr = " + ".join(kpi_terms)
    return severity_expr, params

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


def fetch_kpis_paginated_severity_global_sort(
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
    """
    GLOBAL basado en umbrales del JSON (profiles.main.severity):

      - NO descarta registros (incluye todos los que cumplan el WHERE).

      MODO POR DEFECTO (sin columna seleccionada):
        ORDER BY
            severity_score DESC,
            Date DESC,
            Time DESC

      MODO CON COLUMNA (sort_by_friendly válido):
        ORDER BY
            (metric_expr IS NULL) ASC,   -- NULLS LAST
            metric_expr {ASC|DESC},      -- criterio principal
            severity_score DESC,         -- desempate
            Date DESC,
            Time DESC
    """
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    # WHERE base + params
    where_sql, base_params, uv, uc, un, ut = _filters_where_and_params(
        fecha, hora, vendors, clusters, networks, technologies
    )

    # Expresión de severidad desde el JSON
    severity_expr, thr_params = _build_severity_expr_from_json(profile="main")

    # COUNT total
    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    # columnas amigables
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    order_dir = "ASC" if ascending else "DESC"

    # -------- ¿hay columna seleccionada? --------
    has_custom_sort = sort_by_friendly and (sort_by_friendly in COLMAP)

    if has_custom_sort:
        # métrica real a ordenar
        real_metric = _quote(COLMAP[sort_by_friendly])

        if sort_net:
            metric_expr = (
                f"CASE WHEN {_quote(COLMAP['network'])} = :_sort_net "
                f"THEN {real_metric} ELSE {real_metric} END"
            )
            base_params["_sort_net"] = sort_net
        else:
            metric_expr = real_metric

        nulls_last_expr = f"({metric_expr} IS NULL)"

        # MODO COLUMNA: la métrica es el criterio principal
        order_clause = f"""
            ORDER BY
                {nulls_last_expr} ASC,          -- primero no-nulos
                {metric_expr} {order_dir},      -- métrica clickeada
                severity_score DESC,            -- desempate global
                {_quote(COLMAP['fecha'])} DESC,
                {_quote(COLMAP['hora'])} DESC
        """
    else:
        # MODO POR DEFECTO: EXACTO y estable, sin depender de ascending
        order_clause = f"""
            ORDER BY
                severity_score DESC,
                {_quote(COLMAP['fecha'])} DESC,
                {_quote(COLMAP['hora'])} DESC
        """

    sel_sql = f"""
        SELECT
            {", ".join(select_cols)},
            ({severity_expr}) AS severity_score
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
        {order_clause}
        LIMIT :limit OFFSET :offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        # COUNT
        stmt_count = _prepare_stmt_with_expanding(count_sql, uv, uc, un, ut)
        total = conn.execute(
            stmt_count,
            {**base_params, **thr_params}
        ).scalar() or 0

        # PAGE
        sel_params = {
            **base_params,
            **thr_params,
            "limit": page_size,
            "offset": offset,
        }
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, uv, uc, un, ut)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    # Orden de columnas amigables
    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])

    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df, int(total)



def fetch_kpis_paginated_severity_sort(
    *,
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
    Ordena por severidad tipo:
      - severity_score (suma de 0..4 por KPI usando thresholds de profiles.main.severity)
      - crit_count (cuántos KPIs están en 'critico')
    Incluye sólo filas con al menos 1 KPI en 'critico' (crit_count > 0),
    replicando el comportamiento de tu query original.
    """
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    # WHERE base y params (sin tocar JSON todavía)
    where_sql, params, uv, uc, un, ut = _filters_where_and_params(
        fecha, hora, vendors, clusters, networks, technologies
    )

    # Construye expresiones de severidad a partir del JSON
    cfg = load_threshold_cfg()  # data/umbrales.json
    severity_expr, crit_expr, thr_params = _build_severity_expressions_from_json(cfg, profile="main")

    # COUNT con al menos un KPI en nivel 'critico'
    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
          AND ( {crit_expr} ) > 0
    """

    # SELECT paginado
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    sel_sql = f"""
        SELECT
            {", ".join(select_cols)},
            ({severity_expr}) AS severity_score,
            ({crit_expr})      AS crit_count
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
          AND ( {crit_expr} ) > 0
        ORDER BY
            severity_score DESC,
            crit_count     DESC,
            {_quote(COLMAP['noc_cluster'])} ASC
        LIMIT :_limit OFFSET :_offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        # COUNT
        stmt_count = _prepare_stmt_with_expanding(count_sql, uv, uc, un, ut)
        total = conn.execute(
            stmt_count,
            {**params, **thr_params}
        ).scalar() or 0

        # PAGE
        sel_params = {**params, **thr_params, "_limit": page_size, "_offset": offset}
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
    alarm_kpis = [k for k in _SEVERITY_KPIS if k in COLMAP]

    if not alarm_kpis:
        return pd.DataFrame(columns=["technology","vendor","noc_cluster"]), set()

    # Construye expresiones CASE por KPI y parámetros de umbrales
    flag_cols_sql = []
    thr_params_all = {}

    # Bloques de severidad desde el JSON
    profiles = (cfg.get("profiles") or {})
    prof_main = profiles.get("main") or {}
    sev_cfg = prof_main.get("severity") or {}

    # Construye expresión CASE que vale 1 si el KPI está en nivel "crítico" según severity
    def _flag_expr_for(kpi: str) -> str:
        col_sql = _quote(COLMAP[kpi])
        num_col = f"CAST({col_sql} AS DECIMAL(20,6))"

        kcfg = sev_cfg.get(kpi) or {}
        # puede venir como:
        #   { "orientation":..., "thresholds":..., "per_network":... }
        # o como:
        #   { "default": {...}, "per_network": {...} }
        default_block = (kcfg.get("default") or kcfg) or {}
        per_net_block = kcfg.get("per_network") or {}

        thresholds_def = (default_block.get("thresholds") or {})
        orientation_def = default_block.get("orientation", "lower_is_better")

        # Umbral crítico default
        def_cri = float(thresholds_def.get("critico", 0.0))
        p_def = f"{kpi}_def"
        thr_params_all[f"{p_def}_cri"] = def_cri

        # Condición de alarma para el default
        if orientation_def == "higher_is_better":
            # valores bajos son peores -> alarma si valor <= crítico
            def_cond = f"COALESCE({num_col}, 0) <= :{p_def}_cri"
        else:
            # lower_is_better (tu caso) -> valores altos son peores -> alarma si valor >= crítico
            def_cond = f"COALESCE({num_col}, 0) >= :{p_def}_cri"

        # Si hay per_network, armamos CASE por red usando alias "network"
        if per_net_block:
            when_parts = []
            for idx, (net_name, net_cfg) in enumerate(per_net_block.items()):
                thresholds_net = (net_cfg.get("thresholds") or thresholds_def) or {}
                orientation_net = net_cfg.get("orientation", orientation_def)

                net_cri = float(thresholds_net.get("critico", def_cri))
                p_net = f"{kpi}_net{idx}"
                thr_params_all[f"{p_net}_cri"] = net_cri
                thr_params_all[f"{p_net}_name"] = net_name

                if orientation_net == "higher_is_better":
                    cond_net = f"COALESCE({num_col}, 0) <= :{p_net}_cri"
                else:
                    cond_net = f"COALESCE({num_col}, 0) >= :{p_net}_cri"

                # OJO: aquí comparamos contra el alias 'network', no contra `Network`
                when_parts.append(
                    f"WHEN network = :{p_net}_name AND {cond_net} THEN 1"
                )

            # Si ninguna red coincide, cae al default
            return f"(CASE {' '.join(when_parts)} ELSE (CASE WHEN {def_cond} THEN 1 ELSE 0 END) END)"
        else:
            # sin per_network, sólo default
            return f"(CASE WHEN {def_cond} THEN 1 ELSE 0 END)"

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


