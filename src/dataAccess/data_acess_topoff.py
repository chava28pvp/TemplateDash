# src/dataAccess/data_acess_topoff.py
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text, bindparam

from src.Utils.alarmados import load_threshold_cfg
from src.config import SQLALCHEMY_URL

# =========================================================
# Engine / Config
# =========================================================
_engine = None
_TABLE_NAME = "dashboard_topoff"


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True, pool_recycle=1800)
    return _engine


# =========================================================
# Column mapping (friendly -> real DB column)
# =========================================================
COLMAP = {
    "id": "ID",
    "tech": "Tech",
    "technology": "Technology",
    "vendor": "Vendor",
    "region": "REGION",
    "province": "PROVINCE",
    "municipality": "MUNICIPALITY",
    "fecha": "DATE",
    "hora": "TIME",
    "site_att": "SITE_ATT",
    "rnc": "RNC",
    "nodeb": "NODEB",

    # PS
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
    "cs_traff_erl": "CS_TRAFF_ERL",
    "cs_rrc_ia_percent": "CS_RRC_%IA",
    "cs_rrc_fail": "CS_RRC_FAIL",
    "cs_rab_ia_percent": "CS_RAB_%IA",
    "cs_rab_fail": "CS_RAB_FAIL",
    "cs_drop_dc_percent": "CS_DROP_%DC",
    "cs_drop_abnrel": "CS_DROP_ABNREL",

    # TNL / Unav
    "unav": "Unav",
    "rtx_tnl_tx_percent": "3G_RTX/4G_TNL_%Tx",
    "tnl_abn": "TNL_ABN",
    "tnl_fail": "TNL_FAIL",

    # metadatos
    "archivo_fuente": "Archivo_Fuente",
    "fecha_ejecucion": "Fecha_Ejecucion",
}

BASE_COLUMNS = [
    "fecha", "hora", "technology", "vendor", "region", "province",
    "municipality", "site_att", "rnc", "nodeb",
    "ps_traff_gb", "ps_rrc_ia_percent", "ps_rrc_fail",
    "ps_rab_ia_percent", "ps_rab_fail",
    "ps_s1_ia_percent", "ps_s1_fail",
    "ps_drop_dc_percent", "ps_drop_abnrel",
    "cs_traff_erl", "cs_rrc_ia_percent", "cs_rrc_fail",
    "cs_rab_ia_percent", "cs_rab_fail",
    "cs_drop_dc_percent", "cs_drop_abnrel",
    "unav", "rtx_tnl_tx_percent", "tnl_abn", "tnl_fail",
    "archivo_fuente", "fecha_ejecucion",
]

_MIN_SAFE_COLUMNS = [
    "fecha", "hora", "technology", "vendor", "region", "province", "municipality",
]

_SEVERITY_KPIS_TOPOFF = [
    "ps_rrc_ia_percent",
    "ps_rab_ia_percent",
    "ps_s1_ia_percent",
    "ps_drop_dc_percent",
    "cs_rrc_ia_percent",
    "cs_rab_ia_percent",
    "cs_drop_dc_percent",
    "rtx_tnl_tx_percent",
]


# =========================================================
# Helpers internos
# =========================================================
def _quote(colname: str) -> str:
    return f"`{colname}`"


def _quote_table(name: str) -> str:
    return f"`{name}`"


def _prepare_stmt_with_expanding(
    sql,
    use_regions=False,
    use_provinces=False,
    use_muns=False,
    use_technologies=False,
    use_vendors=False,
    use_fechas=False,
    use_sites=False,
    use_rncs=False,
    use_nodebs=False,
):
    stmt = text(sql)
    if use_regions:
        stmt = stmt.bindparams(bindparam("regions", expanding=True))
    if use_provinces:
        stmt = stmt.bindparams(bindparam("provinces", expanding=True))
    if use_muns:
        stmt = stmt.bindparams(bindparam("muns", expanding=True))
    if use_technologies:
        stmt = stmt.bindparams(bindparam("technologies", expanding=True))
    if use_vendors:
        stmt = stmt.bindparams(bindparam("vendors", expanding=True))
    if use_fechas:
        stmt = stmt.bindparams(bindparam("fechas", expanding=True))
    if use_sites:
        stmt = stmt.bindparams(bindparam("sites", expanding=True))
    if use_rncs:
        stmt = stmt.bindparams(bindparam("rncs", expanding=True))
    if use_nodebs:
        stmt = stmt.bindparams(bindparam("nodebs", expanding=True))
    return stmt


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [v for v in x if v not in (None, "")]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [x]


def _fecha_with_prev(fecha: Optional[str]) -> List[str]:
    """Devuelve [fecha, fecha-1día] si fecha es válida YYYY-MM-DD."""
    if not fecha:
        return []
    try:
        d = datetime.strptime(fecha, "%Y-%m-%d").date()
        prev = (d - timedelta(days=1)).strftime("%Y-%m-%d")
        return [fecha, prev]
    except Exception:
        return [fecha]


@lru_cache(maxsize=1)
def _existing_columns():
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
    fallback = [c for c in _MIN_SAFE_COLUMNS if COLMAP.get(c) in existing_real]
    return fallback


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
    fecha: Optional[str] = None,
    fechas: Optional[List[str]] = None,
    hora: Optional[str] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    technologies: Optional[List[str]] = None,
    vendors: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    rncs: Optional[List[str]] = None,
    nodebs: Optional[List[str]] = None,
):
    where = ["1=1"]
    params: Dict[str, object] = {}

    # fechas
    use_fechas = False
    if fechas:
        where.append(f"{_quote(COLMAP['fecha'])} IN :fechas")
        params["fechas"] = fechas
        use_fechas = True
    elif fecha:
        where.append(f"{_quote(COLMAP['fecha'])} = :fecha")
        params["fecha"] = fecha

    # hora (TopOff no la usa, pero dejamos helper genérico)
    if hora and str(hora).lower() != "todas":
        where.append(f"{_quote(COLMAP['hora'])} = :hora")
        params["hora"] = hora

    regions = _as_list(regions)
    provinces = _as_list(provinces)
    municipalities = _as_list(municipalities)
    technologies = _as_list(technologies)
    vendors = _as_list(vendors)
    sites = _as_list(sites)
    rncs = _as_list(rncs)
    nodebs = _as_list(nodebs)

    use_regions = use_provinces = use_muns = use_technologies = use_vendors = False
    use_sites = use_rncs = use_nodebs = False

    if regions:
        where.append(f"{_quote(COLMAP['region'])} IN :regions")
        params["regions"] = regions
        use_regions = True
    if provinces:
        where.append(f"{_quote(COLMAP['province'])} IN :provinces")
        params["provinces"] = provinces
        use_provinces = True
    if municipalities:
        where.append(f"{_quote(COLMAP['municipality'])} IN :muns")
        params["muns"] = municipalities
        use_muns = True
    if technologies:
        where.append(f"{_quote(COLMAP['technology'])} IN :technologies")
        params["technologies"] = technologies
        use_technologies = True
    if vendors:
        where.append(f"{_quote(COLMAP['vendor'])} IN :vendors")
        params["vendors"] = vendors
        use_vendors = True

    # nuevos filtros exclusivos topoff
    if sites:
        where.append(f"{_quote(COLMAP['site_att'])} IN :sites")
        params["sites"] = sites
        use_sites = True
    if rncs:
        where.append(f"{_quote(COLMAP['rnc'])} IN :rncs")
        params["rncs"] = rncs
        use_rncs = True
    if nodebs:
        where.append(f"{_quote(COLMAP['nodeb'])} IN :nodebs")
        params["nodebs"] = nodebs
        use_nodebs = True

    return (
        " AND ".join(where),
        params,
        use_regions,
        use_provinces,
        use_muns,
        use_technologies,
        use_vendors,
        use_fechas,
        use_sites,
        use_rncs,
        use_nodebs,
    )


def _build_severity_expr_from_json_topoff(profile: str = "topoff"):
    """
    severity_score = SUM_kpi(severity_kpi) usando umbrales JSON.
    """
    cfg = load_threshold_cfg()
    profiles = cfg.get("profiles") or {}
    prof = profiles.get(profile) or profiles.get("main") or {}
    sev_cfg = prof.get("severity") or {}

    params: Dict[str, float] = {}
    kpi_terms: List[str] = []

    def _build_case(num_col: str, prefix: str, orientation: str) -> str:
        if orientation == "higher_is_better":
            return (
                f"CASE "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_cri THEN 4 "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_reg THEN 3 "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_bue THEN 2 "
                f"WHEN COALESCE({num_col}, 0) <= :{prefix}_exc THEN 1 "
                f"ELSE 0 END"
            )
        return (
            f"CASE "
            f"WHEN COALESCE({num_col}, 0) >= :{prefix}_cri THEN 4 "
            f"WHEN COALESCE({num_col}, 0) >= :{prefix}_reg THEN 3 "
            f"WHEN COALESCE({num_col}, 0) >= :{prefix}_bue THEN 2 "
            f"WHEN COALESCE({num_col}, 0) >= :{prefix}_exc THEN 1 "
            f"ELSE 0 END"
        )

    for kpi in _SEVERITY_KPIS_TOPOFF:
        if kpi not in COLMAP:
            continue

        kcfg = sev_cfg.get(kpi) or {}
        default_block = (kcfg.get("default") or kcfg) or {}
        thresholds_def = (default_block.get("thresholds") or {})
        orientation_def = default_block.get("orientation", "lower_is_better")

        def_exc = float(thresholds_def.get("excelente", 0.0))
        def_bue = float(thresholds_def.get("bueno", def_exc))
        def_reg = float(thresholds_def.get("regular", def_bue))
        def_cri = float(thresholds_def.get("critico", def_reg))

        pfx = f"{kpi}_def"
        params[f"{pfx}_exc"] = def_exc
        params[f"{pfx}_bue"] = def_bue
        params[f"{pfx}_reg"] = def_reg
        params[f"{pfx}_cri"] = def_cri

        col_sql = _quote(COLMAP[kpi])
        num_col = f"CAST({col_sql} AS DECIMAL(20,6))"
        case_default = _build_case(num_col, pfx, orientation_def)
        kpi_terms.append(f"({case_default})")

    if not kpi_terms:
        return "0", {}

    severity_expr = " + ".join(kpi_terms)
    return severity_expr, params


# =========================================================
# API pública
# =========================================================
def fetch_topoff_paginated(
    *,
    fecha: Optional[str] = None,
    technologies: Optional[List[str]] = None,
    vendors: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    rncs: Optional[List[str]] = None,
    nodebs: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 50,
    na_as_empty: bool = False,
) -> Tuple[pd.DataFrame, int]:
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    fechas = _fecha_with_prev(fecha)
    where_sql, params, ur, up, um, utech, uvend, uf, us, urc, unb = _filters_where_and_params(
        fecha=None,
        fechas=fechas,
        hora=None,
        regions=regions,
        provinces=provinces,
        municipalities=municipalities,
        technologies=technologies,
        vendors=vendors,
        sites=sites,
        rncs=rncs,
        nodebs=nodebs,
    )

    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

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
        stmt_count = _prepare_stmt_with_expanding(count_sql, ur, up, um, utech, uvend, uf, us, urc, unb)
        total = conn.execute(stmt_count, params).scalar() or 0

        sel_params = dict(params)
        sel_params.update({"limit": page_size, "offset": offset})
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, ur, up, um, utech, uvend, uf, us, urc, unb)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])
    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df, int(total)


def fetch_topoff_paginated_global_sort(
    *,
    fecha: Optional[str] = None,
    technologies: Optional[List[str]] = None,
    vendors: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    rncs: Optional[List[str]] = None,
    nodebs: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 50,
    sort_by_friendly: Optional[str] = None,
    ascending: bool = True,
    na_as_empty: bool = False,
) -> Tuple[pd.DataFrame, int]:
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    fechas = _fecha_with_prev(fecha)
    where_sql, params, ur, up, um, utech, uvend, uf, us, urc, unb = _filters_where_and_params(
        fecha=None,
        fechas=fechas,
        hora=None,
        regions=regions,
        provinces=provinces,
        municipalities=municipalities,
        technologies=technologies,
        vendors=vendors,
        sites=sites,
        rncs=rncs,
        nodebs=nodebs,
    )

    NUMERIC_COLS = {
        "ps_traff_gb", "ps_rrc_ia_percent", "ps_rrc_fail",
        "ps_rab_ia_percent", "ps_rab_fail",
        "ps_s1_ia_percent", "ps_s1_fail",
        "ps_drop_dc_percent", "ps_drop_abnrel",
        "cs_traff_erl", "cs_rrc_ia_percent", "cs_rrc_fail",
        "cs_rab_ia_percent", "cs_rab_fail",
        "cs_drop_dc_percent", "cs_drop_abnrel",
        "unav", "rtx_tnl_tx_percent", "tnl_abn", "tnl_fail",
    }

    if sort_by_friendly:
        sort_by_friendly = sort_by_friendly or None

    if sort_by_friendly and sort_by_friendly in COLMAP:
        real = COLMAP[sort_by_friendly]
        direction = "ASC" if ascending else "DESC"
        if sort_by_friendly in NUMERIC_COLS:
            real_expr = f"CAST(NULLIF({_quote(real)}, '') AS DECIMAL(18,6))"
        else:
            real_expr = f"NULLIF({_quote(real)}, '')"
        order_by = (
            f"({real_expr} IS NULL) ASC, "
            f"{real_expr} {direction}, "
            f"{_quote(COLMAP['fecha'])} DESC, "
            f"{_quote(COLMAP['hora'])} DESC"
        )
    else:
        order_by = f"{_quote(COLMAP['fecha'])} DESC, {_quote(COLMAP['hora'])} DESC"

    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    sel_sql = f"""
        SELECT {", ".join(select_cols)}
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        stmt_count = _prepare_stmt_with_expanding(count_sql, ur, up, um, utech, uvend, uf, us, urc, unb)
        total = conn.execute(stmt_count, params).scalar() or 0

        sel_params = dict(params)
        sel_params.update({"limit": page_size, "offset": offset})
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, ur, up, um, utech, uvend, uf, us, urc, unb)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])
    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df, int(total)


def fetch_topoff_paginated_severity_global_sort(
    *,
    fecha: Optional[str] = None,
    technologies: Optional[List[str]] = None,
    vendors: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    rncs: Optional[List[str]] = None,
    nodebs: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 50,
    sort_by_friendly: Optional[str] = None,
    ascending: bool = True,
    na_as_empty: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Orden global tipo "top offenders" por severidad:
      1) severity_score DESC
      2) luego sort_by_friendly (si viene)
      3) luego fecha/hora DESC
    """
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size

    fechas = _fecha_with_prev(fecha)
    where_sql, params, ur, up, um, utech, uvend, uf, us, urc, unb = _filters_where_and_params(
        fecha=None,
        fechas=fechas,
        hora=None,
        regions=regions,
        provinces=provinces,
        municipalities=municipalities,
        technologies=technologies,
        vendors=vendors,
        sites=sites,
        rncs=rncs,
        nodebs=nodebs,
    )

    severity_expr, thr_params = _build_severity_expr_from_json_topoff(profile="topoff")

    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    if sort_by_friendly and sort_by_friendly in COLMAP:
        order_dir = "ASC" if ascending else "DESC"
        metric_expr = _quote(COLMAP[sort_by_friendly])
        secondary_order_sql = f", {metric_expr} {order_dir}"
    else:
        secondary_order_sql = ""

    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    sel_sql = f"""
        SELECT
            {", ".join(select_cols)},
            ({severity_expr}) AS severity_score
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
        ORDER BY
            severity_score DESC
            {secondary_order_sql},
            {_quote(COLMAP['fecha'])} DESC,
            {_quote(COLMAP['hora'])} DESC
        LIMIT :limit OFFSET :offset
    """

    eng = get_engine()
    with eng.connect() as conn:
        stmt_count = _prepare_stmt_with_expanding(count_sql, ur, up, um, utech, uvend, uf, us, urc, unb)
        total = conn.execute(stmt_count, {**params, **thr_params}).scalar() or 0

        sel_params = {**params, **thr_params, "limit": page_size, "offset": offset}
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, ur, up, um, utech, uvend, uf, us, urc, unb)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])
    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df, int(total)


def fetch_topoff_distinct_options(
    *,
    fecha: Optional[str] = None,
    technologies: Optional[List[str]] = None,
    vendors: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
):
    """
    Devuelve listas distintas para poblar Site/RNC/NodeB según filtros base.
    """
    fechas = _fecha_with_prev(fecha)

    where_sql, params, ur, up, um, utech, uvend, uf, us, urc, unb = _filters_where_and_params(
        fecha=None,
        fechas=fechas,
        hora=None,
        regions=regions,
        provinces=provinces,
        municipalities=municipalities,
        technologies=technologies,
        vendors=vendors,
    )

    sql = f"""
        SELECT DISTINCT
            {_quote(COLMAP['site_att'])} AS site_att,
            {_quote(COLMAP['rnc'])} AS rnc,
            {_quote(COLMAP['nodeb'])} AS nodeb
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(sql, ur, up, um, utech, uvend, uf, us, urc, unb)
        df = pd.read_sql(stmt, conn, params=params)

    sites = sorted([x for x in df.get("site_att", pd.Series()).dropna().unique().tolist() if str(x).strip()])
    rncs = sorted([x for x in df.get("rnc", pd.Series()).dropna().unique().tolist() if str(x).strip()])
    nodebs = sorted([x for x in df.get("nodeb", pd.Series()).dropna().unique().tolist() if str(x).strip()])

    return sites, rncs, nodebs

def fetch_topoff_distinct(
    *,
    regions=None,
    provinces=None,
    municipalities=None,
    technologies=None,
    vendors=None,
    fecha: Optional[str] = None,
):
    fechas = _fecha_with_prev(fecha)

    where_sql, params, ur, up, um, utech, uvend, uf, *rest = _filters_where_and_params(
        fecha=fecha,
        fechas=fechas,
        hora=None,
        regions=regions,
        provinces=provinces,
        municipalities=municipalities,
        technologies=technologies,
        vendors=vendors,
    )

    sql = f"""
        SELECT DISTINCT
            {_quote(COLMAP['fecha'])} AS fecha,
            {_quote(COLMAP['technology'])} AS technology,
            {_quote(COLMAP['vendor'])} AS vendor
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(
            sql,
            use_regions=ur,
            use_provinces=up,
            use_muns=um,
            use_technologies=utech,
            use_vendors=uvend
        )
        df = pd.read_sql(stmt, conn, params=params)

    fechas = sorted([str(x) for x in df["fecha"].dropna().unique().tolist()], reverse=True)
    techs  = sorted([str(x) for x in df["technology"].dropna().unique().tolist()])
    vends  = sorted([str(x) for x in df["vendor"].dropna().unique().tolist()])

    return {
        "fechas": fechas,
        "technologies": techs,
        "vendors": vends,
    }

