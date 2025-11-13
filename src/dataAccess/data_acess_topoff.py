# src/topoff_data_access.py
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text, bindparam

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
# OJO: nombres con % o / requieren backticks
# =========================================================
COLMAP = {
    # Identificadores / claves de negocio
    "id": "ID",
    "tech": "Tech",
    "region": "REGION",
    "province": "PROVINCE",
    "municipality": "MUNICIPALITY",
    "fecha": "DATE",
    "hora": "TIME",

    # Info de sitio
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

    # indisponibilidad / TNL
    "unav": "Unav",
    "rtx_tnl_tx_percent": "3G_RTX/4G_TNL_%Tx",
    "tnl_abn": "TNL_ABN",
    "tnl_fail": "TNL_FAIL",

    # metadatos
    "archivo_fuente": "Archivo_Fuente",
    "fecha_ejecucion": "Fecha_Ejecucion",
}

BASE_COLUMNS = [
    "fecha", "hora", "tech", "region", "province", "municipality",
    "site_att", "rnc", "nodeb",
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

_MIN_SAFE_COLUMNS = ["fecha", "hora", "tech", "region", "province", "municipality"]

# =========================================================
# Helpers internos
# =========================================================
def _quote(colname: str) -> str:
    return f"`{colname}`"

def _quote_table(name: str) -> str:
    return f"`{name}`"

def _prepare_stmt_with_expanding(sql, use_regions=False, use_provinces=False, use_muns=False, use_techs=False):
    stmt = text(sql)
    if use_regions:
        stmt = stmt.bindparams(bindparam("regions", expanding=True))
    if use_provinces:
        stmt = stmt.bindparams(bindparam("provinces", expanding=True))
    if use_muns:
        stmt = stmt.bindparams(bindparam("muns", expanding=True))
    if use_techs:
        stmt = stmt.bindparams(bindparam("techs", expanding=True))
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

    # fallback seguro mínimo
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
    hora: Optional[str] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    techs: Optional[List[str]] = None,
):
    where = ["1=1"]
    params: Dict[str, object] = {}

    if fecha:
        where.append(f"{_quote(COLMAP['fecha'])} = :fecha")
        params["fecha"] = fecha

    if hora and str(hora).lower() != "todas":
        where.append(f"{_quote(COLMAP['hora'])} = :hora")
        params["hora"] = hora

    regions = _as_list(regions)
    provinces = _as_list(provinces)
    municipalities = _as_list(municipalities)
    techs = _as_list(techs)

    use_regions = use_provinces = use_muns = use_techs = False

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
    if techs:
        where.append(f"{_quote(COLMAP['tech'])} IN :techs")
        params["techs"] = techs
        use_techs = True

    return " AND ".join(where), params, use_regions, use_provinces, use_muns, use_techs

# =========================================================
# API pública (ejemplo mínimo)
# =========================================================
def fetch_topoff(
    fecha: Optional[str] = None,
    hora: Optional[str] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    techs: Optional[List[str]] = None,
    limit: Optional[int] = 200,
    na_as_empty: bool = False,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame “amigable” desde dashboard_topoff con filtros básicos.
    - fecha: 'YYYY-MM-DD'
    - hora: 'HH:MM:SS' (o 'todas' para ignorar)
    - regions/provinces/municipalities/techs: lista o string
    """
    # 1) columnas
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    # 2) where
    where_sql, params, ur, up, um, ut = _filters_where_and_params(
        fecha, hora, regions, provinces, municipalities, techs
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
        stmt = _prepare_stmt_with_expanding(sql, ur, up, um, ut)
        df = pd.read_sql(stmt, conn, params=params)

    # 4) orden amigable garantizado
    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])

    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df

# src/topoff_data_access.py  (añade esto al final del archivo)

def fetch_topoff_paginated(
    fecha: Optional[str] = None,
    hora: Optional[str] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    techs: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 50,
    na_as_empty: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Retorna (df, total_rows) con LIMIT/OFFSET.
    Mismo set de filtros que fetch_topoff (aunque por ahora puedes llamarlo sin filtros).
    """
    # 1) columnas válidas
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    # 2) where + params
    where_sql, params, ur, up, um, ut = _filters_where_and_params(
        fecha, hora, regions, provinces, municipalities, techs
    )

    # 3) COUNT total
    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    # 4) SELECT page
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
        # total
        stmt_count = _prepare_stmt_with_expanding(count_sql, ur, up, um, ut)
        total = conn.execute(stmt_count, params).scalar() or 0

        # page
        sel_params = dict(params)
        sel_params.update({"limit": page_size, "offset": offset})
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, ur, up, um, ut)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    # 5) orden de columnas amigables
    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])

    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")

    return df, int(total)

def fetch_topoff_paginated_global_sort(
    fecha: Optional[str] = None,
    hora: Optional[str] = None,
    regions: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    municipalities: Optional[List[str]] = None,
    techs: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 50,
    sort_by_friendly: Optional[str] = None,
    ascending: bool = True,
    na_as_empty: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Igual a fetch_topoff_paginated, pero ordena en SQL por la columna 'friendly' indicada,
    empujando vacíos (NULL o '') al final y casteando numéricos cuando aplique.
    """
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    where_sql, params, ur, up, um, ut = _filters_where_and_params(
        fecha, hora, regions, provinces, municipalities, techs
    )

    # --- columnas tratadas como numéricas para ordenar ---
    NUMERIC_COLS = {
        "ps_traff_gb","ps_rrc_ia_percent","ps_rrc_fail","ps_rab_ia_percent","ps_rab_fail",
        "ps_s1_ia_percent","ps_s1_fail","ps_drop_dc_percent","ps_drop_abnrel",
        "cs_traff_erl","cs_rrc_ia_percent","cs_rrc_fail","cs_rab_ia_percent","cs_rab_fail",
        "cs_drop_dc_percent","cs_drop_abnrel",
        "unav","rtx_tnl_tx_percent","tnl_abn","tnl_fail"
    }

    # ORDER BY con NULLS LAST y manejo de '' como NULL
    if sort_by_friendly and sort_by_friendly in COLMAP:
        real = COLMAP[sort_by_friendly]
        direction = "ASC" if ascending else "DESC"

        # Trata '' como NULL para el ordenamiento
        if sort_by_friendly in NUMERIC_COLS:
            # Orden numérico real
            real_expr = f"CAST(NULLIF({_quote(real)}, '') AS DECIMAL(18,6))"
        else:
            # Orden lexicográfico/fecha según tipo de columna en BD
            real_expr = f"NULLIF({_quote(real)}, '')"

        # Empuja NULL al final (NULLS LAST) y luego ordena por el valor
        order_by = (
            f"({real_expr} IS NULL) ASC, "
            f"{real_expr} {direction}, "
            f"{_quote(COLMAP['fecha'])} DESC, "
            f"{_quote(COLMAP['hora'])} DESC"
        )
    else:
        order_by = f"{_quote(COLMAP['fecha'])} DESC, {_quote(COLMAP['hora'])} DESC"

    # COUNT
    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM {_quote_table(_TABLE_NAME)}
        WHERE {where_sql}
    """

    # PAGE
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
        stmt_count = _prepare_stmt_with_expanding(count_sql, ur, up, um, ut)
        total = conn.execute(stmt_count, params).scalar() or 0

        sel_params = dict(params)
        sel_params.update({"limit": page_size, "offset": offset})
        stmt_sel = _prepare_stmt_with_expanding(sel_sql, ur, up, um, ut)
        df = pd.read_sql(stmt_sel, conn, params=sel_params)

    df = df.reindex(columns=[c for c in friendly_cols if c in df.columns])
    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")
    return df, int(total)

