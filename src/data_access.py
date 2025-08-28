import pandas as pd
from functools import lru_cache
from sqlalchemy import create_engine, text, bindparam
from .config import SQLALCHEMY_URL

_engine = None
_TABLE_NAME = "Dashboard_Master"

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True, pool_recycle=1800)
    return _engine

# --------- Mapeo de columnas amigables -> columnas reales en BD ---------
# OJO: las columnas con % requieren backticks en SQL.
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

# Orden “deseado” para el DataFrame que expones al resto de tu app.
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

def _quote(colname: str) -> str:
    """Coloca backticks para columnas que lo requieran (%, palabras reservadas, etc.)."""
    return f"`{colname}`"

def _prepare_stmt_with_expanding(sql, use_vendors=False, use_clusters=False):
    stmt = text(sql)
    if use_vendors:
        stmt = stmt.bindparams(bindparam("vendors", expanding=True))
    if use_clusters:
        stmt = stmt.bindparams(bindparam("clusters", expanding=True))
    return stmt

@lru_cache(maxsize=1)
def _existing_columns():
    """
    Devuelve set de columnas existentes en la tabla (nombres reales).
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
    Filtra las columnas amigables a solo las que existen en BD (vía COLMAP y INFORMATION_SCHEMA).
    Retorna lista de columnas amigables válidas.
    """
    existing_real = _existing_columns()
    cols = []
    for friendly in requested_friendly_cols:
        real = COLMAP.get(friendly)
        if real and real in existing_real:
            cols.append(friendly)
    return cols if cols else []  # si vacío, más abajo usamos * como fallback

def _select_list_with_aliases(friendly_cols):
    """
    Construye la lista de SELECT con backticks y alias amigables.
    Aplica DATE_FORMAT a la hora para HH:MM:SS (alias 'hora').
    """
    if not friendly_cols:
        return ["*"]

    select_parts = []
    for friendly in friendly_cols:
        real = COLMAP[friendly]
        if friendly == "hora":
            # Normaliza a 'HH:MM:SS'
            select_parts.append(f"DATE_FORMAT({_quote(real)}, '%H:%i:%s') AS {friendly}")
        elif friendly == "fecha":
            select_parts.append(f"{_quote(real)} AS {friendly}")
        else:
            select_parts.append(f"{_quote(real)} AS {friendly}")
    return select_parts

def fetch_kpis(fecha=None, hora=None, vendors=None, clusters=None, limit=None):
    # columnas que mostraremos (amigables)
    friendly_cols = _resolve_columns(BASE_COLUMNS)
    select_cols = _select_list_with_aliases(friendly_cols)

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM {_TABLE_NAME}
        WHERE 1=1
    """
    params = {}
    use_vendors = False
    use_clusters = False

    if fecha:
        sql += f" AND {_quote(COLMAP['fecha'])} = :fecha"
        params["fecha"] = fecha

    if hora and str(hora).lower() != "todas":
        # Acepta '12:00:00' como string; la columna es TIME
        sql += f" AND {_quote(COLMAP['hora'])} = :hora"
        params["hora"] = hora

    if vendors:
        sql += f" AND {_quote(COLMAP['vendor'])} IN :vendors"
        params["vendors"] = list(vendors)
        use_vendors = True

    if clusters:
        sql += f" AND {_quote(COLMAP['noc_cluster'])} IN :clusters"
        params["clusters"] = list(clusters)
        use_clusters = True

    # Orden nuevo por fecha y hora reales
    sql += f" ORDER BY {_quote(COLMAP['fecha'])} DESC, {_quote(COLMAP['hora'])} DESC"

    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)

    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(sql, use_vendors, use_clusters)
        df = pd.read_sql(stmt, conn, params=params)
    return df

def distinct_vendors(fecha=None, hora=None):
    real_vendor = _quote(COLMAP["vendor"])
    sql = f"SELECT DISTINCT {real_vendor} AS vendor FROM {_TABLE_NAME} WHERE 1=1"
    params = {}
    if fecha:
        sql += f" AND {_quote(COLMAP['fecha'])} = :fecha"
        params["fecha"] = fecha
    if hora and str(hora).lower() != "todas":
        sql += f" AND {_quote(COLMAP['hora'])} = :hora"
        params["hora"] = hora

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [r[0] for r in rows]

def distinct_clusters(fecha=None, hora=None, vendors=None):
    real_cluster = _quote(COLMAP["noc_cluster"])
    sql = f"SELECT DISTINCT {real_cluster} AS noc_cluster FROM {_TABLE_NAME} WHERE 1=1"
    params = {}
    use_vendors = False

    if fecha:
        sql += f" AND {_quote(COLMAP['fecha'])} = :fecha"
        params["fecha"] = fecha
    if hora and str(hora).lower() != "todas":
        sql += f" AND {_quote(COLMAP['hora'])} = :hora"
        params["hora"] = hora
    if vendors:
        sql += f" AND {_quote(COLMAP['vendor'])} IN :vendors"
        params["vendors"] = list(vendors)
        use_vendors = True

    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(sql, use_vendors, False)
        rows = conn.execute(stmt, params).fetchall()
    return [r[0] for r in rows]
