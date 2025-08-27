import pandas as pd
from functools import lru_cache
from sqlalchemy import create_engine, text, bindparam
from .config import SQLALCHEMY_URL

_engine = None
_TABLE_NAME = "telecom_kpis"

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True, pool_recycle=1800)
    return _engine

# Lista "deseada" de columnas en el orden que quieres devolverlas.
# Incluye las nuevas columnas agregadas y algunas clásicas.
BASE_COLUMNS = [
    # Identificadores
    "fecha", "hora", "vendor", "noc_cluster",

    # Integridad / PS
    "integrity",
    "total_mbytes_nocperf", "delta_total_mbytes_nocperf",
    "ps_failure_rrc_percent", "ps_failure_rrc",          # agregado: conteo
    "ps_failures_rab_percent", "ps_failures_rab",        # agregado: conteo
    "lcs_ps_rate", "ps_abnormal_releases",

    # CS
    "total_erlangs_nocperf", "delta_total_erlangs_nocperf",
    "cs_failures_rrc_percent", "cs_failures_rrc",        # agregado: conteo
    "cs_failures_rab_percent", "lcs_cs_rate", "cs_abnormal_releases", "cs_failures_rab"

    # Tráfico
    "traffic_gb_att", "delta_traffic_gb_att",
    "traffic_amr_att", "delta_traffic_amr_att",
    "delta_traffic_gb_plmn2", "traffic_gb_plmn2",
    "delta_traffic_amr_plmn2", "traffic_amr_plmn2",
]

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
    Devuelve un set con las columnas existentes en la tabla,
    consultando INFORMATION_SCHEMA. Se cachea para no golpear la BD cada vez.
    """
    eng = get_engine()
    sql = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = :tbl
    """
    with eng.connect() as conn:
        rows = conn.execute(text(sql), {"tbl": _TABLE_NAME}).fetchall()
    return {r[0] for r in rows}

def _resolve_columns(requested_cols):
    """
    Devuelve la lista de columnas a seleccionar preservando el orden,
    pero filtrando solo las que existen en la BD.
    Si ninguna existe, retorna ["*"] para no fallar.
    """
    existing = _existing_columns()
    cols = [c for c in requested_cols if c in existing]
    return cols if cols else ["*"]

def fetch_kpis(fecha=None, hora=None, vendors=None, clusters=None, limit=None):
    # columnas base ya filtradas a las que existen
    select_cols = _resolve_columns(BASE_COLUMNS)

    # reemplazar 'hora' por DATE_FORMAT(hora, '%H:%i:%s') AS hora
    formatted_select = []
    for c in select_cols:
        if c == "hora":
            formatted_select.append("DATE_FORMAT(hora, '%H:%i:%s') AS hora")
        else:
            formatted_select.append(c)

    sql = f"""
        SELECT {", ".join(formatted_select)}
        FROM {_TABLE_NAME}
        WHERE 1=1
    """
    params = {}
    use_vendors = False
    use_clusters = False

    if fecha:
        sql += " AND fecha = :fecha"
        params["fecha"] = fecha

    if hora and str(hora).lower() != "todas":
        sql += " AND hora = :hora"
        params["hora"] = hora  # puedes pasar '12:00:00' como string

    if vendors:
        sql += " AND vendor IN :vendors"
        params["vendors"] = list(vendors)
        use_vendors = True

    if clusters:
        sql += " AND noc_cluster IN :clusters"
        params["clusters"] = list(clusters)
        use_clusters = True

    sql += " ORDER BY fecha DESC, hora DESC"

    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)

    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(sql, use_vendors, use_clusters)
        df = pd.read_sql(stmt, conn, params=params)
    return df


def distinct_vendors(fecha=None, hora=None):
    sql = f"SELECT DISTINCT vendor FROM {_TABLE_NAME} WHERE 1=1"
    params = {}
    if fecha:
        sql += " AND fecha = :fecha"
        params["fecha"] = fecha
    if hora and str(hora).lower() != "todas":
        sql += " AND hora = :hora"
        params["hora"] = hora
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [r[0] for r in rows]

def distinct_clusters(fecha=None, hora=None, vendors=None):
    sql = f"SELECT DISTINCT noc_cluster FROM {_TABLE_NAME} WHERE 1=1"
    params = {}
    use_vendors = False
    if fecha:
        sql += " AND fecha = :fecha"
        params["fecha"] = fecha
    if hora and str(hora).lower() != "todas":
        sql += " AND hora = :hora"
        params["hora"] = hora
    if vendors:
        sql += " AND vendor IN :vendors"
        params["vendors"] = list(vendors)
        use_vendors = True
    eng = get_engine()
    with eng.connect() as conn:
        stmt = _prepare_stmt_with_expanding(sql, use_vendors, False)
        rows = conn.execute(stmt, params).fetchall()
    return [r[0] for r in rows]


