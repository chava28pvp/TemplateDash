import pandas as pd
from sqlalchemy import create_engine, text, bindparam
from .config import SQLALCHEMY_URL

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True, pool_recycle=1800)
    return _engine

BASE_COLUMNS = [
    "fecha","hora","vendor","noc_cluster",
    "total_mbytes_nocperf","delta_total_mbytes_nocperf",
    "ps_failure_rrc_percent","ps_failures_rab_percent","lcs_ps_rate","ps_abnormal_releases",
    "total_erlangs_nocperf","delta_total_erlangs_nocperf",
    "cs_failures_rrc_percent","cs_failures_rab_percent","lcs_cs_rate","cs_abnormal_releases",
    "traffic_gb_att","delta_traffic_gb_att","traffic_amr_att","delta_traffic_amr_att",
    "delta_traffic_gb_plmn2","traffic_gb_plmn2","delta_traffic_amr_plmn2","traffic_amr_plmn2"
]

def _prepare_stmt_with_expanding(sql, use_vendors=False, use_clusters=False):
    stmt = text(sql)
    if use_vendors:
        stmt = stmt.bindparams(bindparam("vendors", expanding=True))
    if use_clusters:
        stmt = stmt.bindparams(bindparam("clusters", expanding=True))
    return stmt

def fetch_kpis(fecha=None, hora=None, vendors=None, clusters=None, limit=None):
    sql = f"""
        SELECT {", ".join(BASE_COLUMNS)}
        FROM telecom_kpis
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
        params["hora"] = hora

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
    sql = "SELECT DISTINCT vendor FROM telecom_kpis WHERE 1=1"
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
    sql = "SELECT DISTINCT noc_cluster FROM telecom_kpis WHERE 1=1"
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
