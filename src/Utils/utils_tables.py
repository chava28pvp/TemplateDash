from datetime import datetime
import pytz
from src.config import TZ


HEADER_MAP = {
    "fecha": "FECHA",
    "hora": "HORA",
    "total_erlangs_nocperf": "CS_TRAFF_ERL",
    "cs_failures_rrc_percent": "CS_RRC_%IA",
    "cs_failures_rab_percent": "CS_RAB_%IA",
    "lcs_cs_rate": "CS_%DC",

    "total_mbytes_nocperf": "PS_TRAFF_GB",
    "ps_failure_rrc_percent": "PS_RRC_%IA",
    "ps_failures_rab_percent": "PS_RAB_%IA",
    "lcs_ps_rate": "PS_%DC",

    "vendor": "VENDOR",
    "noc_cluster": "CLUSTER",
}

# Órdenes (qué columnas y en qué orden mostrar en cada tabla)
TABLE_TOP_ORDER = [
    "fecha", "hora", "total_erlangs_nocperf",
    "cs_failures_rrc_percent", "cs_failures_rab_percent", "lcs_cs_rate",
]

TABLE_VENDOR_SUMMARY_ORDER = [
    "fecha", "hora", "total_mbytes_nocperf",
    "ps_failure_rrc_percent", "ps_failures_rab_percent", "lcs_ps_rate",
]

def cols_from_order(order, header_map=HEADER_MAP):
    """Devuelve [ (Header Amigable, nombre_columna_df), ... ]"""
    return [(header_map.get(col, col), col) for col in order]

