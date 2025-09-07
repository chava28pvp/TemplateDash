from datetime import datetime
import pytz
from src.config import TZ


HEADER_MAP = {
    "fecha": "FECHA",
    "hora": "HORA",
    "cs_traff_erl": "CS_TRAFF_ERL",
    "cs_rrc_ia_percent": "CS_RRC_%IA",
    "cs_rab_ia_percent": "CS_RAB_%IA",
    "cs_drop_dc_percent": "CS_%DC",

    "ps_traff_gb": "PS_TRAFF_GB",
    "ps_rrc_fail": "PS_RRC_%IA",
    "ps_rab_ia_percent": "PS_RAB_%IA",
    "ps_drop_dc_percen": "PS_%DC",

    "vendor": "VENDOR",
    "noc_cluster": "CLUSTER",
}

# Órdenes (qué columnas y en qué orden mostrar en cada tabla)
TABLE_TOP_ORDER = [
    "fecha", "hora", "cs_traff_erl",
    "cs_rrc_ia_percent", "cs_rab_ia_percent", "cs_drop_dc_percent",
]

TABLE_VENDOR_SUMMARY_ORDER = [
    "fecha", "hora", "ps_traff_gb",
    "ps_rrc_fail", "ps_rab_ia_percent", "ps_drop_dc_percen",
]

def cols_from_order(order, header_map=HEADER_MAP):
    """Devuelve [ (Header Amigable, nombre_columna_df), ... ]"""
    return [(header_map.get(col, col), col) for col in order]

