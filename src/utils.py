from datetime import datetime
import pytz
from .config import TZ

def now_local():
    tz = pytz.timezone(TZ)
    return datetime.now(tz)

def default_date_str():
    return now_local().strftime("%Y-%m-%d")

def default_hour_str():
    # redondea a la hora actual HH:MM:SS al inicio de la hora
    return now_local().strftime("%H:00:00")

# Umbrales para colorear celdas (ajústalos a tu operación)
THRESHOLDS = {
    # valores en %
    "ps_failure_rrc_percent": {"warn": 1.0, "bad": 2.0},  # <=1 ok, <=2 warn, >2 bad
    "ps_failures_rab_percent": {"warn": 1.0, "bad": 2.0},
    "cs_failures_rrc_percent": {"warn": 1.0, "bad": 2.0},
    "cs_failures_rab_percent": {"warn": 1.0, "bad": 2.0},
}

def cell_severity(column: str, value):
    """
    Devuelve 'ok' | 'warn' | 'bad' para colorear.
    """
    if value is None:
        return "ok"
    t = THRESHOLDS.get(column)
    if not t:
        return "ok"

    # % de fallas (mientras más alto peor)
    if "warn" in t and "bad" in t:
        if value > t["bad"]:
            return "bad"
        if value > t["warn"]:
            return "warn"
        return "ok"

    # tasas de éxito (mientras más bajo peor)
    if "warn_low" in t and "bad_low" in t:
        if value < t["bad_low"]:
            return "bad"
        if value < t["warn_low"]:
            return "warn"
        return "ok"

    return "ok"
# Config de escalas/formato por KPI para progress bars
PROGRESS_CONFIG = {
    # Ejemplos: ajusta a tus rangos reales
    "cs_failures_rrc":       {"min": 0, "max": 300, "decimals": 0, "label": "{value:.0f}"},
    "ps_failures_rab":       {"min": 0, "max": 300, "decimals": 0, "label": "{value:.0f}"},
    "cs_abnormal_releases":  {"min": 0, "max": 300, "decimals": 0, "label": "{value:.0f}"},
    "ps_failure_rrc":        {"min": 0, "max": 300, "decimals": 0, "label": "{value:.0f}"},
    "ps_abnormal_releases":  {"min": 0, "max": 300, "decimals": 0, "label": "{value:.0f}"},
    # Si alguna columna es realmente porcentaje, déjala en 0–100
    # "lcs_ps_rate": {"min": 0, "max": 100, "decimals": 1, "label": "{value:.1f}%"},
}

def progress_cfg(column: str):
    """
    Devuelve un dict con min/max/decimals/label para una columna de progress bar.
    Si no hay config específica, retorna defaults (0–100).
    """
    cfg = PROGRESS_CONFIG.get(column, {})
    return {
        "min": cfg.get("min", 0.0),
        "max": cfg.get("max", 100.0),
        "decimals": cfg.get("decimals", 1),
        "label": cfg.get("label", "{value:.1f}"),
    }

# config.py
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
