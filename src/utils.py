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
    # tasas de éxito (0-100): invertimos
    "lcs_ps_rate": {"warn_low": 99.0, "bad_low": 98.0},
    "lcs_cs_rate": {"warn_low": 99.0, "bad_low": 98.0}
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
