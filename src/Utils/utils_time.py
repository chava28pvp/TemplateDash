from datetime import datetime, timezone
import pytz
import os
#utils para funciones de tiempo tales como no reiniciar la hora si esta filtado el dashboard
TZ = os.getenv("APP_TZ", "America/Monterrey")
try:
    _TZ = pytz.timezone(TZ)
except Exception:
    _TZ = pytz.utc

def now_local():
    tz = pytz.timezone(TZ)
    return datetime.now(_TZ)

def floor_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

def today_str():
    return now_local().strftime("%Y-%m-%d")

def hour_start_str():
    return floor_to_hour(now_local()).strftime("%H:00:00")

def to_local(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_TZ)

def default_date_str():
    return now_local().strftime("%Y-%m-%d")

def default_hour_str():
    # redondea a la hora actual HH:MM:SS al inicio de la hora
    return now_local().strftime("%H:00:00")