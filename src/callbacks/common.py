from datetime import datetime
from dash import ctx


def reset_page_state(page_size, default_size=50):
    ps = max(1, int(page_size or default_size))
    return {"page": 1, "page_size": ps}


def paginate_state(state, prev_id, next_id, default_size=50):
    state = state or {"page": 1, "page_size": default_size}
    page = int(state.get("page", 1))
    ps = int(state.get("page_size", default_size))

    trig = ctx.triggered_id
    if trig == prev_id:
        page = max(1, page - 1)
    elif trig == next_id:
        page = page + 1

    return {"page": page, "page_size": ps}


def toggle_bool(n_clicks, current_value):
    if not n_clicks:
        return current_value
    return not current_value


def choose_common_available_slot(*slots):
    """
    Toma varias marcas fecha/hora y devuelve la más reciente disponible entre ellas.
    """
    valid = []
    for slot in slots:
        if not slot:
            continue
        fecha = (slot.get("fecha") or "").strip()
        hora = (slot.get("hora") or "").strip()
        if not fecha or not hora:
            continue
        try:
            dt = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        valid.append((dt, {"fecha": fecha, "hora": hora}))

    if not valid:
        return None

    _dt, slot = max(valid, key=lambda x: x[0])
    return slot


def purge_expired_cache_entries(cache: dict, ttl: int, now_ts: float | None = None) -> int:
    """
    Elimina entradas expiradas de un diccionario con esquema:
      key -> {"ts": <epoch>, ...}
    """
    if not cache or ttl is None:
        return 0

    now_ts = now_ts or datetime.now().timestamp()
    stale_keys = []
    for key, value in list(cache.items()):
        ts = (value or {}).get("ts")
        try:
            is_stale = (ts is None) or ((now_ts - float(ts)) >= float(ttl))
        except Exception:
            is_stale = True
        if is_stale:
            stale_keys.append(key)

    for key in stale_keys:
        cache.pop(key, None)

    return len(stale_keys)
