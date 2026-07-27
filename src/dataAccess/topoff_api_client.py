import logging
import copy
import json
import threading
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests
import urllib3

from src.config import (
    TOPOFF_QUERY_API_CA_BUNDLE,
    TOPOFF_QUERY_API_DEBUG,
    TOPOFF_QUERY_API_TIMEOUT,
    TOPOFF_QUERY_API_TOKEN,
    TOPOFF_QUERY_API_TOKEN_HEADER,
    TOPOFF_QUERY_API_TOKEN_PREFIX,
    TOPOFF_QUERY_API_URL,
    TOPOFF_QUERY_API_VERIFY_SSL,
)

logger = logging.getLogger(__name__)
_API_CACHE = {}
_API_CACHE_TTL = 60
_ROWS_CACHE = {}
_ROWS_CACHE_TTL = 120
_ROWS_CACHE_LOCK = threading.Lock()
_ALL_ROWS_PAGE_SIZE = 2000

_NUMERIC_COLS = {
    "ps_traff_gb", "ps_rrc_ia_percent", "ps_rrc_fail",
    "ps_rab_ia_percent", "ps_rab_fail", "ps_s1_ia_percent", "ps_s1_fail",
    "ps_drop_dc_percent", "ps_drop_abnrel",
    "cs_traff_erl", "cs_rrc_ia_percent", "cs_rrc_fail",
    "cs_rab_ia_percent", "cs_rab_fail", "cs_drop_dc_percent", "cs_drop_abnrel",
    "unav", "rtx_tnl_tx_percent", "tnl_abn", "tnl_fail",
}


class TopoffApiError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(TOPOFF_QUERY_API_URL)


def fetch_page(
    *,
    fecha=None,
    hora=None,
    technologies=None,
    vendors=None,
    clusters=None,
    sites=None,
    rncs=None,
    nodebs=None,
    page=1,
    page_size=50,
    mode="recent",
    sort_by=None,
    ascending=True,
):
    cached_rows = None
    if str(mode or "recent").lower() != "alarmado" and not sort_by:
        cached_rows = _fetch_rows_cached(
            fecha=fecha,
            technologies=technologies,
            vendors=vendors,
            clusters=clusters,
            sites=sites,
            rncs=rncs,
            nodebs=nodebs,
        )
    if cached_rows is not None:
        rows = _filter_rows_by_hora(cached_rows, hora)
        rows = _sort_rows(rows, mode=mode, sort_by=sort_by, ascending=ascending)
        total = len(rows)
        page = max(1, int(page or 1))
        page_size = max(1, int(page_size or 50))
        start = (page - 1) * page_size
        return pd.DataFrame(rows[start:start + page_size]), total

    payload = _base_payload(
        fecha=fecha,
        hora=hora,
        technologies=technologies,
        vendors=vendors,
        clusters=clusters,
        sites=sites,
        rncs=rncs,
        nodebs=nodebs,
    )
    payload.update({
        "pagination": {"page": int(page or 1), "page_size": int(page_size or 50)},
        "mode": mode or "recent",
        "sort": {"column": sort_by, "ascending": bool(ascending)},
    })
    data = call_operation("page", payload)
    body = data.get("data") or {}
    return pd.DataFrame(body.get("rows") or []), int(body.get("total") or 0)


def fetch_distinct_options(*, fecha=None, technologies=None, vendors=None, clusters=None):
    payload = _base_payload(
        fecha=fecha,
        technologies=technologies,
        vendors=vendors,
        clusters=clusters,
    )
    data = call_operation("distinct_options", payload)
    body = data.get("data") or {}
    return body.get("sites") or [], body.get("rncs") or [], body.get("nodebs") or []


def fetch_latest_slot():
    data = call_operation("latest_slot", {})
    return (data.get("data") or {}).get("slot")


def _fetch_rows_cached(*, fecha=None, technologies=None, vendors=None, clusters=None, sites=None, rncs=None, nodebs=None):
    payload = _base_payload(
        fecha=fecha,
        hora="todas",
        technologies=technologies,
        vendors=vendors,
        clusters=clusters,
        sites=sites,
        rncs=rncs,
        nodebs=nodebs,
    )
    payload.update({
        "pagination": {"page": 1, "page_size": _ALL_ROWS_PAGE_SIZE},
        "mode": "recent",
        "sort": {"column": None, "ascending": True},
    })
    cache_key = _cache_key(payload)
    now = time.time()
    cached = _ROWS_CACHE.get(cache_key)
    if cached and (now - cached["ts"] < _ROWS_CACHE_TTL):
        return copy.deepcopy(cached["rows"])

    with _ROWS_CACHE_LOCK:
        now = time.time()
        cached = _ROWS_CACHE.get(cache_key)
        if cached and (now - cached["ts"] < _ROWS_CACHE_TTL):
            return copy.deepcopy(cached["rows"])

        try:
            data = call_operation("page", payload)
            body = data.get("data") or {}
            rows = body.get("rows") or []
            total = int(body.get("total") or len(rows))
        except Exception:
            logger.exception("No se pudo cachear universo TopOff; se usara paginacion directa.")
            return None

        if total > len(rows):
            return None
        _ROWS_CACHE[cache_key] = {"ts": now, "rows": copy.deepcopy(rows)}
        return rows


def _filter_rows_by_hora(rows, hora):
    h_range = _hora_range(hora)
    if not h_range:
        return list(rows or [])
    start, end = h_range
    return [
        row for row in (rows or [])
        if start <= str(row.get("hora") or "")[:5] < end
    ]


def _hora_range(hora):
    if not hora:
        return None
    s = str(hora).strip()
    if not s or s.lower() == "todas":
        return None
    try:
        hh = int(s.split(":")[0])
    except Exception:
        return None
    start = max(0, hh - 1)
    end = min(24, hh + 1)
    return f"{start:02d}:00", f"{end:02d}:00"


def _sort_rows(rows, *, mode="recent", sort_by=None, ascending=True):
    rows = list(rows or [])
    mode = str(mode or "recent").lower()
    if mode == "sitio":
        return sorted(
            rows,
            key=lambda row: (
                _text(row.get("site_att")),
                _desc_text(row.get("fecha")),
                _desc_text(row.get("hora")),
            ),
        )
    if sort_by:
        direction = bool(ascending)
        return sorted(
            rows,
            key=lambda row: (
                _sort_value(row, sort_by),
                _desc_text(row.get("fecha")),
                _desc_text(row.get("hora")),
            ),
            reverse=not direction,
        )
    return sorted(rows, key=lambda row: (_text(row.get("fecha")), _text(row.get("hora"))), reverse=True)


def _sort_value(row, col):
    value = row.get(col)
    if col in _NUMERIC_COLS:
        numeric = _to_float(value)
        return (numeric is None, float(numeric or 0.0))
    return (value in (None, ""), _text(value))


def _text(value):
    return "" if value is None else str(value)


def _desc_text(value):
    text = _text(value)
    return "".join(chr(255 - ord(ch)) for ch in text)


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def call_operation(operation: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TOPOFF_QUERY_API_URL:
        raise TopoffApiError("TOPOFF_QUERY_API_URL no esta configurado.")

    body = dict(payload or {})
    body["operation"] = operation
    cache_key = _cache_key(body)
    cached = _API_CACHE.get(cache_key)
    now = time.time()
    if cached and (now - cached["ts"] < _API_CACHE_TTL):
        return copy.deepcopy(cached["data"])

    headers = _auth_headers()
    verify = _verify_setting()
    if verify is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if TOPOFF_QUERY_API_DEBUG:
        logger.warning("topoff api request operation=%s payload_keys=%s", operation, sorted(body.keys()))

    try:
        response = requests.post(
            TOPOFF_QUERY_API_URL,
            json=body,
            headers=headers,
            timeout=TOPOFF_QUERY_API_TIMEOUT,
            verify=verify,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TopoffApiError(f"Error llamando topoff API operation={operation}: {exc}") from exc

    try:
        raw = response.json()
    except ValueError as exc:
        raise TopoffApiError(f"Respuesta no JSON de topoff API operation={operation}: {response.text[:500]}") from exc

    data = _unwrap_platform_response(raw)
    if not data.get("ok", data.get("success", False)):
        message = ((data.get("error") or {}).get("message")) or str(data.get("error") or data)
        raise TopoffApiError(f"topoff API operation={operation} fallo: {message}")
    _API_CACHE[cache_key] = {"ts": now, "data": copy.deepcopy(data)}
    return data


def _base_payload(
    *,
    fecha=None,
    hora=None,
    technologies=None,
    vendors=None,
    clusters=None,
    sites=None,
    rncs=None,
    nodebs=None,
):
    return {
        "view": "topoff",
        "fecha": fecha,
        "hora": hora,
        "filters": {
            "fecha": fecha,
            "hora": hora,
            "technologies": _as_list(technologies),
            "vendors": _as_list(vendors),
            "clusters": _as_list(clusters),
            "sites": _as_list(sites),
            "rncs": _as_list(rncs),
            "nodebs": _as_list(nodebs),
        },
    }


def _auth_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if TOPOFF_QUERY_API_TOKEN_HEADER and TOPOFF_QUERY_API_TOKEN:
        token_value = TOPOFF_QUERY_API_TOKEN
        if TOPOFF_QUERY_API_TOKEN_PREFIX:
            token_value = f"{TOPOFF_QUERY_API_TOKEN_PREFIX} {token_value}"
        headers[TOPOFF_QUERY_API_TOKEN_HEADER] = token_value
    return headers


def _verify_setting():
    if TOPOFF_QUERY_API_CA_BUNDLE:
        return TOPOFF_QUERY_API_CA_BUNDLE
    return bool(TOPOFF_QUERY_API_VERIFY_SSL)


def _unwrap_platform_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict) and ("status" in raw or "Status" in raw):
        return raw["data"]
    return raw


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "")]
    return [value] if value != "" else []


def _cache_key(body: Dict[str, Any]) -> str:
    try:
        return json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        return repr(sorted(body.items()))
