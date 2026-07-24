import logging
import copy
import json
import time
from typing import Any, Dict, Optional

import requests
import urllib3

from src.config import (
    TOPOFF_VISUALS_API_CA_BUNDLE,
    TOPOFF_VISUALS_API_DEBUG,
    TOPOFF_VISUALS_API_MAX_ROWS,
    TOPOFF_VISUALS_API_TIMEOUT,
    TOPOFF_VISUALS_API_TOKEN,
    TOPOFF_VISUALS_API_TOKEN_HEADER,
    TOPOFF_VISUALS_API_TOKEN_PREFIX,
    TOPOFF_VISUALS_API_URL,
    TOPOFF_VISUALS_API_VERIFY_SSL,
)

logger = logging.getLogger(__name__)
_API_CACHE = {}
_API_CACHE_TTL = 60


class TopoffVisualsApiError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(TOPOFF_VISUALS_API_URL)


def fetch_heatmap(
    *,
    fecha=None,
    technologies=None,
    vendors=None,
    clusters=None,
    sites=None,
    rncs=None,
    nodebs=None,
    page=1,
    page_size=50,
    order_by="alarm_bins_pct",
    thresholds_snapshot=None,
):
    payload = _base_payload(
        fecha=fecha,
        technologies=technologies,
        vendors=vendors,
        clusters=clusters,
        sites=sites,
        rncs=rncs,
        nodebs=nodebs,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["pagination"] = {"page": int(page or 1), "page_size": int(page_size or 50)}
    payload["order_by"] = order_by or "alarm_bins_pct"
    data = call_operation("heatmap", payload)
    return data.get("data") or {}


def fetch_histogram(
    *,
    fecha=None,
    domain="PS",
    technologies=None,
    vendors=None,
    clusters=None,
    sites=None,
    rncs=None,
    nodebs=None,
    page=1,
    page_size=50,
    thresholds_snapshot=None,
):
    payload = _base_payload(
        fecha=fecha,
        technologies=technologies,
        vendors=vendors,
        clusters=clusters,
        sites=sites,
        rncs=rncs,
        nodebs=nodebs,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["pagination"] = {"page": int(page or 1), "page_size": int(page_size or 50)}
    payload["domain"] = str(domain or "PS").upper()
    data = call_operation("histogram", payload)
    return data.get("data") or {}


def call_operation(operation: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TOPOFF_VISUALS_API_URL:
        raise TopoffVisualsApiError("TOPOFF_VISUALS_API_URL no esta configurado.")

    body = dict(payload or {})
    body["operation"] = operation
    body.setdefault("options", {})
    body["options"]["max_rows"] = int(TOPOFF_VISUALS_API_MAX_ROWS or 50000)
    cache_key = _cache_key(body)
    cached = _API_CACHE.get(cache_key)
    now = time.time()
    if cached and (now - cached["ts"] < _API_CACHE_TTL):
        return copy.deepcopy(cached["data"])

    headers = _auth_headers()
    verify = _verify_setting()
    if verify is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if TOPOFF_VISUALS_API_DEBUG:
        logger.warning("topoff visuals api request operation=%s payload_keys=%s", operation, sorted(body.keys()))

    try:
        response = requests.post(
            TOPOFF_VISUALS_API_URL,
            json=body,
            headers=headers,
            timeout=TOPOFF_VISUALS_API_TIMEOUT,
            verify=verify,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TopoffVisualsApiError(f"Error llamando topoff visuals API operation={operation}: {exc}") from exc

    try:
        raw = response.json()
    except ValueError as exc:
        raise TopoffVisualsApiError(
            f"Respuesta no JSON de topoff visuals API operation={operation}: {response.text[:500]}"
        ) from exc

    data = _unwrap_platform_response(raw)
    if not data.get("ok", data.get("success", False)):
        message = ((data.get("error") or {}).get("message")) or str(data.get("error") or data)
        raise TopoffVisualsApiError(f"topoff visuals API operation={operation} fallo: {message}")
    _API_CACHE[cache_key] = {"ts": now, "data": copy.deepcopy(data)}
    return data


def _base_payload(
    *,
    fecha=None,
    technologies=None,
    vendors=None,
    clusters=None,
    sites=None,
    rncs=None,
    nodebs=None,
    thresholds_snapshot=None,
):
    payload = {
        "view": "topoff_visuals",
        "fecha": fecha,
        "filters": {
            "fecha": fecha,
            "technologies": _as_list(technologies),
            "vendors": _as_list(vendors),
            "clusters": _as_list(clusters),
            "sites": _as_list(sites),
            "rncs": _as_list(rncs),
            "nodebs": _as_list(nodebs),
        },
    }
    if thresholds_snapshot:
        payload["thresholds_snapshot"] = thresholds_snapshot
    return payload


def _auth_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if TOPOFF_VISUALS_API_TOKEN_HEADER and TOPOFF_VISUALS_API_TOKEN:
        token_value = TOPOFF_VISUALS_API_TOKEN
        if TOPOFF_VISUALS_API_TOKEN_PREFIX:
            token_value = f"{TOPOFF_VISUALS_API_TOKEN_PREFIX} {token_value}"
        headers[TOPOFF_VISUALS_API_TOKEN_HEADER] = token_value
    return headers


def _verify_setting():
    if TOPOFF_VISUALS_API_CA_BUNDLE:
        return TOPOFF_VISUALS_API_CA_BUNDLE
    return bool(TOPOFF_VISUALS_API_VERIFY_SSL)


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
