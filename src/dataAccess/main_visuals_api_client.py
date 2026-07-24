import logging
from typing import Any, Dict, Optional

import requests
import urllib3

from src.config import (
    HEATMAP_QUERY_API_CA_BUNDLE,
    HEATMAP_QUERY_API_DEBUG,
    HEATMAP_QUERY_API_RANK_ONLY_MAX_ROWS,
    HEATMAP_QUERY_API_TIMEOUT,
    HEATMAP_QUERY_API_TOKEN,
    HEATMAP_QUERY_API_TOKEN_HEADER,
    HEATMAP_QUERY_API_TOKEN_PREFIX,
    HEATMAP_QUERY_API_URL,
    HEATMAP_QUERY_API_USE_VISUAL_SERIES,
    HEATMAP_QUERY_API_VERIFY_SSL,
)

logger = logging.getLogger(__name__)


class MainVisualsApiError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(HEATMAP_QUERY_API_URL)


def call_operation(operation: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not HEATMAP_QUERY_API_URL:
        raise MainVisualsApiError("HEATMAP_QUERY_API_URL no esta configurado.")

    body = dict(payload or {})
    body["operation"] = operation

    headers = _auth_headers()
    verify = _verify_setting()
    if verify is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if HEATMAP_QUERY_API_DEBUG:
        logger.warning("heatmap api request operation=%s payload_keys=%s", operation, sorted(body.keys()))

    try:
        response = requests.post(
            HEATMAP_QUERY_API_URL,
            json=body,
            headers=headers,
            timeout=HEATMAP_QUERY_API_TIMEOUT,
            verify=verify,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MainVisualsApiError(f"Error llamando heatmap API operation={operation}: {exc}") from exc

    try:
        raw = response.json()
    except ValueError as exc:
        raise MainVisualsApiError(
            f"Respuesta no JSON de heatmap API operation={operation}: {response.text[:500]}"
        ) from exc

    data = _unwrap_platform_response(raw)
    if not data.get("ok", data.get("success", False)):
        message = ((data.get("error") or {}).get("message")) or str(data.get("error") or data)
        raise MainVisualsApiError(f"heatmap API operation={operation} fallo: {message}")
    return data


def fetch_main_heatmap(
    *,
    fecha,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page=1,
    page_size=50,
    order_by="alarm_hours",
    thresholds_snapshot=None,
    max_rows=200000,
):
    payload = _base_payload(
        fecha=fecha,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["pagination"] = {"page": int(page or 1), "page_size": int(page_size or 50)}
    payload["order_by"] = order_by
    payload["options"] = _options(max_rows=max_rows)
    data = call_operation("main_heatmap", payload)
    return data.get("data") or {}


def fetch_histogram(
    *,
    fecha,
    domain,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page=1,
    page_size=50,
    thresholds_snapshot=None,
    max_rows=200000,
):
    payload = _base_payload(
        fecha=fecha,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["pagination"] = {"page": int(page or 1), "page_size": int(page_size or 50)}
    payload["domain"] = str(domain or "PS").upper()
    payload["options"] = _options(max_rows=max_rows)
    data = call_operation("histogram", payload)
    return data.get("data") or {}


def fetch_integrity_heatmap(
    *,
    fecha,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page=1,
    page_size=50,
    max_rows=200000,
):
    payload = _base_payload(
        fecha=fecha,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
    )
    payload["pagination"] = {"page": int(page or 1), "page_size": int(page_size or 50)}
    payload["options"] = _options(max_rows=max_rows)
    data = call_operation("integrity_heatmap", payload)
    return data.get("data") or {}


def _options(*, max_rows=200000):
    return {
        "max_rows": int(max_rows or 200000),
        "use_visual_series": bool(HEATMAP_QUERY_API_USE_VISUAL_SERIES),
        "rank_only_max_rows": int(HEATMAP_QUERY_API_RANK_ONLY_MAX_ROWS or 50000),
    }


def _base_payload(
    *,
    fecha=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    thresholds_snapshot=None,
):
    payload = {
        "view": "main_visuals",
        "fecha": fecha,
        "filters": {
            "fecha": fecha,
            "vendors": _as_list(vendors),
            "clusters": _as_list(clusters),
            "networks": _as_list(networks),
            "technologies": _as_list(technologies),
        },
    }
    if thresholds_snapshot:
        payload["thresholds_snapshot"] = thresholds_snapshot
    return payload


def _auth_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if HEATMAP_QUERY_API_TOKEN_HEADER and HEATMAP_QUERY_API_TOKEN:
        token_value = HEATMAP_QUERY_API_TOKEN
        if HEATMAP_QUERY_API_TOKEN_PREFIX:
            token_value = f"{HEATMAP_QUERY_API_TOKEN_PREFIX} {token_value}"
        headers[HEATMAP_QUERY_API_TOKEN_HEADER] = token_value
    return headers


def _verify_setting():
    if HEATMAP_QUERY_API_CA_BUNDLE:
        return HEATMAP_QUERY_API_CA_BUNDLE
    return bool(HEATMAP_QUERY_API_VERIFY_SSL)


def _unwrap_platform_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict) and (
        "status" in raw or "Status" in raw
    ):
        return raw["data"]
    return raw


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "")]
    return [value] if value != "" else []
