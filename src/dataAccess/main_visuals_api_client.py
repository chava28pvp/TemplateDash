import logging
import copy
import json
import threading
import time
from typing import Any, Dict, Optional

import requests
import urllib3

from src.config import (
    HEATMAP_QUERY_API_CA_BUNDLE,
    HEATMAP_QUERY_API_CACHE_TTL,
    HEATMAP_QUERY_API_DEBUG,
    HEATMAP_QUERY_API_HISTOGRAM_BLOCK_ROWS,
    HEATMAP_QUERY_API_INTEGRITY_KEY_CHUNK_SIZE,
    HEATMAP_QUERY_API_PAGE_BLOCK_ROWS,
    HEATMAP_QUERY_API_PREFETCH_PAGES,
    HEATMAP_QUERY_API_PREFETCH_WAIT_SECONDS,
    HEATMAP_QUERY_API_RANK_ONLY_MAX_ROWS,
    HEATMAP_QUERY_API_RUNTIME_CANDIDATE_BLOCK_HOURS,
    HEATMAP_QUERY_API_RUNTIME_CANDIDATE_BUFFER,
    HEATMAP_QUERY_API_RUNTIME_CANDIDATE_PAGE,
    HEATMAP_QUERY_API_TIMEOUT,
    HEATMAP_QUERY_API_TOKEN,
    HEATMAP_QUERY_API_TOKEN_HEADER,
    HEATMAP_QUERY_API_TOKEN_PREFIX,
    HEATMAP_QUERY_API_URL,
    HEATMAP_QUERY_API_USE_VISUAL_RANK,
    HEATMAP_QUERY_API_USE_VISUAL_SERIES,
    HEATMAP_QUERY_API_VERIFY_SSL,
)

logger = logging.getLogger(__name__)
_API_CACHE = {}
_PREFETCH_INFLIGHT = set()


class MainVisualsApiError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(HEATMAP_QUERY_API_URL)


def clear_cache() -> None:
    _API_CACHE.clear()
    _PREFETCH_INFLIGHT.clear()


def call_operation(operation: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not HEATMAP_QUERY_API_URL:
        raise MainVisualsApiError("HEATMAP_QUERY_API_URL no esta configurado.")

    body = dict(payload or {})
    body["operation"] = operation
    cache_key = _cache_key(body)
    now = time.time()
    cached = _API_CACHE.get(cache_key)
    if cached and (now - cached["ts"] < HEATMAP_QUERY_API_CACHE_TTL):
        return copy.deepcopy(cached["data"])
    waited = _wait_for_prefetch(cache_key)
    if waited is not None:
        return waited

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
    if HEATMAP_QUERY_API_CACHE_TTL > 0:
        _API_CACHE[cache_key] = {"ts": now, "data": copy.deepcopy(data)}
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
    page = int(page or 1)
    page_size = int(page_size or 50)
    block_rows = _main_heatmap_block_rows(page_size)
    if block_rows > page_size:
        requested_offset = max(0, (page - 1) * page_size)
        block_offset = (requested_offset // block_rows) * block_rows
        block_page = (block_offset // block_rows) + 1
        payload = _base_payload(
            fecha=fecha,
            vendors=vendors,
            clusters=clusters,
            networks=networks,
            technologies=technologies,
            thresholds_snapshot=thresholds_snapshot,
        )
        payload["pagination"] = {"page": block_page, "page_size": block_rows}
        payload["order_by"] = order_by
        payload["options"] = _options(max_rows=max_rows)
        data = call_operation("main_heatmap", payload)
        return _slice_visual_page(
            data.get("data") or {},
            requested_offset=requested_offset,
            requested_limit=page_size,
            block_offset=block_offset,
        )

    payload = _base_payload(
        fecha=fecha,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["pagination"] = {"page": page, "page_size": page_size}
    payload["order_by"] = order_by
    payload["options"] = _options(max_rows=max_rows)
    data = call_operation("main_heatmap", payload)
    _prefetch_following_pages("main_heatmap", payload, page=page, page_size=page_size)
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
    page = int(page or 1)
    page_size = int(page_size or 50)
    block_rows = _main_histogram_block_rows(page_size)
    if block_rows > page_size:
        requested_offset = max(0, (page - 1) * page_size)
        block_offset = (requested_offset // block_rows) * block_rows
        block_page = (block_offset // block_rows) + 1
        payload = _base_payload(
            fecha=fecha,
            vendors=vendors,
            clusters=clusters,
            networks=networks,
            technologies=technologies,
            thresholds_snapshot=thresholds_snapshot,
        )
        payload["pagination"] = {"page": block_page, "page_size": block_rows}
        payload["domain"] = str(domain or "PS").upper()
        payload["options"] = _options(max_rows=max_rows)
        data = call_operation("histogram", payload)
        return _slice_visual_page(
            data.get("data") or {},
            requested_offset=requested_offset,
            requested_limit=page_size,
            block_offset=block_offset,
        )

    payload = _base_payload(
        fecha=fecha,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["pagination"] = {"page": page, "page_size": page_size}
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
        "use_visual_rank": bool(HEATMAP_QUERY_API_USE_VISUAL_RANK),
        "use_visual_series": bool(HEATMAP_QUERY_API_USE_VISUAL_SERIES),
        "rank_only_max_rows": int(HEATMAP_QUERY_API_RANK_ONLY_MAX_ROWS or 50000),
        "runtime_candidate_page": bool(HEATMAP_QUERY_API_RUNTIME_CANDIDATE_PAGE),
        "runtime_candidate_buffer": int(HEATMAP_QUERY_API_RUNTIME_CANDIDATE_BUFFER or 0),
        "runtime_candidate_block_hours": int(HEATMAP_QUERY_API_RUNTIME_CANDIDATE_BLOCK_HOURS or 1),
        "integrity_key_chunk_size": int(HEATMAP_QUERY_API_INTEGRITY_KEY_CHUNK_SIZE or 10),
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


def _cache_key(body: Dict[str, Any]) -> str:
    try:
        return json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        return repr(sorted(body.items()))


def _main_heatmap_block_rows(page_size: int) -> int:
    configured = int(HEATMAP_QUERY_API_PAGE_BLOCK_ROWS or 0)
    if configured <= page_size:
        return page_size
    if page_size > 100:
        return page_size
    return max(page_size, min(configured, page_size * 3))


def _main_histogram_block_rows(page_size: int) -> int:
    configured = int(HEATMAP_QUERY_API_HISTOGRAM_BLOCK_ROWS or 0)
    if configured <= page_size:
        return page_size
    if page_size > 100:
        return page_size
    return max(page_size, min(configured, page_size * 4))


def _prefetch_following_pages(operation: str, payload: Dict[str, Any], *, page: int, page_size: int) -> None:
    count = int(HEATMAP_QUERY_API_PREFETCH_PAGES or 0)
    if count <= 0 or HEATMAP_QUERY_API_CACHE_TTL <= 0 or page_size > 100:
        return
    for next_page in range(int(page) + 1, int(page) + count + 1):
        next_payload = copy.deepcopy(payload)
        next_payload["pagination"] = {"page": next_page, "page_size": int(page_size)}
        body = dict(next_payload)
        body["operation"] = operation
        cache_key = _cache_key(body)
        if cache_key in _API_CACHE or cache_key in _PREFETCH_INFLIGHT:
            continue
        _PREFETCH_INFLIGHT.add(cache_key)
        thread = threading.Thread(
            target=_prefetch_operation,
            args=(operation, next_payload, cache_key),
            daemon=True,
            name=f"main-visuals-prefetch-{operation}-{next_page}",
        )
        thread.start()


def _prefetch_operation(operation: str, payload: Dict[str, Any], cache_key: str) -> None:
    try:
        call_operation(operation, payload)
    except Exception as exc:
        if HEATMAP_QUERY_API_DEBUG:
            logger.warning("heatmap api prefetch operation=%s fallo: %s", operation, exc)
    finally:
        _PREFETCH_INFLIGHT.discard(cache_key)


def _wait_for_prefetch(cache_key: str) -> Optional[Dict[str, Any]]:
    if threading.current_thread().name.startswith("main-visuals-prefetch"):
        return None
    if cache_key not in _PREFETCH_INFLIGHT:
        return None
    deadline = time.time() + max(0.0, float(HEATMAP_QUERY_API_PREFETCH_WAIT_SECONDS or 0))
    while time.time() < deadline:
        cached = _API_CACHE.get(cache_key)
        if cached and (time.time() - cached["ts"] < HEATMAP_QUERY_API_CACHE_TTL):
            return copy.deepcopy(cached["data"])
        if cache_key not in _PREFETCH_INFLIGHT:
            break
        time.sleep(0.05)
    cached = _API_CACHE.get(cache_key)
    if cached and (time.time() - cached["ts"] < HEATMAP_QUERY_API_CACHE_TTL):
        return copy.deepcopy(cached["data"])
    return None


def _slice_visual_page(data: Dict[str, Any], *, requested_offset: int, requested_limit: int, block_offset: int) -> Dict[str, Any]:
    start = max(0, int(requested_offset) - int(block_offset))
    end = start + max(1, int(requested_limit))
    out = copy.deepcopy(data or {})
    for key in ("pct_payload", "unit_payload"):
        payload = out.get(key)
        if isinstance(payload, dict):
            out[key] = _slice_payload_rows(payload, start, end)
    info = dict(out.get("page_info") or {})
    showing = len(((out.get("pct_payload") or out.get("unit_payload") or {}).get("y")) or [])
    info["offset"] = int(requested_offset)
    info["limit"] = int(requested_limit)
    info["showing"] = showing
    out["page_info"] = info
    return out


def _slice_payload_rows(payload: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    row_keys = {
        "z",
        "z_raw",
        "y",
        "row_detail",
        "row_last_ts",
        "row_max_pct",
        "row_max_unit",
        "row_min_pct",
        "row_min_unit",
        "missing_mask",
        "traffic_raw",
    }
    out = dict(payload)
    original_details = payload.get("row_detail") or []
    kept_details = original_details[start:end] if isinstance(original_details, list) else []
    for key in row_keys:
        value = payload.get(key)
        if isinstance(value, list):
            out[key] = value[start:end]
    traffic_by_key = payload.get("traffic_by_key")
    if isinstance(traffic_by_key, dict) and kept_details:
        out["traffic_by_key"] = {detail: traffic_by_key.get(detail) for detail in kept_details if detail in traffic_by_key}
    return out
