import requests
import pandas as pd
import logging

from src.Utils.alarmados import load_threshold_cfg
from src.config import (
    MAIN_QUERY_API_CA_BUNDLE,
    MAIN_QUERY_API_DEBUG,
    MAIN_QUERY_API_MAX_PAGES,
    MAIN_QUERY_API_PAGE_SIZE,
    MAIN_QUERY_API_TIMEOUT,
    MAIN_QUERY_API_TOKEN,
    MAIN_QUERY_API_TOKEN_HEADER,
    MAIN_QUERY_API_TOKEN_PREFIX,
    MAIN_QUERY_API_URL,
    MAIN_QUERY_API_VERIFY_SSL,
)
from src.dataAccess.data_access import BASE_COLUMNS, COLMAP


class MainApiAccessError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


def _headers():
    headers = {"Content-Type": "application/json"}
    if MAIN_QUERY_API_TOKEN:
        token = MAIN_QUERY_API_TOKEN
        if MAIN_QUERY_API_TOKEN_PREFIX:
            token = f"{MAIN_QUERY_API_TOKEN_PREFIX} {token}"
        headers[MAIN_QUERY_API_TOKEN_HEADER] = token
    return headers


def _verify():
    if MAIN_QUERY_API_CA_BUNDLE:
        return MAIN_QUERY_API_CA_BUNDLE
    return MAIN_QUERY_API_VERIFY_SSL


def _request(operation, **payload):
    if not MAIN_QUERY_API_URL:
        raise MainApiAccessError("MAIN_QUERY_API_URL no esta configurado.")

    body = {"view": "main", "operation": operation}
    body.update({k: v for k, v in payload.items() if v is not None})
    _debug_log_request(operation, body)

    response = requests.post(
        MAIN_QUERY_API_URL,
        headers=_headers(),
        json=body,
        timeout=MAIN_QUERY_API_TIMEOUT,
        verify=_verify(),
    )
    if response.status_code >= 400:
        raise MainApiAccessError(f"Error HTTP {response.status_code} consultando API main.")

    try:
        data = response.json()
    except ValueError as exc:
        raise MainApiAccessError("La API no regreso JSON valido.") from exc

    if data.get("success") is False or data.get("ok") is False:
        err = data.get("error")
        if isinstance(err, dict):
            message = err.get("message") or str(err)
        else:
            message = err or "Error consultando API main."
        raise MainApiAccessError(message)

    data = _unwrap_gateway_response(data)
    _debug_log_response(operation, data)

    if MAIN_QUERY_API_DEBUG:
        print(f"[main_api] operation={operation} keys={list(data.keys())}")
    return data


def _debug_log_request(operation, body):
    if not MAIN_QUERY_API_DEBUG:
        return
    safe_body = dict(body)
    safe_body.pop("thresholds_snapshot", None)
    logger.warning("[main_api] request operation=%s payload=%s", operation, safe_body)


def _debug_log_response(operation, data):
    if not MAIN_QUERY_API_DEBUG:
        return
    rows = _extract_rows_from_payload(data)
    logger.warning(
        "[main_api] response operation=%s success=%s total=%s rows=%s data_type=%s sample=%s",
        operation,
        data.get("success") if isinstance(data, dict) else None,
        data.get("total") if isinstance(data, dict) else None,
        len(rows),
        type(data.get("data")).__name__ if isinstance(data, dict) else type(data).__name__,
        rows[:1],
    )


def _unwrap_gateway_response(data):
    if not isinstance(data, dict):
        return data
    inner = data.get("data")
    if (
        isinstance(inner, dict)
        and ("success" in inner or "ok" in inner or "operation" in inner)
    ):
        if inner.get("success") is False or inner.get("ok") is False:
            err = inner.get("error")
            if isinstance(err, dict):
                message = err.get("message") or str(err)
            else:
                message = err or "Error consultando API main."
            raise MainApiAccessError(message)
        return inner
    return data


def _filters(vendors=None, clusters=None, networks=None, technologies=None):
    return {
        "vendors": _as_list(vendors),
        "clusters": _as_list(clusters),
        "networks": _as_list(networks),
        "technologies": _as_list(technologies),
    }


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "")]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return [value]


def _rows_to_df(rows, columns=None, na_as_empty=False):
    df = pd.DataFrame(rows or [])
    if columns:
        existing = [c for c in columns if c in df.columns]
        df = df.reindex(columns=existing)
    if na_as_empty and not df.empty:
        df = df.where(pd.notna(df), "")
    return df


def _thresholds_snapshot():
    try:
        return load_threshold_cfg()
    except Exception:
        return None


def fetch_latest_available_slot():
    response = _request("latest_slot")
    data = response.get("data")
    if isinstance(data, dict):
        return data if data.get("fecha") and data.get("hora") else None
    if response.get("fecha") and response.get("hora"):
        return {"fecha": response.get("fecha"), "hora": response.get("hora")}
    return None


def fetch_main_distinct_catalogs(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    response = _request(
        "distinct_catalogs",
        fecha=fecha,
        hora=hora,
        filters=_filters(vendors, clusters, networks, technologies),
    )
    return response.get("data") or {}


def fetch_kpis_paginated_severity_sort(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page=1,
    page_size=50,
    na_as_empty=False,
):
    response = _request(
        "page",
        mode="alarmado",
        fecha=fecha,
        hora=hora,
        filters=_filters(vendors, clusters, networks, technologies),
        pagination={"page": page, "page_size": page_size},
        sort={"column": None, "ascending": False},
        options={
            "include_total": True,
            "na_as_empty": na_as_empty,
            "page_size": MAIN_QUERY_API_PAGE_SIZE,
            "max_pages": MAIN_QUERY_API_MAX_PAGES,
        },
        thresholds_snapshot=_thresholds_snapshot(),
    )
    rows = _extract_rows(response)
    return _rows_to_df(rows, BASE_COLUMNS, na_as_empty), int(response.get("total") or len(rows))


def fetch_kpis_paginated_severity_global_sort(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    page=1,
    page_size=50,
    sort_by_friendly=None,
    sort_net=None,
    ascending=True,
    na_as_empty=False,
):
    response = _request(
        "page",
        mode="global",
        fecha=fecha,
        hora=hora,
        filters=_filters(vendors, clusters, networks, technologies),
        pagination={"page": page, "page_size": page_size},
        sort={
            "column": sort_by_friendly,
            "network": sort_net,
            "ascending": bool(ascending),
        },
        options={
            "include_total": True,
            "na_as_empty": na_as_empty,
            "page_size": MAIN_QUERY_API_PAGE_SIZE,
            "max_pages": MAIN_QUERY_API_MAX_PAGES,
        },
        thresholds_snapshot=_thresholds_snapshot(),
    )
    rows = _extract_rows(response)
    return _rows_to_df(rows, BASE_COLUMNS, na_as_empty), int(response.get("total") or len(rows))


def fetch_integrity_baseline_week(
    *,
    fecha,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    response = _request(
        "integrity_baseline_week",
        fecha=fecha,
        filters=_filters(vendors, clusters, networks, technologies),
    )
    return _rows_to_df(_extract_rows_or_data(response))


def fetch_progress_max_by_network(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    response = _request(
        "progress_max_by_network",
        fecha=fecha,
        hora=hora,
        filters=_filters(vendors, clusters, networks, technologies),
    )
    return response.get("data") or {}


def fetch_main_alarm_state(
    *,
    fecha,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    response = _request(
        "alarm_state",
        fecha=fecha,
        filters=_filters(vendors, clusters, networks, technologies),
        thresholds_snapshot=_thresholds_snapshot(),
    )
    return _rows_to_df(_extract_rows(response))


def fetch_alarm_meta_for_heatmap(
    *,
    fecha,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
):
    response = _request(
        "alarm_meta",
        fecha=fecha,
        filters=_filters(vendors, clusters, networks, technologies),
        thresholds_snapshot=_thresholds_snapshot(),
    )
    rows = _extract_rows(response)
    df = _rows_to_df(rows)
    raw_keys = response.get("alarm_keys") or []
    alarm_keys_set = {
        (
            item.get("technology"),
            item.get("vendor"),
            item.get("noc_cluster"),
            item.get("network"),
        )
        for item in raw_keys
        if isinstance(item, dict)
    }
    if df.empty:
        df = pd.DataFrame(columns=["technology", "vendor", "noc_cluster"])
    return df, alarm_keys_set


def fetch_kpis_by_keys(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    row_keys=None,
    na_as_empty=False,
):
    response = _request(
        "by_keys",
        fecha=fecha,
        hora=hora,
        filters=_filters(vendors, clusters, networks, technologies),
        row_keys=row_keys or [],
        options={"na_as_empty": na_as_empty},
    )
    return _rows_to_df(_extract_rows(response), BASE_COLUMNS, na_as_empty)


def fetch_kpis(
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    limit=None,
    na_as_empty=False,
    columns=None,
):
    page_size = limit or MAIN_QUERY_API_PAGE_SIZE
    response = _request(
        "page",
        mode="recent",
        fecha=fecha,
        hora=hora,
        filters=_filters(vendors, clusters, networks, technologies),
        pagination={"page": 1, "page_size": page_size},
        columns=columns or BASE_COLUMNS,
        options={"include_total": False, "na_as_empty": na_as_empty},
    )
    return _rows_to_df(_extract_rows(response), columns or BASE_COLUMNS, na_as_empty)


def _extract_rows(response):
    return _extract_rows_from_payload(response)


def _extract_rows_from_payload(response):
    if not isinstance(response, dict):
        return []
    if isinstance(response.get("rows"), list):
        return response.get("rows") or []
    data = response.get("data")
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data.get("rows") or []
    if isinstance(data, list):
        return data
    return []


def _extract_rows_or_data(response):
    rows = _extract_rows(response)
    if rows:
        return rows
    data = response.get("data")
    return data if isinstance(data, list) else []
