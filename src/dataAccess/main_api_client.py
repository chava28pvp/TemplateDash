import logging
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests
import urllib3

from src.config import (
    MAIN_QUERY_API_CA_BUNDLE,
    MAIN_QUERY_API_DEBUG,
    MAIN_QUERY_API_FORCE_RUNTIME_SORT,
    MAIN_QUERY_API_MAX_PAGES,
    MAIN_QUERY_API_PAGE_SIZE,
    MAIN_QUERY_API_SCAN_MAX_ROWS,
    MAIN_QUERY_API_TIMEOUT,
    MAIN_QUERY_API_TOKEN,
    MAIN_QUERY_API_TOKEN_HEADER,
    MAIN_QUERY_API_TOKEN_PREFIX,
    MAIN_QUERY_API_URL,
    MAIN_QUERY_API_VERIFY_SSL,
)

logger = logging.getLogger(__name__)


class MainApiError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(MAIN_QUERY_API_URL)


def call_operation(operation: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not MAIN_QUERY_API_URL:
        raise MainApiError("MAIN_QUERY_API_URL no esta configurado.")

    body = dict(payload or {})
    body["operation"] = operation

    headers = _auth_headers()
    verify = _verify_setting()
    if verify is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if MAIN_QUERY_API_DEBUG:
        logger.warning("main api request operation=%s payload_keys=%s", operation, sorted(body.keys()))

    try:
        response = requests.post(
            MAIN_QUERY_API_URL,
            json=body,
            headers=headers,
            timeout=MAIN_QUERY_API_TIMEOUT,
            verify=verify,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MainApiError(f"Error llamando main API operation={operation}: {exc}") from exc

    try:
        raw = response.json()
    except ValueError as exc:
        raise MainApiError(f"Respuesta no JSON de main API operation={operation}: {response.text[:500]}") from exc

    data = _unwrap_platform_response(raw)
    if not data.get("ok", data.get("success", False)):
        message = ((data.get("error") or {}).get("message")) or str(data.get("error") or data)
        raise MainApiError(f"main API operation={operation} fallo: {message}")
    return data


def fetch_page(
    *,
    mode: str,
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
    columns=None,
    thresholds_snapshot=None,
):
    payload = _base_payload(
        fecha=fecha,
        hora=hora,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
        columns=columns,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["mode"] = mode
    payload["pagination"] = {"page": int(page or 1), "page_size": int(page_size or 50)}
    payload["sort"] = {
        "mode": mode,
        "column": sort_by_friendly,
        "sort_net": sort_net,
        "ascending": bool(ascending),
    }
    payload["options"] = {
        "na_as_empty": bool(na_as_empty),
        "force_in_memory_sort": bool(thresholds_snapshot and MAIN_QUERY_API_FORCE_RUNTIME_SORT),
        "max_rows": int(MAIN_QUERY_API_SCAN_MAX_ROWS or 200000),
    }
    if not MAIN_QUERY_API_FORCE_RUNTIME_SORT:
        payload.pop("thresholds_snapshot", None)

    data = call_operation("table_page", payload)
    df = rows_to_frame(data.get("rows") or [])
    df = enrich_integrity_health_pct(df, fecha=fecha)
    return df, int(data.get("total") or 0)


def enrich_integrity_health_pct(df: pd.DataFrame, *, fecha=None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "integrity_deg_pct" in df.columns and df["integrity_deg_pct"].notna().any():
        return df
    required = {"fecha", "hora", "vendor", "noc_cluster", "technology", "network", "integrity"}
    if not required.issubset(df.columns):
        return df

    row_keys = (
        df[["fecha", "hora", "vendor", "noc_cluster", "technology", "network"]]
        .drop_duplicates()
        .to_dict("records")
    )
    if not row_keys:
        return df

    baseline_map = fetch_integrity_baseline_for_keys(fecha=fecha or df["fecha"].max(), row_keys=row_keys)
    if not baseline_map:
        return df

    out = df.copy()

    def _pct(row):
        key = (row.get("network"), row.get("vendor"), row.get("noc_cluster"), row.get("technology"))
        baseline = baseline_map.get(key)
        integrity = row.get("integrity")
        try:
            if baseline is None or float(baseline) <= 0 or pd.isna(integrity):
                return None
            return max(0.0, min(100.0, (float(integrity) / float(baseline)) * 100.0))
        except Exception:
            return None

    out["integrity_deg_pct"] = out.apply(_pct, axis=1)
    return out


def fetch_integrity_baseline_for_keys(*, fecha, row_keys):
    if not fecha or not row_keys:
        return {}
    narrowed = _filters_from_row_keys(row_keys)
    narrowed_payload = _base_payload(
        fecha=fecha,
        vendors=narrowed.get("vendors"),
        clusters=narrowed.get("clusters"),
        networks=narrowed.get("networks"),
        technologies=narrowed.get("technologies"),
    )
    narrowed_payload["options"] = {"baseline_max_rows": 50000}
    try:
        data = call_operation("integrity_baseline_week", narrowed_payload)
        rows = data.get("rows") or data.get("data") or []
        out = {}
        for item in rows:
            key = (item.get("network"), item.get("vendor"), item.get("noc_cluster"), item.get("technology"))
            baseline = item.get("integrity_week_avg")
            if baseline is not None:
                out[key] = baseline
        if out:
            return out
    except MainApiError as exc:
        logger.warning("No se pudo cargar baseline semanal acotado via API: %s", exc)

    payload = {
        "view": "main",
        "fecha": fecha,
        "row_keys": list(row_keys),
        "pagination": {"page": 1, "page_size": max(1, len(row_keys))},
        "options": {
            "include_total": False,
            "include_integrity_baseline": True,
            "preview_baseline_max_rows": 50000,
        },
    }
    try:
        data = call_operation("computed_preview", payload)
    except MainApiError as exc:
        logger.warning("No se pudo cargar baseline por pagina via API: %s", exc)
        return {}

    out = {}
    for row in data.get("rows") or []:
        key = (row.get("network"), row.get("vendor"), row.get("noc_cluster"), row.get("technology"))
        baseline = row.get("Integrity_Baseline_Debug")
        if baseline is not None:
            out[key] = baseline
    return out


def _filters_from_row_keys(row_keys):
    out = {"vendors": set(), "clusters": set(), "technologies": set(), "networks": set()}
    # row_keys do not include network, so the caller may pass only key columns.
    # The fallback query is still much smaller when constrained by vendor/cluster/technology.
    for key in row_keys or []:
        vendor = key.get("vendor")
        cluster = key.get("noc_cluster")
        tech = key.get("technology")
        network = key.get("network")
        if vendor:
            out["vendors"].add(vendor)
        if cluster:
            out["clusters"].add(cluster)
        if tech:
            out["technologies"].add(tech)
        if network:
            out["networks"].add(network)
    return {k: sorted(v) for k, v in out.items() if v}


def fetch_rows(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    limit=None,
    na_as_empty=False,
    columns=None,
    thresholds_snapshot=None,
):
    page_size = max(1, min(int(limit or MAIN_QUERY_API_PAGE_SIZE), MAIN_QUERY_API_PAGE_SIZE))
    max_pages = 1 if limit else max(1, int(MAIN_QUERY_API_MAX_PAGES or 1))
    rows = []

    for page in range(1, max_pages + 1):
        df, total = fetch_page(
            mode="recent",
            fecha=fecha,
            hora=hora,
            vendors=vendors,
            clusters=clusters,
            networks=networks,
            technologies=technologies,
            page=page,
            page_size=page_size,
            na_as_empty=na_as_empty,
            columns=columns,
            thresholds_snapshot=thresholds_snapshot,
        )
        if not df.empty:
            rows.extend(df.to_dict("records"))
        if limit and len(rows) >= int(limit):
            rows = rows[: int(limit)]
            break
        if df.empty or len(rows) >= total:
            break

    return rows_to_frame(rows)


def fetch_by_keys(
    *,
    row_keys,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    na_as_empty=False,
    columns=None,
    thresholds_snapshot=None,
):
    payload = _base_payload(
        fecha=fecha,
        hora=hora,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
        columns=columns,
        thresholds_snapshot=thresholds_snapshot,
    )
    payload["row_keys"] = list(row_keys or [])
    payload["options"] = {"na_as_empty": bool(na_as_empty)}
    data = call_operation("by_keys", payload)
    return rows_to_frame(data.get("rows") or [])


def fetch_distinct_catalogs(*, fecha=None, hora=None, networks=None, technologies=None):
    payload = _base_payload(
        fecha=fecha,
        hora=hora,
        networks=networks,
        technologies=technologies,
    )
    data = call_operation("distinct_catalogs", payload)
    return data.get("data") or {}


def fetch_latest_slot():
    data = call_operation("latest_slot", {})
    return data.get("data")


def sync_thresholds(*, thresholds, profile="main", config_hash=None, updated_at=None):
    payload = {
        "profile": profile,
        "thresholds": thresholds,
    }
    if config_hash:
        payload["config_hash"] = config_hash
    if updated_at:
        payload["updated_at"] = updated_at
    data = call_operation("sync_thresholds", payload)
    return data.get("data") or {}


def fetch_thresholds(*, profile="main"):
    data = call_operation("get_thresholds", {"profile": profile})
    return data.get("data")


def fetch_integrity_baseline_week(*, fecha, vendors=None, clusters=None, networks=None, technologies=None):
    payload = _base_payload(
        fecha=fecha,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
    )
    data = call_operation("integrity_baseline_week", payload)
    return rows_to_frame(data.get("rows") or data.get("data") or [])


def fetch_progress_max_by_network(*, fecha=None, hora=None, vendors=None, clusters=None, networks=None, technologies=None):
    payload = _base_payload(
        fecha=fecha,
        hora=hora,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
    )
    data = call_operation("progress_max_by_network", payload)
    return data.get("data") or {}


def fetch_alarm_state(*, fecha, vendors=None, clusters=None, networks=None, technologies=None, thresholds_snapshot=None):
    payload = _base_payload(
        fecha=fecha,
        vendors=vendors,
        clusters=clusters,
        networks=networks,
        technologies=technologies,
        thresholds_snapshot=thresholds_snapshot,
    )
    data = call_operation("alarm_state", payload)
    return rows_to_frame(data.get("rows") or data.get("data") or [])


def rows_to_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows or []))


def _base_payload(
    *,
    fecha=None,
    hora=None,
    vendors=None,
    clusters=None,
    networks=None,
    technologies=None,
    columns=None,
    thresholds_snapshot=None,
):
    payload = {
        "view": "main",
        "fecha": fecha,
        "hora": hora,
        "filters": {
            "fecha": fecha,
            "hora": hora,
            "vendors": _as_list(vendors),
            "clusters": _as_list(clusters),
            "networks": _as_list(networks),
            "technologies": _as_list(technologies),
        },
    }
    if columns:
        payload["columns"] = list(columns)
    if thresholds_snapshot:
        payload["thresholds_snapshot"] = thresholds_snapshot
    return payload


def _auth_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MAIN_QUERY_API_TOKEN_HEADER and MAIN_QUERY_API_TOKEN:
        token_value = MAIN_QUERY_API_TOKEN
        if MAIN_QUERY_API_TOKEN_PREFIX:
            token_value = f"{MAIN_QUERY_API_TOKEN_PREFIX} {token_value}"
        headers[MAIN_QUERY_API_TOKEN_HEADER] = token_value
    return headers


def _verify_setting():
    if MAIN_QUERY_API_CA_BUNDLE:
        return MAIN_QUERY_API_CA_BUNDLE
    return bool(MAIN_QUERY_API_VERIFY_SSL)


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
