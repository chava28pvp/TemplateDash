"""One source watermark shared by Main tables, charts and alarm queries.

The watermark advances only when latest_slot is refreshed. No clock arithmetic
or per-filter maximum is used: ingestion can lag or skip hours.
"""
from datetime import datetime
import os
from threading import RLock

from src.config import DATA_SOURCE

EXCLUDE_LATEST_SAMPLE = os.getenv("EXCLUDE_LATEST_SAMPLE", "0").strip().lower() in {
    "1", "true", "yes", "on",
}

_lock = RLock()
_initialized = False
_excluded_slot = None


def enabled():
    return DATA_SOURCE == "api" and EXCLUDE_LATEST_SAMPLE


def refresh_excluded_slot():
    global _initialized, _excluded_slot
    from src.dataAccess import main_api_client

    with _lock:
        response = main_api_client.call_operation(
            "latest_slot", {}, apply_sample_policy=False,
        )
        slot = response.get("data")
        if slot:
            stamp = datetime.fromisoformat(f"{slot['fecha']} {slot['hora']}")
            slot = {"fecha": stamp.strftime("%Y-%m-%d"), "hora": stamp.strftime("%H:%M:%S")}
        _excluded_slot = slot
        _initialized = True
        return dict(slot) if slot else None


def get_excluded_slot():
    with _lock:
        if not _initialized:
            return refresh_excluded_slot()
        return dict(_excluded_slot) if _excluded_slot else None


def prepare_read_payload(payload):
    body = dict(payload or {})
    if enabled():
        slot = get_excluded_slot()
        if slot:
            body["excluded_slot"] = slot
    return body


def cache_token():
    if not enabled():
        return None
    slot = get_excluded_slot() or {}
    return slot.get("fecha"), slot.get("hora")


def require_policy_support(body, response, error_type):
    slot = body.get("excluded_slot")
    if slot and (response.get("meta") or {}).get("excluded_slot") != slot:
        raise error_type(
            "La API no confirmó la exclusión de la última muestra. "
            "Actualiza main_query y main_visuals_query antes de activar EXCLUDE_LATEST_SAMPLE."
        )
