from __future__ import annotations

from typing import Dict, Any

from src.Utils.umbrales.umbrales_manager import UM_MANAGER


def cell_severity(column: str, value) -> str:
    """Return one of: 'excelente' | 'bueno' | 'regular' | 'critico'.
    Falls back to 'bueno' for None to avoid over-highlighting.
    """
    if value is None:
        return "bueno"

    cfg = UM_MANAGER.get_severity(column)
    if not cfg:
        return "bueno"

    ori = cfg.get("orientation", "lower_is_better")
    thr: Dict[str, float] = cfg.get("thresholds", {})
    # sanitize
    e = float(thr.get("excelente", 0))
    b = float(thr.get("bueno", e))
    r = float(thr.get("regular", b))
    c = float(thr.get("critico", r))

    v = float(value)

    if ori == "higher_is_better":
        # thresholds are LOWER bounds per label (excelente ≥ bueno ≥ regular ≥ critico)
        if v >= e:
            return "excelente"
        if v >= b:
            return "bueno"
        if v >= r:
            return "regular"
        return "critico"
    else:
        # thresholds are UPPER bounds per label (excelente ≤ bueno ≤ regular ≤ critico)
        if v <= e:
            return "excelente"
        if v <= b:
            return "bueno"
        if v <= r:
            return "regular"
        return "critico"


def progress_cfg(column: str) -> Dict[str, Any]:
    cfg = UM_MANAGER.get_progress(column) or {}
    return {
        "min": float(cfg.get("min", 0.0)),
        "max": float(cfg.get("max", 100.0)),
        "decimals": int(cfg.get("decimals", 1)),
        "label": str(cfg.get("label", "{value:.1f}")),
    }