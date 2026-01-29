# src/Utils/umbrales/utils_umbrales.py
from __future__ import annotations
from typing import Dict, Any, Optional
import math
from src.Utils.umbrales.umbrales_manager import UM_MANAGER

def _is_missing_number(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    except Exception:
        return True

def cell_severity(column: str, value, network: Optional[str] = None,
                  profile: Optional[str] = None) -> str:
    if _is_missing_number(value):
        return "bueno"

    cfg = (UM_MANAGER.get_severity(column, network=network, profile=profile)
           or UM_MANAGER.get_severity(column, profile=profile)
           or UM_MANAGER.get_severity(column))
    if not cfg:
        return "bueno"

    ori = cfg.get("orientation", "lower_is_better")
    thr: Dict[str, float] = cfg.get("thresholds", {}) or {}

    e = float(thr.get("excelente", 0))
    b = float(thr.get("bueno", e))
    r = float(thr.get("regular", b))
    c = float(thr.get("critico", r))
    mc = thr.get("muy_critico", None)  # <-- nuevo (opcional)

    v = round(float(value), 1)
    eps = 1e-9

    if ori == "higher_is_better":
        # (si algún día usas muy_critico aquí, normalmente sería “muy bajo es peor”)
        if v >= e: return "excelente"
        if v >= b: return "bueno"
        if v >= r: return "regular"
        # si definiste 'critico'/'muy_critico' para higher_is_better, los podrías usar,
        # pero por ahora dejamos el fallback:
        return "critico"

    # lower_is_better
    if v <= e + eps: return "excelente"
    if v <= b + eps: return "bueno"

    # aquí es donde metemos el escalón extra
    if mc is not None:
        mc = float(mc)
        if v < c - eps:      # < 5
            return "regular"
        if v < mc - eps:     # 5 <= v < 10
            return "critico"
        return "muy_critico" # >= 10

    # comportamiento original (4 niveles)
    if v < c - eps:
        return "regular"
    return "critico"


def progress_cfg(column: str, network: Optional[str] = None,
                 profile: Optional[str] = None) -> Dict[str, Any]:
    cfg = (UM_MANAGER.get_progress(column, network=network, profile=profile)
           or UM_MANAGER.get_progress(column, profile=profile)
           or UM_MANAGER.get_progress(column) or {})
    min_v = float(cfg.get("min", 0.0))
    max_v = float(cfg.get("max", 100.0))
    if not (isinstance(min_v, float) and isinstance(max_v, float)) or max_v <= min_v:
        max_v = min_v + 1.0
    return {
        "min": min_v,
        "max": max_v,
        "decimals": int(cfg.get("decimals", 1)),
        "label": str(cfg.get("label", "{value:,.1f}")),
    }
