from __future__ import annotations
from typing import Dict, Any, Optional
import math

from src.Utils.umbrales.umbrales_manager import UM_MANAGER


def _is_missing_number(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    except Exception:
        return True


def cell_severity(column: str, value, network: Optional[str] = None) -> str:
    """
    Devuelve: 'excelente' | 'bueno' | 'regular' | 'critico'.
    - Soporta overrides por network si `UM_MANAGER` está en versión v2 (default/per_network).
    - Si `network` es None o no hay override, usa el default global.
    - Para valores faltantes, cae en 'bueno' (evita sobre-resaltar).

    Parámetros:
      column: base metric name, p.ej. 'ps_rrc_ia_percent'
      value:  numérico
      network: 'ATT' | 'NET' | 'TEF' | None
    """
    if _is_missing_number(value):
        return "bueno"

    # 1) Intenta config por network; si no hay, usa global
    cfg = UM_MANAGER.get_severity(column, network=network) or UM_MANAGER.get_severity(column)
    if not cfg:
        return "bueno"

    ori = cfg.get("orientation", "lower_is_better")
    thr: Dict[str, float] = cfg.get("thresholds", {}) or {}

    # Sanitizar/valores por defecto razonables
    e = float(thr.get("excelente", 0))
    b = float(thr.get("bueno", e))
    r = float(thr.get("regular", b))
    c = float(thr.get("critico", r))

    v = float(value)

    if ori == "higher_is_better":
        # Umbrales como LÍMITES INFERIORES por categoría
        # (excelente ≥ bueno ≥ regular ≥ …)
        if v >= e:
            return "excelente"
        if v >= b:
            return "bueno"
        if v >= r:
            return "regular"
        return "critico"
    else:
        # Umbrales como LÍMITES SUPERIORES por categoría
        # (excelente ≤ bueno ≤ regular ≤ …)
        if  b >= v <= e:
            return "excelente"
        if b <= v < r:
            return "bueno"
        if v >= r:
            return "regular"
        return "critico"


def progress_cfg(column: str, network: Optional[str] = None) -> Dict[str, Any]:
    """
    Devuelve configuración de progress bar (min/max/decimals/label),
    aplicando override por network si existe; si no, default global.
    """
    cfg = UM_MANAGER.get_progress(column, network=network) or UM_MANAGER.get_progress(column) or {}
    # Defaults seguros
    min_v = float(cfg.get("min", 0.0))
    max_v = float(cfg.get("max", 100.0))
    # Evitar rango degenerado
    if not (isinstance(min_v, float) and isinstance(max_v, float)) or max_v <= min_v:
        max_v = min_v + 1.0

    return {
        "min": min_v,
        "max": max_v,
        "decimals": int(cfg.get("decimals", 1)),
        "label": str(cfg.get("label", "{value:,.1f}")),
    }
