# src/Utils/thresholds.py
import os
import json
from functools import lru_cache
from typing import Dict, Any, Optional

def load_threshold_cfg(path: str = "data/umbrales.json") -> Dict[str, Any]:
    """
    Carga con hot-reload: invalida cache automáticamente si cambia el mtime.
    """
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de umbrales: {path}")
    return _load_threshold_cfg_cached(path, mtime)

@lru_cache(maxsize=32)
def _load_threshold_cfg_cached(path: str, _mtime: float) -> Dict[str, Any]:
    # _mtime solo sirve para invalidar el cache cuando el archivo cambia.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clear_threshold_cache() -> None:
    """Limpia manualmente el cache (por si agregas un botón de recarga)."""
    _load_threshold_cfg_cached.cache_clear()

def _get_threshold(kpi: str, network: str, cfg: Dict[str, Any], kind: str) -> Optional[float]:
    """
    kind: 'alarm' | 'excess_base'
    Fallbacks:
      per_network[kind] → default[kind] → per_network['max'] → default['max']
      (si nada existe, retorna None)
    """
    prog = (cfg.get("progress") or {}).get(kpi) or {}
    per_net = (prog.get("per_network") or {}).get(network or "") or {}
    default = prog.get("default") or {}

    # prioridad explícita por tipo
    for block in (per_net, default):
        if kind in block:
            return float(block[kind])

    # fallback razonable a 'max'
    for block in (per_net, default):
        if "max" in block:
            return float(block["max"])

    return None

def alarm_threshold_for(kpi: str, network: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[float]:
    cfg = cfg or load_threshold_cfg()
    return _get_threshold(kpi, network, cfg, kind="alarm")

def excess_base_for(kpi: str, network: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[float]:
    cfg = cfg or load_threshold_cfg()
    # si no hay 'excess_base', caerá a 'max' (o lo que exista) vía _get_threshold
    return _get_threshold(kpi, network, cfg, kind="excess_base")
