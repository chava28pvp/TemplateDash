# src/Utils/thresholds.py
import os
import json
from functools import lru_cache
import pandas as pd
from typing import Dict, Any, Optional

from src.Utils.umbrales.utils_umbrales import cell_severity

SEVERITY_METRICS = [
    "ps_rrc_ia_percent",
    "ps_rab_ia_percent",
    "ps_s1_ia_percent",
    "ps_drop_dc_percent",
    "cs_rrc_ia_percent",
    "cs_rab_ia_percent",
    "cs_drop_dc_percent",
]

# Claves que definen un "registro" para la racha
KEY_COLS = ["network", "vendor", "noc_cluster", "technology"]

# Columnas de tiempo para ordenar
TIME_COLS = ["fecha", "hora"]

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

def add_alarm_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade 2 columnas al DataFrame:
      - has_alarm: True/False si tiene al menos 1 KPI en nivel 'critico'
      - alarmas: racha de horas consecutivas con alarma (se reinicia a 0 cuando no hay alarma)

    Se asume que df tiene columnas:
      - fecha, hora
      - network, vendor, noc_cluster, technology
      - KPIs definidos en SEVERITY_METRICS
    """
    if df is None or df.empty:
        df = pd.DataFrame()
        df["has_alarm"] = []
        df["alarmas"] = []
        return df

    df = df.copy()

    # 1) bandera de alarma por fila
    def _row_has_alarm(row):
        net = row.get("network")
        for metric in SEVERITY_METRICS:
            if metric not in row:
                continue
            val = row[metric]
            if pd.isna(val):
                continue
            try:
                sev = cell_severity(metric, float(val), network=net, profile="main")
            except Exception:
                continue
            if sev == "critico":
                return True
        return False

    df["has_alarm"] = df.apply(_row_has_alarm, axis=1)

    # 2) ordenar por claves + tiempo
    sort_cols = [c for c in (KEY_COLS + TIME_COLS) if c in df.columns]
    df = df.sort_values(sort_cols)

    # 3) calcular racha dentro de cada grupo
    def _compute_group_streak(group: pd.DataFrame) -> pd.DataFrame:
        streak = 0
        out = []
        for flag in group["has_alarm"]:
            if flag:
                streak += 1
            else:
                streak = 0
            out.append(streak)
        group = group.copy()
        group["alarmas"] = out
        return group

    df = (
        df.groupby(KEY_COLS, group_keys=False)
          .apply(_compute_group_streak)
          .reset_index(drop=True)
    )

    return df