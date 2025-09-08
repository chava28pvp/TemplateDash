from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from typing import Any, Dict, List, Optional

from src.config import UMBRAL_JSON_PATH


# Defaults aligned with your table's BASE_* lists
SEVERITY_BASES: List[str] = [
    "ps_rrc_ia_percent",
    "ps_rab_ia_percent",
    "ps_s1_ia_percent",
    "ps_drop_dc_percent",
    "cs_rrc_ia_percent",
    "cs_rab_ia_percent",
    "cs_drop_dc_percent",
    # Optional: include "integrity" if you want it colored
]

PROGRESS_BASES: List[str] = [
    "ps_rrc_fail",
    "ps_rab_fail",
    "ps_s1_fail",
    "ps_drop_abnrel",
    "cs_rrc_fail",
    "cs_rab_fail",
    "cs_drop_abnrel",
]

# Fixed palette (UI shows these; CSS implements them)
SEVERITY_COLORS = {
    "excelente": "#2ecc71",  # green
    "bueno": "#f1c40f",      # yellow
    "regular": "#e67e22",    # orange
    "critico": "#e74c3c",    # red
}


def _orientation_for(metric: str) -> str:
    """Infer how to compare values.
    - "higher_is_better" for IA/Integrity-like metrics
    - "lower_is_better" for FAIL/DROP-like metrics
    """
    m = metric.lower()
    if ("_ia_percent" in m) or ("integrity" in m):
        return "higher_is_better"
    return "lower_is_better"


def _default_config() -> Dict[str, Any]:
    # For severity: store 4 numeric cutoffs. Semantics:
    #  - If orientation == lower_is_better, numbers are UPPER bounds per label (↑ ascending):
    #       v <= excelente → excelente
    #       v <= bueno     → bueno
    #       v <= regular   → regular
    #       else           → critico (uses its bound mainly for UI)
    #  - If orientation == higher_is_better, numbers are LOWER bounds per label (↓ descending):
    #       v >= excelente → excelente
    #       v >= bueno     → bueno
    #       v >= regular   → regular
    #       else           → critico
    severity = {}
    for name in SEVERITY_BASES:
        ori = _orientation_for(name)
        if ori == "higher_is_better":
            severity[name] = {
                "orientation": ori,
                "thresholds": {
                    "excelente": 99.5,
                    "bueno": 98.5,
                    "regular": 97.5,
                    "critico": 0.0,
                },
            }
        else:  # lower_is_better
            # Defaults close to your previous warn/bad examples
            base_max = 100.0
            severity[name] = {
                "orientation": ori,
                "thresholds": {
                    "excelente": 1.0,
                    "bueno": 2.0,
                    "regular": 3.0,
                    "critico": base_max,
                },
            }

    progress = {}
    for name in PROGRESS_BASES:
        progress[name] = {"min": 0, "max": 30, "decimals": 0, "label": "{value:.0f}"}

    return {
        "version": 1,
        "severity": severity,
        "progress": progress,
        "colors": SEVERITY_COLORS,
    }


@dataclass
class UmbralesManager:
    path: Path = field(default=UMBRAL_JSON_PATH)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _config: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.load()

    # -------- I/O --------
    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    self._config = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    self._config = _default_config()
            else:
                self._config = _default_config()
                self.save()  # create file on first run

    def save(self) -> None:
        with self._lock:
            self.path.write_text(json.dumps(self._config, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------- Queries --------
    def config(self) -> Dict[str, Any]:
        return self._config

    def list_metrics(self) -> List[str]:
        s = set(self._config.get("severity", {}).keys()) | set(self._config.get("progress", {}).keys())
        return sorted(s)

    def is_severity(self, metric: str) -> bool:
        return metric in self._config.get("severity", {})

    def is_progress(self, metric: str) -> bool:
        return metric in self._config.get("progress", {})

    def get_severity(self, metric: str, network: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raw = self._config.get("severity", {}).get(metric)
        if not raw:
            return None
        # v1 compat: si no tiene default/per_network, trátalo como default
        if "default" not in raw and "per_network" not in raw:
            return raw

        default = raw.get("default", {})
        if network:
            per_net = (raw.get("per_network") or {}).get(network) or {}
        else:
            per_net = {}

        # Orientación: si no se sobreescribe por network, usa la global
        out = {
            "orientation": per_net.get("orientation", default.get("orientation")),
            "thresholds": (default.get("thresholds") or {}).copy()
        }
        # Mezcla thresholds si vienen parciales por network
        out["thresholds"].update(per_net.get("thresholds") or {})
        return out

    def get_progress(self, metric: str, network: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raw = self._config.get("progress", {}).get(metric)
        if not raw:
            return None
        if "default" not in raw and "per_network" not in raw:
            return raw

        default = raw.get("default", {})
        per_net = (raw.get("per_network") or {}).get(network) or {}
        out = default.copy()
        out.update(per_net)  # min/max/decimals/label pueden venir parciales
        return out

    def upsert_severity(self, metric: str, *, thresholds: Dict[str, float],
                        network: Optional[str] = None, orientation: Optional[str] = None) -> None:
        self._config.setdefault("severity", {})
        entry = self._config["severity"].setdefault(metric, {})

        # Migración implícita si venía en formato v1
        if "default" not in entry and "per_network" not in entry:
            entry = {"default": entry, "per_network": {}}
            self._config["severity"][metric] = entry

        if network:
            entry.setdefault("per_network", {})
            tgt = entry["per_network"].setdefault(network, {})
            if orientation is not None:
                tgt["orientation"] = orientation
            tgt.setdefault("thresholds", {}).update(thresholds)
        else:
            entry.setdefault("default", {})
            tgt = entry["default"]
            if orientation is not None:
                tgt["orientation"] = orientation
            tgt.setdefault("thresholds", {}).update(thresholds)
        self.save()

    def upsert_progress(self, metric: str, *, min_v: float, max_v: float,
                        decimals: Optional[int] = None, label: Optional[str] = None,
                        network: Optional[str] = None) -> None:
        self._config.setdefault("progress", {})
        entry = self._config["progress"].setdefault(metric, {})

        if "default" not in entry and "per_network" not in entry:
            entry = {"default": entry, "per_network": {}}
            self._config["progress"][metric] = entry

        target = None
        if network:
            entry.setdefault("per_network", {})
            target = entry["per_network"].setdefault(network, {})
        else:
            entry.setdefault("default", {})
            target = entry["default"]

        target["min"] = min_v
        target["max"] = max_v
        if decimals is not None:
            target["decimals"] = decimals
        if label is not None:
            target["label"] = label
        self.save()

    # en UmbralesManager
    def clear_network_override(self, metric: str, kind: str, network: str) -> bool:
        """kind: 'severity' | 'progress'"""
        if not network:
            return False
        sec = self._config.get(kind, {})
        entry = sec.get(metric)
        if not entry:
            return False
        # v1 compat: si no tiene estructura v2, no hay override per_network
        per_net = (entry.get("per_network") or {})
        if network in per_net:
            per_net.pop(network, None)
            # limpia contenedor vacío
            if not per_net:
                entry["per_network"] = {}
            self.save()
            return True
        return False


# Singleton-like convenience
UM_MANAGER = UmbralesManager()