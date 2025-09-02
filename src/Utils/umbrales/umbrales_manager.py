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

    def get_severity(self, metric: str) -> Optional[Dict[str, Any]]:
        return self._config.get("severity", {}).get(metric)

    def get_progress(self, metric: str) -> Optional[Dict[str, Any]]:
        return self._config.get("progress", {}).get(metric)

    def upsert_severity(self, metric: str, *, thresholds: Dict[str, float]) -> None:
        ori = _orientation_for(metric)
        self._config.setdefault("severity", {})[metric] = {
            "orientation": ori,
            "thresholds": thresholds,
        }
        self.save()

    def upsert_progress(self, metric: str, *, min_v: float, max_v: float, decimals: int = 0, label: str = "{value:.0f}") -> None:
        self._config.setdefault("progress", {})[metric] = {
            "min": min_v,
            "max": max_v,
            "decimals": decimals,
            "label": label,
        }
        self.save()


# Singleton-like convenience
UM_MANAGER = UmbralesManager()