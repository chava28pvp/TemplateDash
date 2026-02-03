# src/Utils/umbrales/umbrales_manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from typing import Any, Dict, List, Optional
from copy import deepcopy

from src.config import UMBRAL_JSON_PATH

SEVERITY_BASES: List[str] = [
    "ps_rrc_ia_percent", "ps_rab_ia_percent", "ps_s1_ia_percent", "ps_drop_dc_percent",
    "cs_rrc_ia_percent", "cs_rab_ia_percent", "cs_drop_dc_percent",
    # 👇 métricas extra usadas en TopOff
    "rtx_tnl_tx_percent",
]
PROGRESS_BASES: List[str] = [
    "ps_rrc_fail", "ps_rab_fail", "ps_s1_fail", "ps_drop_abnrel",
    "cs_rrc_fail", "cs_rab_fail", "cs_drop_abnrel",
    # 👇 opcional si quieres configurar barras para TNL
    "tnl_fail", "tnl_abn",
]
SEVERITY_COLORS = {
    "excelente": "#2ecc71", "bueno": "#f1c40f", "regular": "#e67e22", "critico": "#e74c3c", "muy_critico": "#bd2130",
}

def _orientation_for(metric: str) -> str:
    m = metric.lower()
    if ("_ia_percent" in m) or ("integrity" in m) or ("_tx_percent" in m):
        return "higher_is_better"
    return "lower_is_better"

def _default_profile_block() -> Dict[str, Any]:
    # Igual que antes, pero SOLO el bloque del perfil (sin version/colors)
    severity: Dict[str, Any] = {}
    for name in SEVERITY_BASES:
        ori = _orientation_for(name)
        if ori == "higher_is_better":
            severity[name] = {"orientation": ori, "thresholds": {
                "excelente": 99.5, "bueno": 98.5, "regular": 97.5, "critico": 0.0}}
        else:
            severity[name] = {"orientation": ori, "thresholds": {
                "excelente": 1.0, "bueno": 2.0, "regular": 3.0, "critico": 100.0}}

    progress: Dict[str, Any] = {}
    for name in PROGRESS_BASES:
        progress[name] = {"min": 0, "max": 30, "decimals": 0, "label": "{value:.0f}"}
    return {"severity": severity, "progress": progress}

def _default_config_v2() -> Dict[str, Any]:
    # Dos perfiles iniciales: main/topoff
    return {
        "version": 2,
        "profiles": {
            "main": _default_profile_block(),
            "topoff": _default_profile_block(),
        },
        "colors": SEVERITY_COLORS,
    }

@dataclass
class UmbralesManager:
    path: Path = field(default=UMBRAL_JSON_PATH)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _config: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    self._config = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    self._config = _default_config_v2()
            else:
                self._config = _default_config_v2()
                self.save()

            # ---- Migración v1 → v2 (no había "profiles") ----
            if "profiles" not in self._config:
                # v1 tenía severity/progress en raíz
                old = deepcopy(self._config)
                profiles = {
                    "main": {"severity": old.get("severity", {}), "progress": old.get("progress", {})},
                    "topoff": {"severity": deepcopy(old.get("severity", {})),
                               "progress": deepcopy(old.get("progress", {}))},
                }
                self._config = {
                    "version": 2,
                    "profiles": profiles,
                    "colors": old.get("colors", SEVERITY_COLORS),
                }
                self.save()

    def save(self) -> None:
        with self._lock:
            self.path.write_text(json.dumps(self._config, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------- Helpers perfil --------
    def _ensure_profile(self, profile: Optional[str]) -> str:
        name = profile or "main"
        self._config.setdefault("profiles", {})
        if name not in self._config["profiles"]:
            self._config["profiles"][name] = _default_profile_block()
            self.save()
        return name

    def _profile_ref(self, profile: Optional[str]) -> Dict[str, Any]:
        name = self._ensure_profile(profile)
        return self._config["profiles"][name]

    # -------- API pública (con perfil) --------
    def config(self, profile: Optional[str] = None) -> Dict[str, Any]:
        if profile is None:
            return self._config
        return self._profile_ref(profile)

    def list_metrics(self, profile: Optional[str] = None) -> List[str]:
        if profile is None:
            # unión de todas las métricas de todos los perfiles (útil para el dropdown)
            allm: set = set()
            for p in (self._config.get("profiles") or {}).values():
                sev = p.get("severity", {})
                pro = p.get("progress", {})
                allm |= set(sev.keys()) | set(pro.keys())
            return sorted(allm)
        p = self._profile_ref(profile)
        return sorted(set(p.get("severity", {}).keys()) | set(p.get("progress", {}).keys()))

    def is_severity(self, metric: str, profile: Optional[str] = None) -> bool:
        return metric in self._profile_ref(profile).get("severity", {})

    def is_progress(self, metric: str, profile: Optional[str] = None) -> bool:
        return metric in self._profile_ref(profile).get("progress", {})

    def get_severity(self, metric: str, network: Optional[str] = None,
                     profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raw = self._profile_ref(profile).get("severity", {}).get(metric)
        if not raw:
            return None
        if "default" not in raw and "per_network" not in raw:
            return raw
        default = raw.get("default", {})
        per_net = (raw.get("per_network") or {}).get(network) or {} if network else {}
        out = {"orientation": per_net.get("orientation", default.get("orientation")),
               "thresholds": (default.get("thresholds") or {}).copy()}
        out["thresholds"].update(per_net.get("thresholds") or {})
        return out

    def get_progress(self, metric: str, network: Optional[str] = None,
                     profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raw = self._profile_ref(profile).get("progress", {}).get(metric)
        if not raw:
            return None
        if "default" not in raw and "per_network" not in raw:
            return raw
        default = raw.get("default", {})
        per_net = (raw.get("per_network") or {}).get(network) or {} if network else {}
        out = default.copy()
        out.update(per_net)
        return out

    def upsert_severity(self, metric: str, *, thresholds: Dict[str, float],
                        profile: Optional[str] = None, network: Optional[str] = None,
                        orientation: Optional[str] = None) -> None:
        prof = self._profile_ref(profile)
        prof.setdefault("severity", {})
        entry = prof["severity"].setdefault(metric, {})
        if "default" not in entry and "per_network" not in entry:
            entry = {"default": entry, "per_network": {}}
            prof["severity"][metric] = entry
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
                        profile: Optional[str] = None, network: Optional[str] = None,
                        decimals: Optional[int] = None, label: Optional[str] = None) -> None:
        prof = self._profile_ref(profile)
        prof.setdefault("progress", {})
        entry = prof["progress"].setdefault(metric, {})
        if "default" not in entry and "per_network" not in entry:
            entry = {"default": entry, "per_network": {}}
            prof["progress"][metric] = entry
        if network:
            entry.setdefault("per_network", {})
            tgt = entry["per_network"].setdefault(network, {})
        else:
            entry.setdefault("default", {})
            tgt = entry["default"]
        tgt["min"] = float(min_v)
        tgt["max"] = float(max_v)
        if decimals is not None:
            tgt["decimals"] = int(decimals)
        if label is not None:
            tgt["label"] = str(label)
        self.save()

    def clear_network_override(self, metric: str, kind: str, network: str,
                               profile: Optional[str] = None) -> bool:
        if not network:
            return False
        prof = self._profile_ref(profile)
        sec = prof.get(kind, {})
        entry = sec.get(metric)
        if not entry:
            return False
        per_net = (entry.get("per_network") or {})
        if network in per_net:
            per_net.pop(network, None)
            if not per_net:
                entry["per_network"] = {}
            self.save()
            return True
        return False

# Singleton
UM_MANAGER = UmbralesManager()
