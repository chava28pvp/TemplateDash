import json
import os
from pathlib import Path


class UmbralesManager:
    def __init__(self, config_path="data/umbrales_config.json"):
        self.config_path = config_path
        self.default_config = self._load_default_config()
        self.user_config = self._load_user_config()

    def _load_default_config(self):
        """Carga la configuración por defecto"""
        default_path = Path(__file__).parent / "default_umbrales.json"
        with open(default_path, 'r') as f:
            return json.load(f)

    def _load_user_config(self):
        """Carga la configuración del usuario o crea una nueva"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except:
                return self.default_config.copy()
        return self.default_config.copy()

    def save_config(self):
        """Guarda la configuración actual"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.user_config, f, indent=2)

    def get_umbral(self, metric_name):
        """Obtiene configuración de una métrica específica"""
        return self.user_config['umbrales'].get(metric_name,
                                                self.default_config['umbrales'].get(metric_name))

    def update_umbral(self, metric_name, new_config):
        """Actualiza la configuración de una métrica"""
        if metric_name in self.user_config['umbrales']:
            self.user_config['umbrales'][metric_name] = new_config
            self.save_config()

    def reset_to_default(self, metric_name=None):
        """Restablece a valores por defecto"""
        if metric_name:
            if metric_name in self.default_config['umbrales']:
                self.user_config['umbrales'][metric_name] = self.default_config['umbrales'][metric_name].copy()
        else:
            self.user_config = self.default_config.copy()
        self.save_config()

    def get_all_metrics(self):
        """Obtiene todas las métricas configurables"""
        return list(self.user_config['umbrales'].keys())

    def get_cell_severity(self, metric_name, value):
        """Determina la severidad basada en los umbrales"""
        config = self.get_umbral(metric_name)
        if not config or value is None:
            return "neutral"

        for nivel in config['niveles']:
            if value >= nivel['limite']:
                return nivel['nombre'].lower()
        return "critical"

    def get_progress_cfg(self, metric_name):
        """Obtiene configuración para progress bars"""
        config = self.get_umbral(metric_name)
        if config:
            return {
                "min": config.get('min', 0),
                "max": config.get('max', 100),
                "decimals": config.get('decimals', 1),
                "label": config.get('label', "{value:.1f}")
            }
        return {"min": 0, "max": 100, "decimals": 1, "label": "{value:.1f}"}