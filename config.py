# Configuración de la base de datos
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345',
    'database': 'prueba',
    'port': 3306
}

# Configuración del dashboard
DASHBOARD_CONFIG = {
    'title': 'Dashboard Analytics',
    'theme': 'bootstrap'
}

# Configuración de umbrales para tablas
# Configuración de umbrales para telecom KPIs
THRESHOLDS = {
    'ps_failure_rrc_percent': {
        'critico': 2.0,
        'advertencia': 1.5,
        'ok': 0
    },
    'ps_failures_rab_percent': {
        'critico': 1.0,
        'advertencia': 0.8,
        'ok': 0
    },
    'cs_failures_rrc_percent': {
        'critico': 1.2,
        'advertencia': 0.9,
        'ok': 0
    },
    'cs_failures_rab_percent': {
        'critico': 0.8,
        'advertencia': 0.6,
        'ok': 0
    },
    'lcs_ps_rate': {
        'excelente': 99.0,
        'bueno': 98.0,
        'regular': 97.0
    },
    'lcs_cs_rate': {
        'excelente': 99.2,
        'bueno': 98.5,
        'regular': 97.5
    }
}

# Colores para los umbrales
COLORS = {
    'critico': '#ff4444',    # Rojo
    'advertencia': '#ffbb33', # Amarillo
    'ok': '#00C851',         # Verde
    'excelente': '#00C851',  # Verde
    'bueno': '#ffbb33',      # Amarillo
    'regular': '#ff4444'     # Rojo
}