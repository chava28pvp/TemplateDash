import mysql.connector
from mysql.connector import Error
import pandas as pd
from config import DB_CONFIG


class MySQLConnection:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**DB_CONFIG)
            print("✅ Conexión a MySQL establecida correctamente")
        except Error as e:
            print(f"❌ Error conectando a MySQL: {e}")
            raise e

    def execute_query(self, query, params=None):
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            cursor.close()
            return result
        except Error as e:
            print(f"❌ Error ejecutando query: {e}")
            self.connect()  # Intentar reconectar
            return None

    def get_dataframe(self, query, params=None):
        try:
            df = pd.read_sql(query, self.connection, params=params)
            return df
        except Error as e:
            print(f"❌ Error obteniendo DataFrame: {e}")
            self.connect()  # Intentar reconectar
            return pd.DataFrame()

    def close(self):
        if self.connection:
            self.connection.close()
            print("🔌 Conexión a MySQL cerrada")


# Instancia global de conexión
db_connection = MySQLConnection()


# Funciones de consulta específicas para telecom KPIs
def get_telecom_data(start_date=None, end_date=None, regions=None):
    """Obtiene datos de telecom_kpis con filtros opcionales"""
    query = """
    SELECT fecha, region, 
           total_mbytes_nocperf, delta_total_mbytes_nocperf,
           ps_failure_rrc_percent, ps_failures_rab_percent,
           lcs_ps_rate, ps_abnormal_releases,
           total_erlangs_nocperf, delta_total_erlangs_nocperf,
           cs_failures_rrc_percent, cs_failures_rab_percent,
           lcs_cs_rate, cs_abnormal_releases,
           traffic_gb_att, delta_traffic_gb_att,
           traffic_amr_att, delta_traffic_amr_att
    FROM telecom_kpis 
    WHERE 1=1
    """

    params = []

    # Filtro por fecha
    if start_date:
        query += " AND fecha >= %s"
        params.append(start_date)
    if end_date:
        query += " AND fecha <= %s"
        params.append(end_date)

    # Filtro por regiones
    if regions and len(regions) > 0:
        placeholders = ', '.join(['%s'] * len(regions))
        query += f" AND region IN ({placeholders})"
        params.extend(regions)

    query += " ORDER BY fecha DESC, region"

    print(f"📋 Ejecutando query: {query}")
    print(f"📊 Parámetros: {params}")

    df = db_connection.get_dataframe(query, params)
    print(f"✅ Datos obtenidos: {len(df)} registros")

    return df


def get_available_regions():
    """Obtiene las regiones disponibles en la base de datos"""
    query = "SELECT DISTINCT region FROM telecom_kpis ORDER BY region"
    result = db_connection.execute_query(query)
    regions = [item['region'] for item in result] if result else []
    print(f"🌍 Regiones disponibles: {regions}")
    return regions


def get_date_range():
    """Obtiene el rango de fechas disponible en la base de datos"""
    query = "SELECT MIN(fecha) as min_date, MAX(fecha) as max_date FROM telecom_kpis"
    result = db_connection.execute_query(query)
    if result and len(result) > 0:
        min_date = result[0]['min_date']
        max_date = result[0]['max_date']
        print(f"📅 Rango de fechas: {min_date} to {max_date}")
        return min_date, max_date
    return None, None


def get_kpi_summary():
    """Resumen de KPIs por región"""
    query = """
    SELECT 
        region,
        AVG(ps_failure_rrc_percent) as avg_ps_rrc_failure,
        AVG(ps_failures_rab_percent) as avg_ps_rab_failure,
        AVG(lcs_ps_rate) as avg_lcs_ps,
        AVG(cs_failures_rrc_percent) as avg_cs_rrc_failure,
        AVG(cs_failures_rab_percent) as avg_cs_rab_failure,
        AVG(lcs_cs_rate) as avg_lcs_cs,
        SUM(total_mbytes_nocperf) as total_data_traffic,
        SUM(total_erlangs_nocperf) as total_voice_traffic
    FROM telecom_kpis 
    GROUP BY region
    """
    return db_connection.get_dataframe(query)