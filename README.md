# Dash Telecom KPIs Monitor (dashboard master)

Template de dashboard en Dash con conexión MySQL, filtros dinámicos y actualización periódica.

## Instalación
1. Crea y activa tu venv (opcional) e instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Copia `.env.example` a `.env` y ajusta credenciales de MySQL y zona horaria.
3. Crea la base y tabla con el script de `db/sample.sql`:
   ```sql
   SOURCE db/sample.sql;
   ```
4. Ejecuta la app:
   ```bash
   python app.py
   ```
   Abre http://127.0.0.1:8050

## Notas
- Intervalo de refresco configurable por `REFRESH_INTERVAL_MS` en `.env`
- Filtros predeterminados a la **fecha y hora local** definidos por `TZ` (America/Monterrey por defecto)
- Colores por umbrales y progress bars configurables en `src/utils.py` y `components/kpi_table.py`
