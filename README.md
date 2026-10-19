# Dashboard Master

Template de dashboard en Dash con soporte para MySQL o SQLite local.

## Exclusión temporal de la última muestra (Main API)

La fuente puede publicar una hora parcialmente cargada y completarla al publicar
la siguiente. El ajuste opcional `EXCLUDE_LATEST_SAMPLE` excluye la última
combinación **fecha + hora disponible en Main**, sin depender de la hora actual.
Si la última muestra es 08:00, esa muestra se excluye. Al aparecer 09:00,
se permite 08:00 y se excluye 09:00.

El filtro automático muestra la fecha y hora del reloj. La tabla y su contexto
consultan exactamente la fecha y hora seleccionadas, sin sustituirlas por una
muestra anterior. Si el filtro indica 14:00 y no hay datos permitidos para esa
hora, la tabla muestra "Sin datos". Las consultas históricas y la opción `Todas`
también conservan su selección y excluyen únicamente la última muestra.

El corte se aplica antes de calcular alarmas, ordenar y paginar: tabla principal,
heatmaps de Main e integridad, histogramas, máximos y exportaciones de Main.
Los filtros no cambian el corte global. Los días históricos anteriores se
conservan. Este interruptor corresponde a Main en modo API.

### Aplicar

1. Publica el contenido actualizado de `serverless/main_query.py` y
   `serverless/main_visuals_query.py` en las funciones de consulta correspondientes.
   Verifica el nombre de la función y el entorno en las URLs de `.env.api`:
   `MAIN_QUERY_API_URL` y `HEATMAP_QUERY_API_URL`. En esta configuración,
   `main_visuals_query.py` corresponde a la función **`heatmap_query`**, no a una
   función llamada `main_visuals_query`. Publicar solo en sandbox no actualiza
   las funciones consultadas por URLs que terminan en `/env/production`.
   No requiere cambios en las tablas ni en los procesos de carga.
2. Agrega `EXCLUDE_LATEST_SAMPLE=1` a `.env.api` y reinicia el dashboard:
   ```powershell
   $env:ENV="api"
   .\venv\Scripts\python.exe app.py
   ```
   Si tu Python activo ya tiene las dependencias del proyecto, puedes usar
   `py app.py`.

El interruptor está desactivado por defecto. La API debe confirmar el corte en
`meta.excluded_slot`; si todavía ejecuta una versión anterior, el cliente muestra
un error de compatibilidad en lugar de presentar datos parciales como filtrados.
Con el ajuste activo se usan los cálculos existentes sobre los datos consultados,
porque los rankings/series precalculados pueden incluir la muestra descartada.
Esto puede incrementar el tiempo de consulta si antes usabas esos precálculos.

### Revertir

Cambia a `EXCLUDE_LATEST_SAMPLE=0` en `.env.api` y reinicia. El comportamiento
anterior se restablece y las cachés en memoria se limpian. No es necesario
despublicar las funciones ni cambiar los umbrales.

### Verificar

Pruebas sin conexión a la fuente:
```powershell
$env:ENV="api"
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Comparación de las tres últimas muestras de la fuente original (solo lectura):
```powershell
$env:ENV="api"
.\venv\Scripts\python.exe tools/inspect_latest_sample.py
```

Diagnóstico de fechas cargadas, compatibilidad del filtro y baseline de
integridad para una muestra de combinaciones actuales (solo lectura):
```powershell
$env:ENV="api"
.\venv\Scripts\python.exe tools/diagnose_integrity.py
```

## Ejecucion local
1. Crea y activa tu venv e instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Si usaras SQLite, ajusta `.env.sqlite` y ejecuta:
   ```bash
   python app.py
   ```
3. La app abre en `http://127.0.0.1:8050`.

## Empaquetado Windows
1. Instala PyInstaller en tu entorno:
   ```bash
   pip install pyinstaller
   ```
2. Edita `.env.sqlite` y define temporalmente `SQLITE_PATH` con la ruta real de la base que quieres usar como semilla del ejecutable.
3. Construye el ejecutable:
   ```powershell
   .\build_windows.ps1
   ```
4. El resultado queda en `dist\TemplateDash.exe`.

## Comportamiento del ejecutable
- La SQLite final se usa desde `%LOCALAPPDATA%\TemplateDash\app.db`.
- Si `app.db` no existe, el ejecutable copia una base inicial incluida en el paquete.
- `umbrales.json` se guarda en `%LOCALAPPDATA%\TemplateDash\umbrales.json`.
- `assets` y `data` se incluyen en el bundle de PyInstaller.
