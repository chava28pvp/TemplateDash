# Dash Telecom KPIs Monitor

Template de dashboard en Dash con soporte para MySQL o SQLite local.

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
