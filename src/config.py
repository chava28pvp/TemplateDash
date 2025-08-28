# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 1) Determina ENV (development por default)
ENV = os.getenv("ENV", "development")

# 2) Candidatos de ruta (prioridad: .env.test.<ENV> luego .env.test)
#    - CWD (desde donde ejecutas python)
#    - Carpeta del archivo actual (src/)
#    - Raíz del proyecto (dos niveles arriba de src/ por tu estructura)
HERE = Path(__file__).resolve()
FILE_DIR = HERE.parent                 # .../DashTest/src
PROJECT_ROOT = HERE.parents[1]         # .../DashTest  <-- raíz del proyecto
CWD = Path.cwd()                       # C:\Users\...\DashTest

candidates = [
    CWD / f".env.{ENV}",
    CWD / ".env",
    PROJECT_ROOT / f".env.{ENV}",
    PROJECT_ROOT / ".env",
    FILE_DIR / f".env.{ENV}",
    FILE_DIR / ".env",
]

loaded = None
for p in candidates:
    if p.exists():
        load_dotenv(p, override=True)
        loaded = p
        break

# Variables
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "prueba")

REFRESH_INTERVAL_MS = int(os.getenv("REFRESH_INTERVAL_MS", "60000"))
TZ = os.getenv("TZ", "America/Monterrey")

SQLALCHEMY_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
