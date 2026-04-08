# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from dotenv import dotenv_values
from PyInstaller.utils.hooks import collect_all


PROJECT_ROOT = Path.cwd()
ENV_PATH = PROJECT_ROOT / ".env.sqlite"
env_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}

datas = [
    (str(PROJECT_ROOT / "assets"), "assets"),
    (str(PROJECT_ROOT / "data"), "data"),
]

sqlite_seed = env_values.get("SQLITE_PATH")
if sqlite_seed:
    sqlite_seed_path = Path(sqlite_seed).expanduser()
    if sqlite_seed_path.exists():
        datas.append((str(sqlite_seed_path), "data"))

hiddenimports = []
binaries = []

for package_name in ("dash", "plotly", "dash_bootstrap_components", "dash_ag_grid"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports


a = Analysis(
    ["launcher.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="TemplateDash-debug",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
