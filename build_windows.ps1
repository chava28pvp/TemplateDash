$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot "venv\\Scripts\\python.exe"
$pythonCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

& $pythonCmd -m PyInstaller --clean --noconfirm TemplateDash.spec
