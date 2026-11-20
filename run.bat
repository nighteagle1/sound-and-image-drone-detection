@echo off
rem Запуск обнаружения дронов на Windows: run.bat [--host 0.0.0.0] [--port N] [--no-browser]
chcp 65001 >nul
cd /d "%~dp0"

if not exist .venv (
    py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Нужен Python 3.11 или новее.
        exit /b 1
    )
)
".venv\Scripts\python" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo Нужен Python 3.11 или новее. Удалите папку .venv и установите подходящий Python.
    exit /b 1
)
".venv\Scripts\python" -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 exit /b 1

set PYTHONPATH=%~dp0src
".venv\Scripts\python" -m dronedetect %*
