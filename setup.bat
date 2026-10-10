@echo off
REM One-time setup on Windows: creates a virtual environment and installs everything.
cd /d "%~dp0"
python --version >nul 2>&1 || (echo Python not found. Install it from python.org and tick "Add python.exe to PATH". & exit /b 1)
python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || exit /b 1
python -m pytest || exit /b 1
echo.
echo Setup complete. Now run:  run.bat all --quick
