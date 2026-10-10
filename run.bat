@echo off
REM Run any PHYVAL command inside the virtual environment, e.g.
REM   run.bat all --quick
REM   run.bat run-build --fault cfo_uncorrected --severity 0.6 --budget 0.3
cd /d "%~dp0"
if not exist .venv (echo Run setup.bat first & exit /b 1)
call .venv\Scripts\activate.bat
if "%~1"=="" (python -m phyval all) else (python -m phyval %*)
