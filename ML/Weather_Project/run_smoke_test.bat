@echo off
setlocal

cd /d "%~dp0"

echo ----------------------------------------
echo Weather Dashboard - Model Smoke Test
echo ----------------------------------------
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate the virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
pip install -r requirements-runtime.txt

echo.
python test_model.py
echo.
pause

endlocal
