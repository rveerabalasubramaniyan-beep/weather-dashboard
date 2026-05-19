@echo off
setlocal

cd /d "%~dp0"

echo ----------------------------------------
echo Weather Dashboard - Flask Launcher
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

if not exist "model\weather_model.pkl" (
    echo Trained model not found. Please keep the model folder with the project.
    pause
    exit /b 1
)

echo.
echo Starting Flask app at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server.
echo.
python app.py

endlocal
