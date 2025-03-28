@echo off
cd /d "%~dp0"

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Create virtual environment if not exists
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
if exist requirements.txt (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
) else (
    echo [WARNING] requirements.txt not found. Ensure dependencies are installed manually.
)

:: Check if model files exist
if not exist model\random_forest.pkl (
    echo [ERROR] Model file not found: model\random_forest.pkl
    pause
    exit /b
)

if not exist model\feature_metadata.pkl (
    echo [ERROR] Feature metadata file not found: model\feature_metadata.pkl
    pause
    exit /b
)

:: Start the Flask app
python app.py

:: Deactivate virtual environment
deactivate