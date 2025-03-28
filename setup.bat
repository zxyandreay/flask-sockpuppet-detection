@echo off
cd /d %~dp0
echo [INFO] Setting up virtual environment...

:: Remove existing virtual environment if it exists
if exist venv rmdir /s /q venv

:: Create a new virtual environment
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b
)

:: Activate virtual environment
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b
)

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo [INFO] Installing required packages...
pip install --no-cache-dir Flask pandas numpy textblob joblib scikit-learn > install_log.txt 2>&1

if errorlevel 1 (
    echo [ERROR] Package installation failed! Check install_log.txt for details.
    pause
    exit /b
)

echo [SUCCESS] Setup complete! Virtual environment and dependencies are ready.
pause