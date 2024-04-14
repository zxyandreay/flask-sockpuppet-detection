@echo off
cd /d %~dp0

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install Flask pandas numpy textblob joblib gunicorn

echo Setup completed successfully!
pause