@echo off
cd /d %~dp0

python -m venv venv
call venv\Scripts\activate

pip install Flask pandas numpy textblob joblib

pause