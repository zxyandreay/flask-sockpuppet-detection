@echo off
cd /d %~dp0

echo Creating a Virtual Environment...
python -m venv venv

call venv\Scripts\activate

echo Installing Flask...
pip install Flask

echo Installing pandas...
pip install pandas

echo Installing numpy...
pip install numpy

echo Installing textblob...
pip install textblob

echo Installing scikit-learn...
pip install scikit-learn

echo All required packages have been installed.
pause
