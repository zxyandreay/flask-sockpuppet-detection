@echo off
echo Cloning repository...
md flask-sockpuppet-detection
cd flask-sockpuppet-detection
git clone https://github.com/zxyandreay/flask-sockpuppet-detection.git .
echo Repository cloned!

echo Setting up virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing dependencies...
pip install Flask pandas numpy textblob joblib scikit-learn

echo Setup complete!
pause