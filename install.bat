@echo off
cd /d %~dp0

python -m venv venv
call venv\Scripts\activate

pip install Flask pandas numpy textblob scikit-learn nltk

python -c "import nltk; nltk.download('stopwords')"
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('wordnet')"

pause
