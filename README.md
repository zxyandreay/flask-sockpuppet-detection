# Flask Sockpuppet Detection App

## Overview
This Flask application analyzes and predicts whether a given text input is a "sockpuppet" or not based on text features. A sockpuppet is an online identity used for purposes of deception within online communities. The app uses machine learning for text analysis, employing a RandomForestClassifier model trained on a pre-processed dataset.

## Getting Started

### Prerequisites

- **Python 3.x**: Required to run the application.
- **Flask**: A lightweight WSGI web application framework used to serve the web application.
- **Pandas**: An open-source data analysis and manipulation tool, used for handling data operations.
- **NumPy**: A library for supporting large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions, essential for numerical operations in the application.
- **TextBlob**: Used for performing sentiment analysis, providing polarity and subjectivity scores for text input.
- **joblib**: For loading the pre-trained RandomForestClassifier model and the TF-IDF vectorizer from disk.
- **Pip package manager**: Needed to install the dependencies.
### Installation

Download and run the `setup.bat` script, which will clone the repository and install all necessary dependencies:

```batch
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
```
### Running the Application
Use `start.bat` to activate the virtual environment and automatically run the application. The application will also automatically open your default web browser to `http://127.0.0.1:5000/`:

```batch
@echo off
cd /d %~dp0
call venv\Scripts\activate
python app.py
pause
```
## Features
- Text preprocessing including tokenization, stop words removal, and lemmatization.
- Text analysis using TF-IDF vectorization.
- Sentiment analysis to extract polarity and subjectivity features.
- Sockpuppet prediction using RandomForestClassifier.
- Web interface for easy interaction with the model.

## Technology Stack
- Flask: Web framework for serving the application.
- Pandas and NumPy: For data manipulation and numerical operations.
- NLTK: For natural language processing, including text preprocessing.
- TextBlob: For performing sentiment analysis.
- Scikit-learn: For machine learning model building and vectorization.

## License
Specify the license under which the project is released.
