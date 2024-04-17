# Flask Sockpuppet Detection App

## Overview

This Flask application is designed to detect sockpuppet contributions on Wikipedia by analyzing user comments. A sockpuppet is an online identity used deceptively within online communities, often to manipulate discussions or opinions. The app leverages machine learning techniques, utilizing a RandomForestClassifier model that has been trained on a dataset of pre-processed text features derived from Wikipedia comments.

## Getting Started

### Prerequisites

- **Python 3.x**: Required to run the application.
- **Flask**: A lightweight WSGI web application framework used to serve the web application.
- **Pandas**: An open-source data analysis and manipulation tool, used for handling data operations.
- **NumPy**: A library for supporting large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions, essential for numerical operations in the application.
- **TextBlob**: Used for performing sentiment analysis, providing polarity and subjectivity scores for text input.
- **Scikit-learn**: A machine learning library used for model operations such as loading and predicting.
- **joblib**: For loading the pre-trained RandomForestClassifier model and the TF-IDF vectorizer from disk.
- **Pip package manager**: Needed to install the dependencies.
### Installation

1. Clone the repository to your local machine:
```batch
git clone https://github.com/zxyandreay/flask-sockpuppet-detection.git
```
2. Run `setup.bat` to create and activate a virtual environment, and install all the necessary prerequisites:
```batch
@echo off
cd /d %~dp0

python -m venv venv
call venv\Scripts\activate

pip install Flask pandas numpy textblob joblib scikit-learn

pause
```
### Running the Application

1. Run the `start.bat` script to activate the virtual environment and start the application. This script also automatically opens the application in your default web browser:
```batch
@echo off
cd /d %~dp0
call venv\Scripts\activate
python app.py
pause
```
2. Please wait as it may take a few moments for the application to launch and the browser to open automatically. The application will be accessible at `http://127.0.0.1:5000/`.

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
