# Flask Sockpuppet Detection App

## Overview

This Flask application is designed to detect sockpuppet activities on Wikipedia by analyzing user-generated comments. Sockpuppets are online identities used deceptively within online communities to manipulate discussions or public opinion. The application employs a machine learning model, specifically a **RandomForestClassifier**, trained on a dataset of pre-processed text features, including **embeddings** and **sentiment analysis** scores.

## How It Works

1. **Model Loading:**
   - The application loads a pre-trained **RandomForestClassifier** model from disk (`random_forest.pkl`).
   - Feature metadata (`feature_metadata.pkl`) is loaded to retrieve feature names.
   - An **embeddings lookup table** (`embeddings_lookup.csv`) is loaded to provide precomputed embeddings for known texts.
2. **Data Processing & Feature Extraction:**
   - **Text Preprocessing:**
     - Converts text to lowercase.
     - Removes URLs, special characters, and non-alphanumeric symbols.
   - **Embedding Retrieval:**
     - If a match is found in `embeddings_lookup.csv`, the corresponding embedding is used.
     - If not found, **a random normal distribution embedding is generated** to simulate missing embeddings.
   - **Sentiment Analysis:**
     - Computes **polarity** (positive/negative tone) and **subjectivity** (factual vs. opinion-based content) using **TextBlob**.
   - The extracted features (embeddings + sentiment scores) are formatted into a structured input vector.
3. **Prediction:**
   - The processed text features are passed to the **RandomForestClassifier**.
   - The model predicts whether the input corresponds to a **sockpuppet** or **non-sockpuppet** account.
4. **Web Interface:**
   - The app provides a **Flask-based web UI** where users can submit text for analysis.
   - The prediction result is displayed on a new page (`result.html`).
   - The app automatically opens in the default web browser upon startup.

## Getting Started

### Prerequisites

Ensure the following dependencies are installed:

- **Python 3.x**: Required to run the application.
- **Flask**: Web framework for serving the application.
- **Pandas**: Used for handling and processing tabular data.
- **NumPy**: Supports large, multi-dimensional arrays and mathematical computations.
- **TextBlob**: Performs sentiment analysis, extracting polarity and subjectivity scores.
- **Scikit-learn**: Machine learning library for model operations.
- **Joblib**: Loads the pre-trained **RandomForestClassifier** model.
- **Pip**: Required to install dependencies.

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/zxyandreay/flask-sockpuppet-detection.git
   ```

2. Run the `setup.bat` script to create and activate a virtual environment, then install dependencies:

   ```batch
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
   ```

### Running the Application

1. Run the `start.bat` script to activate the virtual environment and start the Flask server:

   ```batch
   @echo off
   cd /d %~dp0
   call venv\Scripts\activate
   python app.py
   pause
   ```

2. The application will launch on `http://127.0.0.1:5000/`. Open the browser to interact with the web interface.

## Features

- **Text Preprocessing**: Cleans and normalizes text input.
- **Embeddings-Based Analysis**: Utilizes precomputed embeddings where available or generates new ones dynamically.
- **Sentiment Analysis**: Extracts **polarity** and **subjectivity** using TextBlob.
- **Machine Learning Detection**: Utilizes a **RandomForestClassifier** to predict sockpuppet activity.
- **Web-Based UI**: Flask-powered interface for easy interaction and testing.
- **Automated Browser Launch**: Opens the app in a web browser upon startup.

## API Endpoints

- `GET /` - Loads the application’s main interface.
- `POST /predict` - Accepts text input, processes it, and returns the **sockpuppet** classification result.

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, TextBlob
- **Data Handling**: Pandas, NumPy
- **Deployment**: Locally hosted via Flask (for testing)

## License

This project is licensed under the **MIT License**, allowing free use, modification, and distribution, provided that credit is given to the original authors.

**Note:** This project was developed as part of an **academic thesis requirement** and is not intended for commercial use or large-scale deployment.