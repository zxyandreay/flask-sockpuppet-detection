# Flask Sockpuppet Detection App

## Overview

This Flask application detects sockpuppet activities on Wikipedia by analyzing user-generated comments. Sockpuppets are online identities used deceptively to manipulate discussions or public opinion. The application employs a machine learning model, specifically a **RandomForestClassifier**, trained on a dataset of pre-processed text features, including **S-BERT embeddings** and **sentiment analysis** scores.

## Table of Contents

- How It Works
- Getting Started
- Features
- API Endpoints
- Example Usage
- Technology Stack
- License

## How It Works

1. **Model Loading**
   - Loads a pre-trained **RandomForestClassifier** model from disk (`random_forest.pkl`).
   - Retrieves feature names from `feature_metadata.pkl`.
   - Uses an **embeddings lookup table** (`embeddings_lookup.csv`) for precomputed text embeddings.
   - Dynamically generates embeddings using **Sentence-BERT (S-BERT)** when needed.
2. **Data Processing & Feature Extraction**
   - **Text Preprocessing**
     - Converts text to lowercase.
     - Removes URLs, special characters, and non-alphanumeric symbols.
   - **Embedding Retrieval**
     - If a match exists in `embeddings_lookup.csv`, the corresponding embedding is used.
     - Otherwise, generates **new embeddings dynamically** using **S-BERT** (`all-MiniLM-L6-v2`).
   - **Sentiment Analysis**
     - Computes **polarity** (positive/negative sentiment) and **subjectivity** (factual vs. opinion-based content) using **TextBlob**.
   - Extracted features (embeddings + sentiment scores) are formatted into a structured input vector.
3. **Prediction**
   - Processed text features are passed to the **RandomForestClassifier**.
   - The model predicts whether the input corresponds to a **sockpuppet** or **non-sockpuppet** account.
4. **Web Interface**
   - Provides a **Flask-based web UI** for users to submit text for analysis.
   - Displays the prediction result on `result.html`.
   - Automatically launches in the default web browser upon startup.

## Getting Started

### Prerequisites

Ensure the following dependencies are installed:

- **Python 3.x**: Required to run the application.
- **Flask**: Web framework for serving the application.
- **Pandas**: Handles and processes tabular data.
- **NumPy**: Supports large, multi-dimensional arrays and mathematical computations.
- **TextBlob**: Performs sentiment analysis, extracting polarity and subjectivity scores.
- **Scikit-learn**: Machine learning library for model operations.
- **Joblib**: Loads the pre-trained **RandomForestClassifier** model.
- **Sentence-Transformers**: Provides **S-BERT embeddings** for text analysis.
- **Pip**: Required to install dependencies.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zxyandreay/flask-sockpuppet-detection.git
   ```

2. Run the `setup.bat` script to set up the environment:

   ```batch
   @echo off
   cd /d %~dp0
   echo [INFO] Setting up virtual environment...
   
   if exist venv rmdir /s /q venv
   python -m venv venv
   call venv\Scripts\activate
   
   echo [INFO] Upgrading pip and installing dependencies...
   python -m pip install --upgrade pip
   pip install Flask pandas numpy textblob joblib scikit-learn sentence-transformers
   
   echo [SUCCESS] Setup complete!
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

2. The application will launch on `http://127.0.0.1:5000/`. Open a browser to interact with the web interface.

## Features

- **Text Preprocessing**: Cleans and normalizes text input.
- **S-BERT Embeddings**: Generates embeddings dynamically for unseen text.
- **Sentiment Analysis**: Extracts **polarity** and **subjectivity** using TextBlob.
- **Machine Learning Detection**: Utilizes a **RandomForestClassifier** to predict sockpuppet activity.
- **Web-Based UI**: Flask-powered interface for easy interaction and testing.
- **Automated Browser Launch**: Opens the app in a web browser upon startup.

## API Endpoints

- `GET /` - Loads the application’s main interface.
- `POST /predict` - Accepts text input, processes it, and returns the **sockpuppet** classification result.

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, TextBlob, Sentence-BERT
- **Data Handling**: Pandas, NumPy
- **Deployment**: Locally hosted via Flask (for testing)

## License

This project is licensed under the **MIT License**, allowing free use, modification, and distribution, provided that credit is given to the original authors.

**Note:** This project was developed as part of an **academic thesis requirement** and is not intended for commercial use or large-scale deployment.