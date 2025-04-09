# Flask Sockpuppet Detection App

## Overview

This Flask web application detects potential **sockpuppet accounts**—deceptive online identities used to manipulate discussions—based on the textual content of user comments. The system uses **machine learning** with a focus on **semantic embeddings** and **sentiment analysis** to determine whether an input text likely originates from a sockpuppet account.

## How It Works

1. **Model & Metadata Initialization**
   - Loads a pre-trained **RandomForestClassifier** model from `model/random_forest.pkl`.
   - Loads a list of input **feature names** from `feature_metadata.pkl`.
   - Optionally loads a **lookup table** (`embeddings_lookup.csv`) of precomputed **Sentence-BERT embeddings** for known inputs.
   - Loads a **pre-trained PCA model** (`pca_model.pkl`) for reducing embedding dimensionality.
2. **Input Handling & Preprocessing**
   - User input is captured from the web form (`input_text`) or as a JSON POST request.
   - Text undergoes **cleaning**, including:
     - Stripping URLs
     - Removing special characters
     - Collapsing extra whitespace
   - If the cleaned input is empty (e.g. just symbols or whitespace), an error message is returned.
3. **Embedding Retrieval**
   - If the input text exists in the precomputed **embeddings lookup table**, it reuses the stored vector.
   - Otherwise, it generates a fresh embedding using **Sentence-BERT (paraphrase-MiniLM-L6-v2)**.
   - The embedding is:
     - Averaged across multiple runs (default: 3) for stability
     - Normalized to unit length
     - Transformed via PCA if available
4. **Sentiment Feature Extraction**
   - Using **TextBlob**, the following sentiment metrics are extracted:
     - **Polarity** (positive/negative tone)
     - **Subjectivity** (factual vs. opinion-based)
5. **Feature Vector Assembly**
   - The final feature vector is constructed to match the structure expected by the model:
     - Embedding components (e.g., `embedding_0`, `embedding_1`, ...)
     - Sentiment scores (`polarity`, `subjectivity`)
6. **Prediction & Output**
   - The assembled feature vector is passed into the **RandomForestClassifier**.
   - The model outputs probabilities for **sockpuppet** and **non-sockpuppet**.
   - The result and its confidence level are displayed on the `result.html` page and on the terminal.
7. **Web Interface**
   - A simple, intuitive **Flask-based UI** allows users to submit text.
   - The interface automatically launches in the default browser on app startup.

## Getting Started

### Prerequisites

Ensure the following dependencies are installed:

- **Python 3.x**: Required to run the application.
- **Flask**: Web framework used to serve the interface and handle routing.
- **Pandas**: Handles structured data, especially for embedding lookups and feature construction.
- **NumPy**: Supports numerical operations and vector transformations.
- **TextBlob**: Performs sentiment analysis, extracting polarity and subjectivity scores from input text.
- **Scikit-learn**: Powers the machine learning pipeline and PCA transformation.
- **Joblib**: Loads the pre-trained Random Forest and PCA models.
- **Sentence-Transformers**: Generates Sentence-BERT embeddings for input text.
- **Pip**: Required to install all dependencies.
- **Standard Libraries**: `re`, `threading`, and `webbrowser` (these come with Python and do not need separate installation).

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/zxyandreay/flask-sockpuppet-detection.git
   ```
   
2. Run the `setup.bat` script to set up the environment:

   ```
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

   ```
   @echo off
   cd /d %~dp0
   call venv\Scripts\activate
   python app.py
   pause
   ```

2. The application will launch on `http://127.0.0.1:5000/`. Open a browser to interact with the web interface.

## Features

- 🧼 **Text Cleaning**: Removes URLs, non-letter characters, and excess whitespace.
- 🧠 **Dynamic Embeddings**: Uses **Sentence-BERT** with optional averaging to encode input meaning.
- 💡 **Sentiment Analysis**: Adds interpretability via **TextBlob**’s polarity and subjectivity metrics.
- 📉 **PCA Compression**: Reduces dimensionality of embeddings for model efficiency.
- 🔍 **Sockpuppet Detection**: Predicts likelihood using a pre-trained **Random Forest** classifier.
- 🌐 **User-Friendly Web UI**: Flask-based frontend with automatic browser launch.
- ⚙️ **Fallback Mechanisms**: Embedding generation activates only if precomputed vectors are unavailable.

## API Endpoints

| Method | Route      | Description                               |
| ------ | ---------- | ----------------------------------------- |
| GET    | `/`        | Loads the main UI for user interaction    |
| POST   | `/predict` | Accepts user input and returns prediction |

## Technology Stack

- **Backend**: Python, Flask
- **ML Model**: Scikit-learn (Random Forest), Sentence-BERT, PCA
- **NLP Tools**: TextBlob (sentiment), Sentence-Transformers
- **Frontend**: HTML (served by Flask)
- **Utilities**: NumPy, Pandas, Joblib

## License

This project is licensed under the **MIT License**, allowing free use, modification, and distribution, provided that credit is given to the original authors.

**Note:** This project was developed as part of an **academic thesis requirement** and is not intended for commercial use or large-scale deployment.