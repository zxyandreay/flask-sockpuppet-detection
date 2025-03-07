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
    ```bash
    git clone https://github.com/zxyandreay/flask-sockpuppet-detection.git
    ```
2. Run `setup.bat` to create and activate a virtual environment, and install all the necessary prerequisites:
    ```batch
    @echo off
    cd /d %~dp0
    echo [INFO] Setting up virtual environment...
    
    :: Delete existing virtual environment if it exists
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
    
    :: Upgrade pip to the latest version
    echo [INFO] Upgrading pip...
    python -m pip install --upgrade pip
    
    :: Install dependencies with --no-cache-dir to avoid corruption issues
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

- Text analysis using TF-IDF vectorization.
- Sentiment analysis to extract polarity and subjectivity features.
- Sockpuppet prediction using RandomForestClassifier.
- Web interface for easy interaction with the model.

## Technology Stack

- Flask: Web framework for serving the application.
- Pandas and NumPy: For data manipulation and numerical operations.
- TextBlob: For performing sentiment analysis.
- Scikit-learn: For machine learning model building and vectorization.

## License

The application is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, provided that the following conditions are met:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- The software is provided "as is", without any warranty of any kind.


**Note:** This project is a student project made for an academic thesis requirement.