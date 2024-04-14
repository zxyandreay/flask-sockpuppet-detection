from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
import os
import sklearn  # For checking scikit-learn version
from textblob import TextBlob
import joblib  # Import joblib for loading the model and vectorizer
import logging

# Initialize Flask application and configure logging
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Check sklearn version
expected_sklearn_version = '0.24.1'  # example version, adjust as needed
if sklearn.__version__ != expected_sklearn_version:
    raise ImportError(f"Expected scikit-learn version {expected_sklearn_version}, got {sklearn.__version__}")

# Check if model files exist and load them
model_path = 'model/sockpuppet_model.pkl'
vectorizer_path = 'model/tfidf_vectorizer.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist.")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"The vectorizer file {vectorizer_path} does not exist.")

model = joblib.load(model_path)
tfidf = joblib.load(vectorizer_path)

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|https?://\S+|www\.\S+|<.*?>', '', text)
    text = re.sub(r'\W|\d+', ' ', text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form.get('input_text', None)
        if not input_text:
            raise ValueError("No input text provided")

        # Log the model details
        app.logger.debug(f"Model type: {type(model)}")
        app.logger.debug("Model attributes and methods:")
        app.logger.debug(dir(model))

        processed_text = preprocess(input_text)
        X_input_tfidf = tfidf.transform([processed_text]).toarray()
        polarity = TextBlob(processed_text).sentiment.polarity
        subjectivity = TextBlob(processed_text).sentiment.subjectivity
        X_input = np.hstack((X_input_tfidf, np.array([[polarity, subjectivity]])))

        prediction = model.predict(X_input)[0]
        result = 'sockpuppet' if prediction == 1 else 'non-sockpuppet'
    except Exception as e:
        app.logger.error(f'Error during prediction: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Ensure debug is False for production
