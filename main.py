from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import joblib  # Import joblib for loading the model and vectorizer

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer from the 'model' folder
model = joblib.load('model/sockpuppet_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

def preprocess(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|https?://\S+|www\.\S+|<.*?>', '', text)
    text = re.sub(r'\W|\d+', '', text)  # Remove non-alphanumeric characters and numbers
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form['input_text']
        processed_text = preprocess(input_text)
        X_input_tfidf = tfidf.transform([processed_text]).toarray()
        polarity = TextBlob(processed_text).sentiment.polarity
        subjectivity = TextBlob(processed_text).sentiment.subjectivity
        X_input = np.hstack((X_input_tfidf, np.array([[polarity, subjectivity]])))

        prediction = model.predict(X_input)[0]
        result = 'sockpuppet' if prediction == 1 else 'non-sockpuppet'

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Use debug=False for production
