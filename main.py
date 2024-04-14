from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib  # Import joblib for loading the model and vectorizer

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load('model/sockpuppet_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

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
    input_text = request.form['input_text']
    processed_text = preprocess(input_text)
    X_input_tfidf = tfidf.transform([processed_text]).toarray()
    polarity = TextBlob(processed_text).sentiment.polarity
    subjectivity = TextBlob(processed_text).sentiment.subjectivity
    X_input = np.hstack((X_input_tfidf, np.array([[polarity, subjectivity]])))

    prediction = model.predict(X_input)[0]
    result = 'sockpuppet' if prediction == 1 else 'non-sockpuppet'
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Change debug to False for production
