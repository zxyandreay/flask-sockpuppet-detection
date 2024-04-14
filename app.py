from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
from threading import Timer
import webbrowser
from textblob import TextBlob
import joblib  # Import joblib for loading the model and vectorizer

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer from the 'model' folder
model = joblib.load('model/sockpuppet_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# Preprocess function (simplified without tokenization, stop words removal, and lemmatization)
def preprocess(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|https?://\S+|www\.\S+|<.*?>', '', text)  # Remove text within brackets and HTML tags
    text = re.sub(r'\W|\d+', '', text)  # Remove non-alphanumeric characters and numbers
    return text

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    processed_text = preprocess(input_text)
    X_input_tfidf = tfidf.transform([processed_text]).toarray()
    polarity = TextBlob(processed_text).sentiment.polarity
    subjectivity = TextBlob(processed_text).sentiment.subjectivity
    X_input = np.hstack((X_input_tfidf, np.array([[polarity, subjectivity]])))

    # Get the binary prediction for the class
    prediction = model.predict(X_input)[0]  # 0 for 'non-sockpuppet', 1 for 'sockpuppet'

    # Translate the binary result to a meaningful string
    result = 'sockpuppet' if prediction == 1 else 'non-sockpuppet'

    return render_template('result.html', prediction=result)

# Define route for browser
def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

# Run the Flask application
if __name__ == '__main__':
    Timer(1, open_browser).start()  # Wait a bit before opening the browser
    app.run(debug=True, use_reloader=False)  # Disable the reloader