from flask import Flask, request, render_template
import webbrowser
import numpy as np
import re
from threading import Timer
from textblob import TextBlob
from joblib import load

# Initialize Flask application
app = Flask(__name__, template_folder='html')

# Load the trained model and TF-IDF vectorizer
model = load('model/model.joblib')
tfidf = load('model/tfidf_vectorizer.joblib')

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|\=.*?\=', '', text)
    text = re.sub(r'\W', ' ', text)
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

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
