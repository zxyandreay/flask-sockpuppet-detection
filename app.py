from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
from threading import Timer
import webbrowser
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask application
app = Flask(__name__, template_folder='html')

# Load training and testing data
train_data = pd.read_csv('data/wikipedia_sockpuppet_dataset_TRAIN.csv')
test_data = pd.read_csv('data/wikipedia_sockpuppet_dataset_TEST.csv')

# Initialize and fit TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=1000)
tfidf.fit(pd.concat([train_data['edit_text'], test_data['edit_text']]))

# Sentiment Analysis and feature preparation
train_data['polarity'] = train_data['edit_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
train_data['subjectivity'] = train_data['edit_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Combining features
X_train = np.hstack((tfidf.transform(train_data['edit_text']).toarray(), 
                     train_data[['polarity', 'subjectivity']].values))
y_train = train_data['is_sockpuppet']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Preprocess function
def preprocess(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|<.*?>', '', text)  # Remove text within brackets and HTML tags
    text = re.sub(r'\W|\d+', '', text)  # Remove non-alphanumeric characters and numbers
    return text.strip()

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

    prediction = model.predict(X_input)[0]
    result = 'sockpuppet' if prediction == 1 else 'non-sockpuppet'

    return render_template('result.html', prediction=result)

# Define route for opening the browser automatically
def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

# Run the Flask application
if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
