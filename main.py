from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Initialize Flask application
app = Flask(__name__)

# Load the dataset
DATA_PATH = 'data/wikipedia_sockpuppet_dataset_TRAIN.csv'
data = pd.read_csv(DATA_PATH)

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|https?://\S+|www\.\S+|<.*?>', '', text)
    text = re.sub(r'\W|\d+', ' ', text)
    return text

# Preprocessing the data
data['processed_edit_text'] = data['edit_text'].apply(preprocess)

# Text analysis and feature engineering
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(data['processed_edit_text']).toarray()
data['polarity'] = data['processed_edit_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['subjectivity'] = data['processed_edit_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
X = np.hstack((X_tfidf, data[['polarity', 'subjectivity']].values))

# Handling the target variable
imputer = SimpleImputer(strategy='median')
y = data['is_sockpuppet'].values
y = pd.Series(imputer.fit_transform(y.reshape(-1, 1)).ravel())

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

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
