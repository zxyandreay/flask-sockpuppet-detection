from flask import Flask, request, render_template, jsonify
import webbrowser
import pandas as pd
import numpy as np
import re
from threading import Timer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
<<<<<<< HEAD
=======
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
>>>>>>> 81654a88 (update)

# Initialize Flask application
app = Flask(__name__)

# Load the dataset
DATA_PATH = 'data/wikipedia_sockpuppet_dataset_TRAIN.csv'
data = pd.read_csv(DATA_PATH)

# Preprocess function (simplified without tokenization, stop words removal, and lemmatization)
def preprocess(text):
<<<<<<< HEAD
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|https?://\S+|www\.\S+|<.*?>', '', text)  # Remove text within brackets and HTML tags
    text = re.sub(r'\W|\d+', ' ', text)  # Remove non-alphanumeric characters and numbers
    return text
=======
    # Convert text to lowercase
    text = text.lower()
    # Remove text within brackets and HTML tags
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|https?://\S+|www\.\S+|<.*?>', '', text)
    # Remove non-alphanumeric characters and numbers
    text = re.sub(r'\W|\d+', ' ', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Rejoin words into a cleaned string
    cleaned_text = ' '.join(tokens)
    return cleaned_text
>>>>>>> 81654a88 (update)

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
