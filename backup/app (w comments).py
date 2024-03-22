# Import necessary modules from Flask for web application handling
from flask import Flask, request, render_template, jsonify
# Module to open web browser programmatically
import webbrowser
# pandas for data manipulation and CSV file reading
import pandas as pd
# numpy for numerical operations and array handling
import numpy as np
# Timer for scheduling tasks, used here to delay browser opening
from threading import Timer
# TextBlob for natural language processing, used for sentiment analysis
from textblob import TextBlob
# TF-IDF vectorizer for text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
# RandomForestClassifier for building the machine learning model
from sklearn.ensemble import RandomForestClassifier
# SimpleImputer for handling missing data
from sklearn.impute import SimpleImputer
# train_test_split for splitting the data into training and test sets
from sklearn.model_selection import train_test_split

# Create a new Flask web application instance
app = Flask(__name__)

# Define the path to the dataset file
DATA_PATH = 'data/wikipedia_sockpuppet_dataset.csv'
# Read the dataset into a pandas DataFrame
data = pd.read_csv(DATA_PATH)

# Function to preprocess the text data
def preprocess(text):
    # Convert text to lowercase or handle missing values
    if isinstance(text, str):
        return text.lower()
    elif pd.isna(text):
        return ""  # Return empty string for missing values
    else:
        return str(text).lower()  # Ensure all data is string and lowercase

# Apply the preprocessing function to the 'edit_text' column of the dataset
data['edit_text'] = data['edit_text'].apply(preprocess)
# Store the processed text in a new column
data['processed_edit_text'] = data['edit_text']

# Initialize the TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=1000)
# Transform the processed text into TF-IDF feature vectors
X_tfidf = tfidf.fit_transform(data['processed_edit_text']).toarray()
# Compute sentiment polarity and subjectivity using TextBlob
data['polarity'] = data['processed_edit_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['subjectivity'] = data['processed_edit_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
# Combine TF-IDF features with sentiment analysis features
X = np.hstack((X_tfidf, data[['polarity', 'subjectivity']].values))

# Initialize the imputer for handling missing values in the target variable
imputer = SimpleImputer(strategy='median')
# Extract the target variable 'is_sockpuppet' from the dataset
y = data['is_sockpuppet'].values
# Apply imputation to the target variable
y = pd.Series(imputer.fit_transform(y.reshape(-1, 1)).ravel())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model using the training data
model.fit(X_train, y_train)

# Flask route for the home page
@app.route('/')
def index():
    # Render and return the 'index.html' template
    return render_template('index.html')

# Flask route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the input text from the form submission
    input_text = request.form['input_text']
    # Preprocess the input text
    processed_text = preprocess(input_text)
    # Transform the processed text into TF-IDF features
    X_input_tfidf = tfidf.transform([processed_text]).toarray()
    # Compute sentiment polarity and subjectivity for the input text
    polarity = TextBlob(processed_text).sentiment.polarity
    subjectivity = TextBlob(processed_text).sentiment.subjectivity
    # Combine TF-IDF features with sentiment features
    X_input = np.hstack((X_input_tfidf, np.array([[polarity, subjectivity]])))

    # Predict the probability of each class
    probabilities = model.predict_proba(X_input)
    # Extract the probability of the 'sockpuppet' class
    sockpuppet_probability = probabilities[0][1]

    # Convert the probability to a percentage
    probability_percentage = round(sockpuppet_probability * 100, 2)

    # Render and return the 'result.html' template with the prediction result
    return render_template('result.html', prediction=probability_percentage)

# Function to open the default web browser to the home page of the app
def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

# Main entry point of the Flask application
if __name__ == '__main__':
    # Schedule the browser to open after 1 second
    Timer(1, open_browser).start()
    # Start the Flask application without using the auto-reloader
    app.run(debug=True, use_reloader=False)
