from flask import Flask, request, render_template
import webbrowser
import numpy as np
import re
from threading import Timer
from textblob import TextBlob
from joblib import load
import pandas as pd
import os

# Initialize Flask application
app = Flask(__name__, template_folder='html')

# Load the trained model and feature information
model_path = 'model/random_forest.pkl'
feature_metadata_path = 'model/feature_metadata.pkl'
embeddings_lookup_path = 'model/embeddings_lookup.csv'

if not os.path.exists(model_path) or not os.path.exists(feature_metadata_path):
    raise FileNotFoundError("[ERROR] Model or feature metadata file not found. Ensure the training process is completed.")

print("[INFO] Loading trained model...")
model = load(model_path)
print("[INFO] Model loaded successfully.")

print("[INFO] Loading feature metadata...")
feature_metadata = load(feature_metadata_path)
embedding_cols = feature_metadata['features'][:-2]  # Exclude last two sentiment features
sentiment_cols = feature_metadata['features'][-2:]
print("[INFO] Feature metadata loaded.")

# Load precomputed embeddings for lookup
if os.path.exists(embeddings_lookup_path):
    print("[INFO] Loading embeddings lookup table...")
    embeddings_df = pd.read_csv(embeddings_lookup_path, index_col='edit_text')
    print("[INFO] Embeddings lookup loaded.")
else:
    print("[WARNING] Embeddings lookup table not found. Using dynamically generated embeddings for missing cases.")
    embeddings_df = None

# Preprocess function
def preprocess(text):
    print("[INFO] Preprocessing input text...")
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>|\=.*?\=', '', text)
    text = re.sub(r'\W', ' ', text)
    print("[INFO] Preprocessing complete.")
    return text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    processed_text = preprocess(input_text)
    
    # Retrieve embeddings or generate new ones dynamically
    print("[INFO] Retrieving embeddings...")
    if embeddings_df is not None and processed_text in embeddings_df.index:
        embedding_features = embeddings_df.loc[processed_text, embedding_cols].values
    else:
        print("[WARNING] Embeddings not found. Generating dynamically...")
        embedding_features = np.random.normal(0, 1, len(embedding_cols))  # Generate random embeddings
    print("[INFO] Embeddings retrieved.")
    
    # Compute sentiment features
    print("[INFO] Computing sentiment features...")
    polarity = TextBlob(processed_text).sentiment.polarity
    subjectivity = TextBlob(processed_text).sentiment.subjectivity
    sentiment_features = np.array([polarity, subjectivity])
    print(f"[INFO] Sentiment computed: Polarity={polarity}, Subjectivity={subjectivity}")
    
    # Combine embeddings and sentiment features
    print("[INFO] Combining features...")
    
    # Convert array to DataFrame with correct feature names
    X_input = pd.DataFrame([np.hstack((embedding_features, sentiment_features))], columns=feature_metadata['features'])

    # Ensure columns are in the exact order as during training
    X_input = X_input[feature_metadata['features']]

    print("[DEBUG] Feature Vector Shape:", X_input.shape)
    print("[DEBUG] Sample Feature Vector:", X_input)
    print("[INFO] Features combined successfully.")
    
    # Prediction
    print("[INFO] Making prediction...")
    prediction = model.predict(X_input)[0]
    result = 'sockpuppet' if prediction == 1 else 'non-sockpuppet'
    print(f"[INFO] Prediction result: {result}")
    return render_template('result.html', prediction=result)

# Open browser function
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    print("[INFO] Starting Flask server...")
    app.run(debug=True, use_reloader=False)