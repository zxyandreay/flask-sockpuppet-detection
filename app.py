import pandas as pd
import numpy as np
import joblib
import os
import webbrowser
import re
from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from threading import Timer

# Specify the correct template folder
app = Flask(__name__, template_folder="html")

# Load trained model and metadata
model_path = "model/random_forest.pkl"
feature_metadata_path = "model/feature_metadata.pkl"
embeddings_lookup_path = "model/embeddings_lookup.csv"
pca_model_path = "model/pca_model.pkl"

print("[INFO] Loading model...")
model = joblib.load(model_path)
print("[INFO] Model loaded successfully.")

print("[INFO] Loading feature metadata...")
feature_metadata = joblib.load(feature_metadata_path)
feature_names = feature_metadata['features']
print("[INFO] Feature metadata loaded.")

# Load embeddings lookup table
if os.path.exists(embeddings_lookup_path):
    print("[INFO] Loading embeddings lookup table...")
    embeddings_lookup = pd.read_csv(embeddings_lookup_path)
    embeddings_lookup.set_index('edit_text', inplace=True)
    print("[INFO] Embeddings lookup table loaded.")
else:
    print("[WARNING] Embeddings lookup table not found. Defaulting to dynamic embedding generation.")
    embeddings_lookup = None

# Load Sentence-BERT model
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load PCA model
if os.path.exists(pca_model_path):
    print("[INFO] Loading pre-trained PCA model...")
    pca = joblib.load(pca_model_path)
    print("[INFO] PCA model loaded successfully.")
else:
    print("[ERROR] Pre-trained PCA model not found! Ensure the model is saved as 'pca_model.pkl'.")
    pca = None

# 🔧 Added: Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)              # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()             # remove extra whitespace
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json if request.is_json else request.form
    edit_text = data.get('input_text', "").strip()

    if not edit_text:
        return render_template("result.html", prediction=None, error_message="Missing input text.")

    # Clean input before processing
    cleaned_text = clean_text(edit_text)

    # Check if cleaned text is empty (meaningless input)
    if not cleaned_text:
        return render_template("result.html", prediction=None, error_message="Text is empty after cleaning, please input meaningful text.", warning=True)

    # Use precomputed embedding if available
    if embeddings_lookup is not None and edit_text in embeddings_lookup.index:
        print("[INFO] 🔁 Using precomputed embeddings from lookup table.")
        features = embeddings_lookup.loc[edit_text].values.reshape(1, -1)
    else:
        print(f"[INFO] 🧠 Generating new embeddings for: \"{cleaned_text}\"")
        num_runs = 3
        embedding_runs = []
        for i in range(num_runs):
            embedding = sbert_model.encode([cleaned_text])
            embedding_runs.append(embedding[0])
            print(f"[DEBUG] Run {i+1} embedding sample: {embedding[0][:5]}...")  # just a snippet

        averaged_embedding = np.mean(embedding_runs, axis=0)
        normalized_embedding = averaged_embedding / np.linalg.norm(averaged_embedding)

        if pca:
            features = pca.transform(normalized_embedding.reshape(1, -1))
        else:
            return render_template("result.html", prediction=None, error_message="PCA model is missing.")

        print(f"[INFO] ✅ Final normalized and PCA-reduced embedding: {features[0][:5]}... (truncated)")

    polarity = TextBlob(cleaned_text).sentiment.polarity
    subjectivity = TextBlob(cleaned_text).sentiment.subjectivity

    print(f"[INFO] ✨ Sentiment features → Polarity: {polarity}, Subjectivity: {subjectivity}")

    X_input = np.zeros((1, len(feature_names)))
    print("[INFO] 📊 Assembled feature vector:")

    for i, feature in enumerate(feature_names):
        if feature.startswith("embedding_"):
            embedding_index = int(feature.split("_")[-1])
            if embedding_index < features.shape[1]:
                X_input[0, i] = features[0, embedding_index]
        elif feature == "polarity":
            X_input[0, i] = polarity
        elif feature == "subjectivity":
            X_input[0, i] = subjectivity
        print(f"  - {feature}: {X_input[0, i]}")

    probabilities = model.predict_proba(X_input)[0]
    prediction = "sockpuppet" if probabilities[1] >= 0.5 else "non-sockpuppet"
    percent_sock = probabilities[1] * 100
    percent_non_sock = probabilities[0] * 100

    print(f"[RESULT] 🧾 Prediction: {prediction.upper()}")
    print(f"[RESULT] 📈 Confidence → Sockpuppet: {percent_sock:.2f}%, Non-Sockpuppet: {percent_non_sock:.2f}%")

    return render_template("result.html", prediction=prediction)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
