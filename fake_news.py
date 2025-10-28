
# -------------------------------------------------
# 🧠 Fake News Detection (Auto-Retrain Version)
# -------------------------------------------------

import os
import time
import hashlib
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download("stopwords")

MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
HASH_PATH = "data_hash.txt"

# -------------------------------------------------
# Step 1: Compute dataset hash (for change detection)
# -------------------------------------------------
def compute_dataset_hash(true_path="True.csv", fake_path="Fake.csv"):
    true_hash = hashlib.md5(open(true_path, "rb").read()).hexdigest()
    fake_hash = hashlib.md5(open(fake_path, "rb").read()).hexdigest()
    return true_hash + fake_hash

# -------------------------------------------------
# Step 2: Clean text
# -------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().strip()
    return text

# -------------------------------------------------
# Step 3: Train the model
# -------------------------------------------------
def train_and_save_model():
    print("\n Loading dataset...")
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    # Balance the dataset
    min_len = min(len(true_df), len(fake_df))
    true_df = true_df.sample(min_len, random_state=42)
    fake_df = fake_df.sample(min_len, random_state=42)

    df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    print(" Real news samples:", len(true_df))
    print(" Fake news samples:", len(fake_df))

    stop_words = set(stopwords.words("english"))
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Model trained successfully with accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # Save current dataset hash
    dataset_hash = compute_dataset_hash()
    with open(HASH_PATH, "w") as f:
        f.write(dataset_hash)

    print(" Model and vectorizer saved successfully!")

# -------------------------------------------------
# Step 4: Load or retrain model if dataset changed
# -------------------------------------------------
def load_or_retrain_model():
    dataset_hash = compute_dataset_hash()
    previous_hash = None

    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            previous_hash = f.read().strip()

    if not os.path.exists(MODEL_PATH) or dataset_hash != previous_hash:
        print("\n Dataset changed or model missing — retraining...")
        train_and_save_model()
    else:
        print("\n Model is up-to-date. Loading existing model...")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# -------------------------------------------------
# Step 5: Prediction function
# -------------------------------------------------
def predict_news(news, model, vectorizer):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]
    pred = model.predict(vector)[0]
    confidence = round(np.max(prob), 2)
    print("\n Input:", news)
    print(" Prediction:", pred)
    print(" Confidence:", confidence)
    print(" Probabilities:", dict(zip(model.classes_, prob)))

# -------------------------------------------------
# Step 6: Main
# -------------------------------------------------
if __name__ == "__main__":
    model, vectorizer = load_or_retrain_model()

    # Example predictions
    predict_news("NASA confirms water found on Mars in new research.", model, vectorizer)
    predict_news("Aliens have landed in New York City according to anonymous reports.", model, vectorizer)
    predict_news("Government of India launches a new AI initiative for education.", model, vectorizer)




