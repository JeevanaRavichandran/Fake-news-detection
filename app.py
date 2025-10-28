# -------------------------------------------------
# 🧠 Fake News Detection Web App (app.py)
# -------------------------------------------------

import streamlit as st
import joblib
import numpy as np
import re

# -------------------------------------------------
# Step 1: Load the saved model and vectorizer
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# -------------------------------------------------
# Step 2: Clean text for prediction
# -------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().strip()
    return text

# -------------------------------------------------
# Step 3: Predict function
# -------------------------------------------------
def predict_news(news, model, vectorizer):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]
    pred = model.predict(vector)[0]
    confidence = round(np.max(prob) * 100, 2)
    return pred, confidence, dict(zip(model.classes_, prob))

# -------------------------------------------------
# Step 4: Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title=" Fake News Detection", layout="centered")
st.title(" Fake News Detection using AI")
st.markdown("Detect whether a news headline or article is **REAL or FAKE** using a trained AI model.")

# User input box
user_input = st.text_area(" Enter a news headline or article:")

# Prediction button
if st.button(" Predict"):
    if user_input.strip() == "":
        st.warning(" Please enter some text before predicting.")
    else:
        model, vectorizer = load_model()
        pred, confidence, probs = predict_news(user_input, model, vectorizer)

        # Display result in color
        if pred == "REAL":
            st.success(f" Prediction: {pred} ({confidence:.2f}% confidence)")
        else:
            st.error(f" Prediction: {pred} ({confidence:.2f}% confidence)")

        st.markdown("###  Prediction Details:")
        st.json(probs)

st.markdown("---")
st.caption("Built with Streamlit, Scikit-learn, and NLP")

