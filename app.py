# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Emotion classes
emoji_map = {
    "joy": "😊", "sad": "😢", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲"
}

# Title
st.title("Emotion Classifier from Text 🧠💬")
st.subheader("Understand what your sentence expresses.")

# Text input
text_input = st.text_area("Enter your sentence here:", height=150)

# Clean function for educational preview
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

# Prediction logic
if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        preprocessed_text = clean_text(text_input)  # Apply preprocessing
        vector = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

# Display prediction + chart
if 'prediction' in st.session_state and 'probabilities' in st.session_state:
    prediction = st.session_state['prediction']
    probabilities = st.session_state['probabilities']
    raw_input = st.session_state['raw_input']

    st.markdown(f"### Predicted Emotion: **{prediction.capitalize()}** {emoji_map.get(prediction, '')}")

    prob_df = pd.DataFrame({
        'Emotion': model.classes_,
        'Confidence': np.round(probabilities, 4)
    }).sort_values(by='Confidence', ascending=False)

    st.bar_chart(prob_df.set_index('Emotion'))

    if st.checkbox("Show how the model processed your input"):
        st.code(clean_text(raw_input), language="text")


# Footer
st.caption("Developed as part of final year project — Personality and Emotion Analysis from Text (2025)")
