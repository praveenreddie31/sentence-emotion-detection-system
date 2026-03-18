import os
import streamlit as st
import joblib

# Get base directory (project root)
base_dir = os.path.dirname(os.path.dirname(__file__))

# Correct paths
model_path = os.path.join(base_dir, "model", "emotion_model.pkl")
vectorizer_path = os.path.join(base_dir, "model", "vectorizer.pkl")

# Load model
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# UI
st.title("💬 Emotion Detection App")
st.write("Enter a sentence and detect its emotion!")

user_input = st.text_area("Enter your text here:")

if st.button("Detect Emotion"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Emotion: {prediction}")
    else:
        st.warning("Please enter some text!")