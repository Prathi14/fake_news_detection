import streamlit as st
import pickle
import re

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Streamlit UI
st.title("📰 Fake News Detection App")
user_input = st.text_area("Enter News Content:")

if st.button("Check"):
    if user_input.strip():  # Check for empty input
        input_vec = vectorizer.transform([clean_text(user_input)])
        prediction = model.predict(input_vec)[0]
        result = "❌ Fake News" if prediction == 1 else "✅ Real News"
        st.write(f"Prediction: {result}")
    else:
        st.write("⚠️ Please enter some news content to check.")
