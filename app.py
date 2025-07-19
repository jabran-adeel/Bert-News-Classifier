# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set page config
st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")

# Load model and tokenizer
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "./results/checkpoint-125"  # Adjust path as needed

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

label_map = {0: "🌍 World", 1: "🏅 Sports", 2: "💼 Business", 3: "🔬 Sci/Tech"}

# UI Layout
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📰 News Topic Classifier</h1>", unsafe_allow_html=True)
st.markdown("### Enter a news headline below and click the button to classify it:")

user_input = st.text_area("✍️ News Headline", placeholder="e.g. NASA announces new moon mission")

if st.button("🚀 Classify News"):
    if not user_input.strip():
        st.warning("Please enter a news headline to classify.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']


        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()
        confidence = torch.nn.functional.softmax(logits, dim=1).squeeze()[prediction].item()

        # Display results
        st.success(f"**Predicted Category:** {label_map[prediction]}")
        st.progress(int(confidence * 100))
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with 🤖 using BERT | By <a href='https://github.com/jabran-adeel' target='_blank'>Jabran Adeel</a></div>", unsafe_allow_html=True)