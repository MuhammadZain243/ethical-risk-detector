import streamlit as st
import sys
import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import MODEL_SAVE_PATH

CLASS_NAMES = ['bias', 'surveillance', 'transparency']

st.title("🧠 Ethical Risk Detection with Explainability")

text = st.text_area("Enter public sector AI project description:", height=200)

if st.button("Run Prediction") and text.strip():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()

    pred_dict = {label: float(prob) for label, prob in zip(CLASS_NAMES, probs)}

    st.subheader("📊 Predicted Risks:")
    st.json(pred_dict)
