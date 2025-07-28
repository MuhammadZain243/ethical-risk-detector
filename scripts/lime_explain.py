# === lime_explain.py ===

from lime.lime_text import LimeTextExplainer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np
import pandas as pd

# Load model and tokenizer
model_path = "results/checkpoint-9"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model.eval()

# Load sample text
df = pd.read_csv('data/cleaned/document_metadata_with_labels.csv')
sample_text = df['Text'].iloc[0][:512]  # Truncate text before passing to LIME
  # choose any row

# Define class names
class_names = ['Bias_weak', 'Surveillance_weak', 'Transparency_weak']

# Prediction wrapper for LIME
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).numpy()
    return probs

# LIME Explainer
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(sample_text, predict_proba, num_features=10)

# Save to HTML
exp.save_to_file("lime_explanation.html")
