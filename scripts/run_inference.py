# scripts/run_inference.py

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import MODEL_SAVE_PATH, LABELS

# Load model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model.eval()  # Set to inference mode

def predict_risks(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Convert probs to 0 or 1 (threshold = 0.5)
    preds = (probs >= 0.5).astype(int)
    return {label: int(pred) for label, pred in zip(LABELS, preds)}

# Example usage
if __name__ == "__main__":
    test_text = "The algorithm prioritizes job applicants based on historical performance data, which may reflect gender or ethnic biases."
    result = predict_risks(test_text)
    print("Predicted Risks:", result)
