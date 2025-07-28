import shap
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load trained model
model_path = "results/checkpoint-9"  # adjust if different
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model.eval()

# Load sample text
df = pd.read_csv('data/cleaned/document_metadata_with_labels.csv')
sample_texts = df['Text'].iloc[:2].tolist()  # just 2 samples for now

# ✅ SHAP needs predict to accept list[str]
def predict(texts: list[str]):
    # Ensure texts is a list of strings
    texts = [str(t) for t in texts]

    # Tokenize and predict
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)
    return probs.numpy()

# Use SHAP's PartitionExplainer
explainer = shap.Explainer(predict, tokenizer)
shap_values = explainer(sample_texts)

# Save HTML explanation
for i, text in enumerate(sample_texts):
    html = shap.plots.text(shap_values[i], display=False)
    with open(f"shap_explanation_{i}.html", "w", encoding="utf-8") as f:
        f.write(html)
