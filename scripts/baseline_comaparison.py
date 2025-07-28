import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, hamming_loss
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer

# === Load Data ===
df = pd.read_csv('data/cleaned/document_metadata_with_labels.csv')
y_true = df[['Bias_weak', 'Surveillance_weak', 'Transparency_weak']].values
labels = ['Bias_weak', 'Surveillance_weak', 'Transparency_weak']

# === 1. Keyword-Based Classifier ===
keyword_dict = {
    'Bias_weak': ['bias', 'discrimination', 'prejudice'],
    'Surveillance_weak': ['monitoring', 'tracking', 'surveillance', 'camera'],
    'Transparency_weak': ['transparent', 'openness', 'accountability', 'disclosure']
}

def keyword_classifier(text):
    return [
        int(any(word.lower() in text.lower() for word in keyword_dict[label]))
        for label in labels
    ]

df['keyword_preds'] = df['Text'].apply(keyword_classifier)
keyword_preds = pd.DataFrame(df['keyword_preds'].tolist(), columns=labels)

# === 2. Zero-Shot Classifier (BART) ===
from tqdm import tqdm
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zero_shot_preds = []

for text in tqdm(df['Text'].tolist(), desc="Zero-shot predictions"):
    result = zero_shot(text, candidate_labels=labels, multi_label=True)
    preds = [1 if score > 0.5 else 0 for score in result['scores']]
    zero_shot_preds.append(preds)

zero_shot_df = pd.DataFrame(zero_shot_preds, columns=labels)

# === 3. RoBERTa Classifier Predictions ===
model_path = "results/checkpoint-9"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model.eval()

def roberta_predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)
    return (probs > 0.5).int().numpy()

# Process in batches to save memory
batch_size = 16
roberta_preds = []
for i in tqdm(range(0, len(df), batch_size), desc="RoBERTa predictions"):
    batch_texts = df['Text'].iloc[i:i+batch_size].tolist()
    preds = roberta_predict(batch_texts)
    roberta_preds.extend(preds)
roberta_preds = np.array(roberta_preds)

# === 4. Metrics ===
print("\n📊 Performance Comparison:")
print("🔹 Keyword Classifier:")
print("F1 Score:", f1_score(y_true, keyword_preds.values, average='macro'))
print("Hamming Loss:", hamming_loss(y_true, keyword_preds.values))

print("\n🔹 Zero-Shot (BART):")
print("F1 Score:", f1_score(y_true, zero_shot_df.values, average='macro'))
print("Hamming Loss:", hamming_loss(y_true, zero_shot_df.values))

print("\n🔹 RoBERTa Fine-tuned:")
print("F1 Score:", f1_score(y_true, roberta_preds, average='macro'))
print("Hamming Loss:", hamming_loss(y_true, roberta_preds))
