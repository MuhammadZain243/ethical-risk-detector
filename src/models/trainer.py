# src/models/trainer.py

import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from src.config import PROCESSED_DATA_PATH, MODEL_SAVE_PATH

# ---------------------------
# Custom Trainer for BCE Loss
# ---------------------------
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits
        labels = labels.type(torch.float32)

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ---------------------------
# Load and preprocess dataset
# ---------------------------
def load_labeled_dataset(path=PROCESSED_DATA_PATH):
    df = pd.read_csv(path)
    df["labels"] = df[["bias_weak", "surveillance_weak", "transparency_weak"]].values.tolist()
    return df[["text", "labels"]]

def tokenize_and_format(example, tokenizer, max_length=256):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
    tokens["labels"] = [float(x) for x in example["labels"]]  # return as list, not tensor
    return tokens

# ---------------------------
# Evaluation metrics
# ---------------------------
def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, precision_score, recall_score
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

# ---------------------------
# Main training logic
# ---------------------------
def train_model():
    df = load_labeled_dataset()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(lambda x: tokenize_and_format(x, tokenizer), batched=False)
    val_ds = val_ds.map(lambda x: tokenize_and_format(x, tokenizer), batched=False)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=3,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        logging_dir="./logs",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        do_eval=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH)
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    train_model()
