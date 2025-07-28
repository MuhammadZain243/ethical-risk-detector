import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, hamming_loss
import pandas as pd
from datasets import Dataset

# Load the labeled data (document_metadata_with_labels.csv)
df = pd.read_csv('data/cleaned/document_metadata_with_labels.csv')

# Define the label columns (Bias_weak, Surveillance_weak, Transparency_weak)
label_columns = ['Bias_weak', 'Surveillance_weak', 'Transparency_weak']

# Prepare the dataset for Hugging Face's Trainer
def preprocess_data(df):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Tokenize the texts
    encodings = tokenizer(df['Text'].tolist(), truncation=True, padding=True, max_length=512)

    labels = df[label_columns].values  # Extract the labels (binary for each ethical risk)
    labels = torch.tensor(labels, dtype=torch.float)

    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels.tolist()
    })

# Prepare the dataset
dataset = preprocess_data(df)

# Stratified K-Fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluation metrics
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(axis=-1)  # Get the class with highest probability
#     f1 = f1_score(labels, preds, average='macro')  # Macro F1-score
#     hl = hamming_loss(labels, preds)  # Hamming loss
#     return {'f1_score': f1, 'hamming_loss': hl}

def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids

    # Apply sigmoid since we're using multi-label classification
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()  # Binarize predictions at threshold 0.5

    f1 = f1_score(labels, preds, average='macro')
    hl = hamming_loss(labels, preds)

    return {
        'f1_score': f1,
        'hamming_loss': hl
    }


# Training function using Hugging Face Trainer
def train_model(train_idx, val_idx):
    train_data = dataset.select(train_idx)
    val_data = dataset.select(val_idx)

    # Load pre-trained RoBERTa model for sequence classification
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3,problem_type="multi_label_classification")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device) 

    training_args = TrainingArguments(
        output_dir='./results', 
        eval_strategy='epoch',  # Evaluate after every epoch
        save_strategy='epoch',        # Save after every epoch
        learning_rate=2e-5, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16, 
        num_train_epochs=3, 
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,  # Load the best model at the end of training
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_data, 
        eval_dataset=val_data, 
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

# Perform 5-fold cross-validation
for train_idx, val_idx in kf.split(df, df['Bias_weak']):  # Using 'Bias_weak' for stratification
    print(f"Training fold with {len(train_idx)} train samples and {len(val_idx)} validation samples")
    train_model(train_idx, val_idx)
