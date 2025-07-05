# scripts/label_cleaned_data.py

import json
import pandas as pd
import os
from src.labeling.apply_labeling import apply_labeling_functions
from src.data_loader import save_processed_data

def load_cleaned_json(path="data/cleaned/govuk_cleaned.json"):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

if __name__ == "__main__":
    # Step 1: Load cleaned data
    df = load_cleaned_json()

    # Step 2: Take a sample (or all)
    sample_df = df.sample(n=min(50, len(df)), random_state=42)
    sample_df.to_csv("data/raw/sample_cleaned.csv", index=False)

    # Step 3: Apply labeling functions using Snorkel
    labeled_df = apply_labeling_functions()

    # Step 4: Save labeled data
    os.makedirs("data/processed", exist_ok=True)
    save_processed_data(labeled_df, "data/processed/labeled_sample.csv")
    print("✅ Labeled data saved to: data/processed/labeled_sample.csv")
