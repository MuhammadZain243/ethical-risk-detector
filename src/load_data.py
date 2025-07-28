# src/data_loader.py

import pandas as pd
import os
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, RANDOM_SEED
from sklearn.model_selection import train_test_split

def load_raw_data(path=RAW_DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}")
    return pd.read_csv(path)

def clean_text(text):
    return str(text).strip().lower()

def preprocess_dataframe(df: pd.DataFrame):
    df = df.drop_duplicates()
    df = df.dropna(subset=["Text"])
    df["Text"] = df["Text"].apply(clean_text)
    return df

def save_processed_data(df: pd.DataFrame, path=PROCESSED_DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def split_data(df: pd.DataFrame, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)

if __name__ == "__main__":
    df = load_raw_data()
    df = preprocess_dataframe(df)
    train, test = split_data(df)
    print(f"✅ Train: {len(train)}, Test: {len(test)}")