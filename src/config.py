# src/config.py

import os

DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "sample_cleaned.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "labeled_sample.csv")

MODEL_SAVE_PATH = "models/final_model"
LABELS = ["bias", "surveillance", "transparency"]

RANDOM_SEED = 42
MAX_SEQ_LENGTH = 256
