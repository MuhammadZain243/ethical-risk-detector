import os
import json
import re
from pathlib import Path
from unidecode import unidecode

def clean_text(text):
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(page \d+|© crown copyright.*)", "", text, flags=re.I)
    text = text.strip()
    return text

def process_txt_folder(source_folder, source_name, save_path):
    cleaned_data = []

    for i, file in enumerate(Path(source_folder).glob("*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text)
        
        if len(cleaned) < 300:
            continue

        entry = {
            "id": f"{source_name}_{i:03}",
            "source": source_name,
            "title": file.stem.replace("_", " ").title(),
            "text": cleaned,
            "year": 2023
        }
        cleaned_data.append(entry)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as out:
        json.dump(cleaned_data, out, indent=2)
    print(f"✅ Saved cleaned {source_name} data to: {save_path}")

if __name__ == "__main__":
    process_txt_folder(
        source_folder="data/raw/govuk",
        source_name="govuk",
        save_path="data/cleaned/govuk_cleaned.json"
    )
