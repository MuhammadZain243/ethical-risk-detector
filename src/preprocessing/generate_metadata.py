import json
import os
import csv

def generate_metadata_from_preprocessed(preprocessed_file, metadata_csv_path):
    with open(preprocessed_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    metadata = []

    for i, document in enumerate(data):
        entry = {
            "ID": document["id"],
            "Source": document["source"],
            "Title": document["title"],
            "Text": document["text"],
            "Year": document["year"],
        }
        metadata.append(entry)

    os.makedirs(os.path.dirname(metadata_csv_path), exist_ok=True)

    with open(metadata_csv_path, "w", encoding="utf-8", newline='') as csvfile:
        fieldnames = ["ID", "Source", "Title", "Text", "Year"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"✅ Saved document metadata to: {metadata_csv_path}")

if __name__ == "__main__":
    preprocessed_file = "data/cleaned/govuk_cleaned.json"
    metadata_csv_path = "data/cleaned/document_metadata.csv"
    generate_metadata_from_preprocessed(preprocessed_file, metadata_csv_path)
