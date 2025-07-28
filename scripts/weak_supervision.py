import sys
import os

# Add the root directory (where 'src' folder is) to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the necessary modules
from src.labeling.apply_labeling import apply_labeling_functions

def main():
    # Apply labeling functions to the raw data
    df_labeled = apply_labeling_functions()

    # Save the labeled data to CSV
    output_path = "data/cleaned/document_metadata_with_labels.csv"
    df_labeled.to_csv(output_path, index=False)

    print(f"Labeled data saved to {output_path}")

if __name__ == "__main__":
    main()
