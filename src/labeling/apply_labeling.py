from snorkel.labeling import PandasLFApplier
from src.labeling.snorkel_rules import (
    bias_keywords,
    surveillance_keywords,
    transparency_keywords,
    ABSTAIN
)
from src.load_data import load_raw_data, preprocess_dataframe
from src.config import RAW_DATA_PATH

def apply_labeling_functions():
    # Load and clean raw sample CSV
    df = load_raw_data(RAW_DATA_PATH)
    print(f"Columns in the DataFrame: {df.columns}")

    df = preprocess_dataframe(df)

    # Apply labeling functions separately
    bias_lfs = [bias_keywords]
    surveillance_lfs = [surveillance_keywords]
    transparency_lfs = [transparency_keywords]

    bias_applier = PandasLFApplier(bias_lfs)
    surveillance_applier = PandasLFApplier(surveillance_lfs)
    transparency_applier = PandasLFApplier(transparency_lfs)

    df["Bias_weak"] = bias_applier.apply(df=df).flatten()
    df["Surveillance_weak"] = surveillance_applier.apply(df=df).flatten()
    df["Transparency_weak"] = transparency_applier.apply(df=df).flatten()

    # Replace ABSTAIN (-1) with 0 (weak negative)
    df["Bias_weak"] = df["Bias_weak"].replace(ABSTAIN, 0)
    df["Surveillance_weak"] = df["Surveillance_weak"].replace(ABSTAIN, 0)
    df["Transparency_weak"] = df["Transparency_weak"].replace(ABSTAIN, 0)

    return df[["ID", "Source", "Title", "Text", "Bias_weak", "Surveillance_weak", "Transparency_weak"]]
