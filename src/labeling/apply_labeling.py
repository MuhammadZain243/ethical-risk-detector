# src/labeling/apply_labeling.py

import pandas as pd
from snorkel.labeling import PandasLFApplier
from src.labeling.snorkel_rules import (
    bias_keywords,
    surveillance_keywords,
    transparency_keywords,
    ABSTAIN
)
from src.data_loader import load_raw_data, preprocess_dataframe
from src.config import RAW_DATA_PATH

def apply_labeling_functions():
    # Load and clean raw sample CSV
    df = load_raw_data(RAW_DATA_PATH)
    df = preprocess_dataframe(df)

    # Apply labeling functions separately
    bias_lfs = [bias_keywords]
    surveillance_lfs = [surveillance_keywords]
    transparency_lfs = [transparency_keywords]

    bias_applier = PandasLFApplier(bias_lfs)
    surveillance_applier = PandasLFApplier(surveillance_lfs)
    transparency_applier = PandasLFApplier(transparency_lfs)

    df["bias_weak"] = bias_applier.apply(df=df).flatten()
    df["surveillance_weak"] = surveillance_applier.apply(df=df).flatten()
    df["transparency_weak"] = transparency_applier.apply(df=df).flatten()

    # Replace ABSTAIN (-1) with 0 (weak negative)
    df["bias_weak"] = df["bias_weak"].replace(ABSTAIN, 0)
    df["surveillance_weak"] = df["surveillance_weak"].replace(ABSTAIN, 0)
    df["transparency_weak"] = df["transparency_weak"].replace(ABSTAIN, 0)

    return df[["id", "source", "title", "text", "bias_weak", "surveillance_weak", "transparency_weak"]]
