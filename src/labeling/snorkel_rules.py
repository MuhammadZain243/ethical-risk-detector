from snorkel.labeling import labeling_function

# Label values
ABSTAIN = -1
BIAS = 1
SURVEILLANCE = 1
TRANSPARENCY = 1

# ---------------------- BIAS ----------------------

@labeling_function()
def bias_keywords(x):
    text = x["Text"]

    direct_keywords = [
        "bias", "discrimination", "fairness", "unfair", "prejudice",
        "inequity", "equality", "inclusion", "unjust", "biased"
    ]

    social_keywords = [
        "gender", "ethnic", "race", "minority", "disabled", "age group",
        "underrepresented", "diversity", "marginalized"
    ]

    contextual_phrases = [
        "historical performance data", "historic data", "unequal outcome",
        "social inequality", "disparate impact", "skewed decisions",
        "algorithmic unfairness", "biased outcomes"
    ]

    if any(kw in text for kw in direct_keywords):
        return BIAS
    if any(kw in text for kw in social_keywords) and "data" in text:
        return BIAS
    if any(phrase in text for phrase in contextual_phrases):
        return BIAS

    return ABSTAIN

# ------------------ SURVEILLANCE ------------------

@labeling_function()
def surveillance_keywords(x):
    text = x["Text"]

    surveillance_keywords = [
        "monitoring", "surveillance", "facial recognition", "face scan",
        "biometric", "camera", "cctv", "license plate reader",
        "observation", "tracking", "geo-tracking", "location data",
        "behavioral profiling", "data capture", "live video feed"
    ]

    contextual_phrases = [
        "real-time video analysis", "identify individuals in public",
        "mass surveillance", "crowd monitoring", "public space scanning"
    ]

    if any(kw in text for kw in surveillance_keywords):
        return SURVEILLANCE
    if any(phrase in text for phrase in contextual_phrases):
        return SURVEILLANCE

    return ABSTAIN

# ------------------ TRANSPARENCY ------------------

@labeling_function()
def transparency_keywords(x):
    text = x["Text"]

    transparency_keywords = [
        "transparency", "accountability", "explainable", "interpretable",
        "traceability", "auditability", "white box", "model explanation",
        "documentation", "visibility", "responsiveness", "disclosure"
    ]

    contextual_phrases = [
        "lack of explanation", "black box", "no visibility into decision",
        "unable to justify outcome", "opaque logic", "hidden logic"
    ]

    if any(kw in text for kw in transparency_keywords):
        return TRANSPARENCY
    if any(phrase in text for phrase in contextual_phrases):
        return TRANSPARENCY

    return ABSTAIN
