# 🧠 Ethical Risk Detector (Responsible AI by Design)

A full-featured NLP system to detect **ethical risks** in public sector AI policies and documents using weak supervision and transformer-based models. This project aims to identify indicators of:

- 🔍 **Bias**
- 📹 **Surveillance**
- 🧾 **Lack of Transparency**

---

## 📁 Project Structure

ethical-risk-detector/
|
├── data/
│ ├── raw/ ← Unprocessed .txt files scraped from gov.uk
│ ├── cleaned/ ← JSON files after preprocessing
│ └── processed/ ← Labeled CSV for training


├── src/

│ ├── config.py ← Global path and constants

│ ├── scrapping/ ← Web scrapers (e.g., govuk_scraper.py)

│ ├── preprocessing/ ← Text cleaning logic

│ ├── labeling/ ← Snorkel rules and labeler
│ ├── models/ ← Model training, evaluation, inference
│ └── utils/ ← Utility functions (tokenization, logging, etc.)
│
├── scripts/
│ ├── scrape_data.py ← Run the gov.uk scraper
│ ├── clean_data.py ← Run data cleaning pipeline
│ ├── label_cleaned_data.py ← Apply Snorkel labeling
│ ├── run_inference.py ← Inference CLI
│ └── train_model.sh ← Shell wrapper for training
│
├── logs/ ← Training logs
├── models/ ← Saved model checkpoints
├── requirements.txt ← All required packages
├── README.md ← This file
└── .env ← Optional secrets (not versioned)

---

## 🛠 Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-org/ethical-risk-detector.git
cd ethical-risk-detector
pip install -r requirements.txt
```

2. Install Torch + Transformers

```bash
pip install torch torchvision torchaudio
pip install transformers datasets snorkel scikit-learn
```

⚙️ Usage
Step 1: Scrape raw data

```bash
python scripts/scrape_data.py
```

Step 2: Clean and preprocess

```bash
python scripts/clean_data.py
```

Step 3: Apply Snorkel labeling functions

```bash
python scripts/label_cleaned_data.py
```

Step 4: Train multi-label classifier (Roberta)

```bash
python -m src.models.trainer

```

Step 5: Run inference

```bash
python scripts/run_inference.py

```

You’ll be prompted to enter custom text:

```bash
Enter text: AI models trained on historical data may reinforce discrimination.
Predicted Risks: {'bias': 1, 'surveillance': 0, 'transparency': 1}

```

📈 Model Details
Model: RobertaForSequenceClassification with 3-label output (bias, surveillance, transparency)
Format: Multi-label classification using BCEWithLogitsLoss
Input: Cleaned text from policy documents
Output: Probability (converted to binary via 0.5 threshold)

🧠 Snorkel Labeling Functions
The model is trained using weak supervision via Snorkel. Labeling functions include:
bias_keywords(): Detects fairness, social discrimination, systemic bias
surveillance_keywords(): Flags facial recognition, tracking, real-time monitoring
transparency_keywords(): Detects black-box warnings, lack of explanation or audit
You can find them in src/labeling/snorkel_rules.py.

✅ Sample Output
| id | title | text | bias | surveillance | transparency |
| ---------- | -------------------------------- | ---- | ---- | ------------ | ------------ |
| govuk_001 | AI Regulation | ... | 1 | 0 | 1 |
| govuk_002 | Facial Recognition in Public Use | ... | 0 | 1 | 0 |

📌 Future Work
SHAP/LIME-based explainability
Human-in-the-loop review dashboard
Fine-tuning on manually annotated corpus
Zero-shot inference with TARS/BART variants

👤 Author
Muhammad Zain Ul Abideen
💼 Full Stack AI Developer
📧 zainm2432003@gmail.com
