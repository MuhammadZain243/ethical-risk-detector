# 🧠 Ethical Risk Detector (Responsible AI by Design)

A full-featured NLP pipeline to detect **ethical risks** in UK government AI policy documents using **weak supervision** and **transformer-based models**. The system identifies textual indicators of:

- 🔍 **Bias**
- 📹 **Surveillance**
- 🧾 **Lack of Transparency**

---

## 📁 Project Structure

```
ethical-risk-detector/
├── data/
│   ├── raw/            # Unprocessed .txt files scraped from gov.uk
│   ├── cleaned/        # Preprocessed JSON files
│   └── processed/      # Labeled CSVs ready for model training
│
├── src/
│   ├── scrapping/              # Web scrapers (e.g., govuk_scraper.py)
│   ├── preprocessing/          # Text cleaning logic
│   ├── labeling/               # Snorkel labeling functions
│   ├── models/                 # Model training and evaluation
|   ├── config.py               # Global paths and constants
│   └── data_loader.py          # Load data
│
├── scripts/
│   ├── label_cleaned_data.py   # Apply labeling
│   └── run_inference.py        # Run predictions on new text
│
├── logs/               # Training logs
├── models/             # Saved model checkpoints
├── requirements.txt    # Required packages
├── README.md           # Project overview (this file)
└── .env                # Local environment variables (not versioned)
```

---

## 🛠️ Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-org/ethical-risk-detector.git
cd ethical-risk-detector
pip install -r requirements.txt
```

### 2. Install Key Libraries

```bash
pip install torch torchvision torchaudio
pip install transformers datasets snorkel scikit-learn
```

---

## ⚙️ Pipeline Usage

### Step 1: Scrape Raw Data

```bash
python scripts/scrape_data.py
```

### Step 2: Clean and Preprocess

```bash
python scripts/clean_data.py
```

### Step 3: Apply Snorkel Labeling Functions

```bash
python scripts/label_cleaned_data.py
```

### Step 4: Train Roberta Multi-label Classifier

```bash
python -m src.models.trainer
```

### Step 5: Run Inference

```bash
python scripts/run_inference.py
```

You’ll be prompted to enter text:

```
Enter text: AI models trained on historical data may reinforce discrimination.
Predicted Risks: {'bias': 1, 'surveillance': 0, 'transparency': 1}
```

---

## 🤖 Model Details

- **Model**: `RobertaForSequenceClassification`
- **Type**: Multi-label classification (3 labels: bias, surveillance, transparency)
- **Loss**: `BCEWithLogitsLoss`
- **Input**: Preprocessed policy document text
- **Output**: Risk probabilities → binary labels (threshold = 0.5)

---

## 🧠 Snorkel Labeling Functions

Weak supervision is used to auto-label examples using rules in `src/labeling/snorkel_rules.py`. Key functions include:

- `bias_keywords()`: Flags fairness, systemic bias, social discrimination
- `surveillance_keywords()`: Flags tracking, facial recognition, mass monitoring
- `transparency_keywords()`: Flags black-box warnings, lack of audit/explainability

---

## ✅ Example Output

| id        | title                            | bias | surveillance | transparency |
| --------- | -------------------------------- | ---- | ------------ | ------------ |
| govuk_001 | AI Regulation                    | 1    | 0            | 1            |
| govuk_002 | Facial Recognition in Public Use | 0    | 1            | 0            |

---

## 🔭 Future Work

- SHAP/LIME-based interpretability
- Human-in-the-loop review interface
- Fine-tuning on manually annotated corpus
- Zero-shot inference using TARS/BART

---

## 👤 Author

**Muhammad Zain Ul Abideen**  
💼 Full Stack AI Developer  
📧 zainm2432003@gmail.com
