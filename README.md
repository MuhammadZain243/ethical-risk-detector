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
├── ├── labeling/               # Assign labels
│   ├── scrapping/              # Web scrapers (e.g., govuk_scraper.py)
│   ├── preprocessing/          # Text cleaning logic
│   ├── models/                 # Model training and evaluation
├── ├── config.py
├── └── load_data.py
│
├── scripts/
│
├── requirements.txt    # Required packages
└── README.md           # Project overview (this file)
```

---

## 🛠️ Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-org/ethical-risk-detector.git
cd ethical-risk-detector
pip install -r requirements.txt
```

---

## ⚙️ Pipeline Usage

### Step 1: Data Collection

```bash
python src/scraping/scrape_govuk.py
```

### Step 2: Data Preprocessing

```bash
python src/preprocessing/govuk_preprocessing.py
```

### Step 3: Document Metadata Sheet

```bash
python src/preprocessing/generate_metadata.py
```

### Step 4: Use Snorkel for Weak Supervision

```bash
python scripts/weak_supervision.py
```

### Step 5: Model Development

```bash
python scripts/train_transformer_model.py
```

### Step 6: SHAP and LIME Explaination

```bash
python scripts/shap_explain.py
python scripts/lime_explain.py
```

### Step 7: Comparison

```bash
python scripts/baseline_comparison.py
```
