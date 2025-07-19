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
│   └── models/                 # Model training and evaluation
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
