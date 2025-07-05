# 🧠 Ethical Risk Detector (Responsible AI by Design)

A full-featured NLP system to detect **ethical risks** in public sector AI policies and documents using weak supervision and transformer-based models. This project aims to identify indicators of:

- 🔍 **Bias**
- 📹 **Surveillance**
- 🧾 **Lack of Transparency**

---

## 📁 Project Structure

D:.
|   .env
|   .gitignore
|   README.md
|   requirements.txt
|   tree.txt
|   
+---data
|   +---cleaned
|   |       govuk_cleaned.json
|   |       
|   +---processed
|   |       labeled_sample.csv
|   |       
|   \---raw
|       |   sample_cleaned.csv
|       |   
|       \---govuk
|               AI_ethics_0.txt
|               AI_ethics_1.txt
|               AI_ethics_2.txt
|               AI_ethics_3.txt
|               AI_ethics_4.txt
|               AI_fairness_0.txt
|               AI_fairness_1.txt
|               AI_fairness_2.txt
|               AI_fairness_3.txt
|               AI_fairness_4.txt
|               AI_procurement_0.txt
|               AI_procurement_1.txt
|               AI_procurement_2.txt
|               AI_procurement_3.txt
|               AI_procurement_4.txt
|               algorithm_transparency_0.txt
|               algorithm_transparency_1.txt
|               algorithm_transparency_2.txt
|               algorithm_transparency_3.txt
|               algorithm_transparency_4.txt
|               automated_decision-making_0.txt
|               automated_decision-making_1.txt
|               automated_decision-making_2.txt
|               automated_decision-making_3.txt
|               automated_decision-making_4.txt
|               Responsible_AI_0.txt
|               Responsible_AI_1.txt
|               Responsible_AI_2.txt
|               Responsible_AI_3.txt
|               Responsible_AI_4.txt
|               
+---logs
|       events.out.tfevents.1751711683.Zain.5812.0
|       events.out.tfevents.1751711767.Zain.16636.0
|       events.out.tfevents.1751712054.Zain.10188.0
|       events.out.tfevents.1751712473.Zain.4076.0
|       events.out.tfevents.1751712639.Zain.13140.0
|       events.out.tfevents.1751712764.Zain.13624.0
|       events.out.tfevents.1751712873.Zain.432.0
|       events.out.tfevents.1751713813.Zain.8104.0
|       
+---models
|   \---final_model
|       |   config.json
|       |   merges.txt
|       |   model.safetensors
|       |   special_tokens_map.json
|       |   tokenizer_config.json
|       |   training_args.bin
|       |   vocab.json
|       |   
|       \---checkpoint-24
|               config.json
|               merges.txt
|               model.safetensors
|               optimizer.pt
|               rng_state.pth
|               scheduler.pt
|               special_tokens_map.json
|               tokenizer_config.json
|               trainer_state.json
|               training_args.bin
|               vocab.json
|               
+---scripts
|   |   label_cleaned_data.py
|   |   run_inference.py
|   |   
|   \---__pycache__
|           label_cleaned_data.cpython-312.pyc
|           run_inference.cpython-312.pyc
|           
\---src
    |   config.py
    |   data_loader.py
    |   
    +---labeling
    |   |   apply_labeling.py
    |   |   snorkel_rules.py
    |   |   
    |   \---__pycache__
    |           apply_labeling.cpython-312.pyc
    |           snorkel_rules.cpython-312.pyc
    |           
    +---models
    |   |   trainer.py
    |   |   
    |   \---__pycache__
    |           trainer.cpython-312.pyc
    |           
    +---preprocessing
    |       cleaner.py
    |       
    +---scraping
    |       govuk_scraper.py
    |       
    \---__pycache__
            config.cpython-312.pyc
            data_loader.cpython-312.pyc

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
