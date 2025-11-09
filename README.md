# SHL - Generative AI Assignment

## 📄 Overview
This submission implements an end-to-end **Generative AI Recommendation System** for SHL’s product catalog.
It includes:
1. Web scraping and dataset preparation (`shl_catalog.csv`)
2. TF-IDF-based Recommendation Engine (`step3_recommender_tfidf_boosted_roleaware.py`)
3. REST API Service built using FastAPI (`app.py`)

The solution follows all requirements mentioned in the SHL assignment PDF:
- Generates Top-10 SHL product recommendations per query.
- Excludes “Pre-packaged Job Solutions”.
- Provides `/health` and `/recommend` endpoints.
- Evaluates Recall@10 on the Train Set.

---

## 🧰 Folder Structure
```
.
├── app.py                                     # FastAPI service
├── step3_recommender_tfidf_boosted_roleaware.py  # TF-IDF recommender (boosted + role-aware)
├── shl_catalog.csv                            # Scraped SHL catalog
├── Gen_AI Dataset.xlsx                        # Given dataset
├── predictions_test_tfidf.csv                 # Final test predictions
├── eval_train_recall_at_10_tfidf.txt          # Evaluation metrics
├── health.json                                # /health sample output
├── recommend_sample.json                      # /recommend sample output
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation
1. Clone or unzip the folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Step-by-Step Execution

### **Step 1 — Evaluate & Generate Predictions**
Run the recommender:
```bash
python step3_recommender_tfidf_boosted_roleaware.py --build-and-eval
python step3_recommender_tfidf_boosted_roleaware.py --predict-test
```
Outputs:
- `eval_train_recall_at_10_tfidf.txt`
- `predictions_test_tfidf.csv`

---

### **Step 2 — Launch API Service**
Start the FastAPI service:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open in browser:
- API Docs → http://127.0.0.1:8000/docs
- Health Check → http://127.0.0.1:8000/health

---

### **Step 3 — Example API Calls**

#### Health Check
**GET** `/health`  
**Response:**
```json
{
  "status": "ok",
  "catalog_size": 528
}
```

#### Recommendation
**POST** `/recommend`  
**Body:**
```json
{
  "query": "Hiring Java developer who can collaborate with business teams",
  "topk": 10
}
```

**Response:**
```json
{
  "query": "Hiring Java developer who can collaborate with business teams",
  "topk": 10,
  "results": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/java-8-new/", "score": 0.85},
    ...
  ]
}
```

---

## 📊 Evaluation
- **Mean Recall@10:** 0.04  
- **Train Queries Evaluated:** 10  
- **Test Predictions Generated:** 9 × 10 = 90 rows  
- “Pre-packaged Job Solutions” filtered: ✅  
- Output format matches PDF submission format.

---

## 🧠 Key Design Features
- **TF-IDF (1–3 grams)** for short title matching  
- **Skill-term boosting** for relevant skills (Java, SQL, Python, etc.)  
- **Role-aware penalty** for deprioritizing generic “communication” items in engineering queries  
- **Lightweight dependencies** (no deep learning)

---

## ✅ Final Deliverables
| File | Description |
|------|--------------|
| `shl_catalog.csv` | Scraped SHL test catalog |
| `Gen_AI Dataset.xlsx` | Provided input dataset |
| `step3_recommender_tfidf_boosted_roleaware.py` | Model building + evaluation script |
| `app.py` | FastAPI app for /health and /recommend |
| `predictions_test_tfidf.csv` | Model output for test queries |
| `eval_train_recall_at_10_tfidf.txt` | Evaluation metric |
| `requirements.txt` | Python environment dependencies |
| `README.md` | Run instructions |
| `health.json`, `recommend_sample.json` | Sample API outputs |

---

## 🧑‍💻 Author
**Name:** Gaurav Raj  
**Institute:** NIT Karnataka  
**Role:** AI/ML Research Intern | Data Science Enthusiast
