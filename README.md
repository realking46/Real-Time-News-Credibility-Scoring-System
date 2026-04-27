# 📰 Real-Time News Credibility Scoring System

> **MLOps Individual Project** — Nishant Singh  
> A fully automated Feature–Training–Inference (FTI) pipeline that scores news article credibility in real time.

---

## 🎯 What It Does

Given a news article (text or URL), the system outputs:
- **Credibility Score** — 0 to 100
- **Risk Label** — Low / Medium / High

---

## 🏗️ Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Feature Pipeline  │────▶│  Training Pipeline  │────▶│ Inference Pipeline  │
│                     │     │                     │     │                     │
│ • NewsAPI / RSS     │     │ • Reads Parquet      │     │ • Streamlit UI      │
│ • LIAR / FakeNews   │     │ • Fine-tunes BERT    │     │ • Loads best model  │
│ • BERT embeddings   │     │ • Logs to MLflow     │     │ • Returns score +   │
│ • Saves Parquet     │     │ • Registers model    │     │   risk label        │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         ▲                            ▲                            │
         │                            │                            │
    [Airflow DAG]              [Airflow DAG]                  [On Demand]
    (Daily)                    (Weekly)
```

---

## 🗂️ Project Structure

```
├── feature_pipeline/      # Ingest → compute features → store Parquet
├── training_pipeline/     # Train BERT → evaluate → register to MLflow
├── inference_pipeline/    # Load model → predict → Streamlit UI
├── feature_store/         # Versioned Parquet files (gitignored)
├── dags/                  # Airflow DAGs for scheduling
├── docker/                # Dockerfiles + docker-compose.yml
├── tests/                 # pytest test suite
├── .github/workflows/     # GitHub Actions CI
├── .env.example           # Environment variable template
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & configure
```bash
git clone https://github.com/realking46/Real-Time-News-Credibility-Scoring-System.git
cd Real-Time-News-Credibility-Scoring-System
cp .env.example .env
# Edit .env and add your NewsAPI key
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the full stack with Docker
```bash
docker-compose -f docker/docker-compose.yml up
```

### 4. Run pipelines individually
```bash
# Backfill historical data (run once)
python feature_pipeline/backfill.py

# Run feature pipeline
python feature_pipeline/ingest.py

# Run training pipeline
python training_pipeline/train.py

# Launch Streamlit UI
streamlit run inference_pipeline/app.py
```

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10 |
| Data Ingestion | NewsAPI, BeautifulSoup, feedparser |
| Feature Store | Versioned Parquet (pyarrow) |
| Model | BERT (`bert-base-uncased`) |
| Training | PyTorch + HuggingFace Transformers |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| CI | GitHub Actions |

---

## 📊 Features Computed

| Feature | Description |
|---|---|
| BERT embedding | 768-dim `[CLS]` token vector |
| Article length | Word count |
| Sensational word count | Keywords like "breaking", "shocking", "exclusive" |
| Source reliability score | Domain-based lookup table |
| Publication recency | Hours since published |
| Credibility score (label) | Weighted aggregation used for training |

---

## 🔑 Environment Variables

Copy `.env.example` → `.env` and fill in:

| Variable | Description |
|---|---|
| `NEWS_API_KEY` | Free key from [newsapi.org](https://newsapi.org) |
| `MLFLOW_TRACKING_URI` | Default: `http://localhost:5000` |
| `FEATURE_STORE_PATH` | Default: `./feature_store` |
| `MLFLOW_MODEL_NAME` | Default: `news-credibility-scorer` |

---

## ✅ CI Status

![CI](https://github.com/realking46/Real-Time-News-Credibility-Scoring-System/actions/workflows/ci.yml/badge.svg)

---

## 📄 License

MIT
