# Real-Time News Credibility Scoring System

A live MLOps project that scores the credibility of news articles using automated Feature–Training–Inference pipelines.

The system ingests static and live news data, builds features, trains models, tracks experiments with MLflow, serves predictions through FastAPI and Streamlit, automates live scoring with Airflow, and provides monitoring reports.

---

## 1. Project Overview

Online misinformation spreads quickly, and users often need a simple way to estimate whether a news article or claim appears credible.

This project builds a **cloud-ready live ML system** instead of a static notebook. The main goal is not maximum model accuracy, but a complete MLOps pipeline that can:

- ingest dynamic news data,
- process and store features,
- train and track models,
- serve predictions through a UI,
- automate recurring pipeline runs,
- monitor predictions and drift.

Given a news article or claim, the system returns:

- **Credibility Score:** 0–100
- **Risk Level:** Low / Medium / High
- **Prediction Label:** real / fake

## Architecture Diagram

The project follows the Feature–Training–Inference architecture.

See the full diagram here:

[Project Architecture](docs/architecture.md)
![Architecture Diagram](docs/images/Live%20News%20Prediction-2026-05-11-114240.png)

---

## 2. Course Requirement Alignment

| Requirement | Project Implementation |
|---|---|
| Live ML system | Live RSS ingestion + NewsAPI module |
| Dynamic data | RSS feeds, NewsAPI, optional scraping |
| FTI architecture | Feature, Training, Inference pipelines |
| Feature Store | Versioned Parquet/CSV feature store |
| Model tracking | MLflow experiments and model registry |
| Automation | Apache Airflow DAG |
| Reproducibility | Docker, requirements, README commands |
| UI demo | Streamlit interface |
| API inference | FastAPI endpoint |
| Monitoring | Prediction monitoring + Evidently drift report |
| Code quality | Modular source code + unit tests |
| CI/CD readiness | GitHub Actions workflow |

---

## 3. Architecture

## 4. FTI Pipeline Design

The project follows the Feature–Training–Inference (FTI) architecture.

#### 4.1 Feature Pipeline

The feature pipeline:

 - loads static datasets,
 - ingests live RSS news,
 - supports NewsAPI ingestion,
 - supports BeautifulSoup-based article scraping,
 - standardizes text data,
 - computes engineered features,
 - stores features in Parquet and CSV format.

```
src/ingestion/load_static_data.py
src/ingestion/rss_ingest.py
src/ingestion/newsapi_ingest.py
src/ingestion/article_scraper.py
src/features/build_features.py
src/features/build_live_features.py
```
#### 4.2 Training Pipeline

The training pipeline:

 - reads data from the feature store,
 - trains a baseline classifier,
 - logs metrics and parameters to MLflow,
 - registers models in MLflow,
 - saves a local model artifact for stable inference.

Main files:
```
src/training/train_baseline.py
src/training/train_bert.py
```

#### 4.3 Inference Pipeline

The inference pipeline:

 - loads the trained model,
 - predicts credibility,
 - returns score, risk level, and label,
 - exposes a FastAPI endpoint,
 - provides a Streamlit UI.

Main files:
```
src/inference/predict.py
src/inference/predict_live_news.py
src/inference/api.py
src/inference/predict_bert.py
app/streamlit_app.py
app/huggingface_app.py     
```

#### 4.4 Monitoring Pipeline

The monitoring pipeline:

 - summarizes prediction distributions,
 - tracks risk-level distribution,
 - creates markdown monitoring reports,
 - optionally creates Evidently drift reports.

Main files:
```
src/monitoring/prediction_monitor.py
src/monitoring/evidently_report.py
```

## 5. Tech Stack

| Component            | Tools                               |
| -------------------- | ----------------------------------- |
| Programming Language | Python                              |
| Static Data          | LIAR, FakeNewsNet                   |
| Live Data            | RSS feeds, NewsAPI                  |
| Web Scraping         | Requests, BeautifulSoup             |
| Data Processing      | Pandas, NumPy                       |
| Feature Store        | Parquet, CSV                        |
| Baseline Model       | TF-IDF + Logistic Regression        |
| Optional Model       | BERT / bert-base-uncased            |
| Training Framework   | scikit-learn, PyTorch, Transformers |
| Tracking             | MLflow                              |
| API                  | FastAPI                             |
| UI                   | Streamlit                           |
| Automation           | Apache Airflow                      |
| Monitoring           | MLflow, custom reports, Evidently   |
| Containerization     | Docker, Docker Compose              |
| Testing              | pytest                              |
| CI/CD                | GitHub Actions                      |
| Cloud Demo           | HuggingFace Spaces                  |

## 6. Repository Structure
```text
real-time-news-credibility/
│
├── app/
│   ├── streamlit_app.py
│   └── huggingface_app.py
│
├── airflow/
│   └── dags/
│       └── news_credibility_pipeline.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── feature_store/
│
├── models/
│
├── reports/
│   ├── monitoring/
│   └── evidently/
│
├── src/
│   ├── ingestion/
│   │   ├── load_static_data.py
│   │   ├── rss_ingest.py
│   │   ├── newsapi_ingest.py
│   │   └── article_scraper.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── build_live_features.py
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   └── train_bert.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   ├── predict_live_news.py
│   │   ├── predict_bert.py
│   │   └── api.py
│   │
│   ├── monitoring/
│   │   ├── prediction_monitor.py
│   │   └── evidently_report.py
│   │
│   └── utils/
│
├── tests/
│   ├── test_features.py
│   ├── test_prediction_format.py
│   └── test_ingestion_utils.py
│
├── Dockerfile
├── docker-compose.yml
├── docker-compose.airflow.yml
├── requirements.txt
├── pytest.ini
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```
## 7. Data Sources
#### 7.1 Static Training Data

The baseline model is trained using static datasets:

LIAR Dataset
Contains political claims with truthfulness labels.
Original labels are mapped to binary classes.
FakeNewsNet
Contains real and fake news articles from PolitiFact and GossipCop.
#### 7.2 Live Data

The live pipeline uses:

BBC RSS feeds,
Reuters-style RSS feeds,
Guardian RSS feeds,
optional NewsAPI ingestion,
optional BeautifulSoup article scraping.

RSS is used as the primary dynamic data source because it is stable and does not require an API key.

## 8. Label Mapping

The LIAR dataset contains six labels:
```
true
mostly-true
half-true
barely-true
false
pants-fire
```

They are mapped to binary labels:
| Original Label | Binary Label |
| -------------- | ------------ |
| true           | real         |
| mostly-true    | real         |
| half-true      | real         |
| barely-true    | fake         |
| false          | fake         |
| pants-fire     | fake         |

The final binary format is:

| Label | Label ID |
| ----- | -------- |
| fake  | 0        |
| real  | 1        |

## 9. Feature Engineering

The project computes lightweight, reproducible text features.

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| `text_length`            | Number of characters         |
| `word_count`             | Number of words              |
| `sentence_count`         | Number of sentences          |
| `avg_word_length`        | Average word length          |
| `exclamation_count`      | Number of exclamation marks  |
| `uppercase_ratio`        | Ratio of uppercase letters   |
| `punctuation_ratio`      | Ratio of punctuation symbols |
| `sensational_word_count` | Count of sensational words   |
| `title_length`           | Title character length       |
| `has_url`                | Whether URL exists           |

These features are stored in:
```
data/feature_store/news_features.parquet
data/feature_store/live_news_features.parquet
```

## 10. Models
#### 10.1 Production Baseline Model

The main stable model is:
```
TF-IDF Vectorizer + Logistic Regression
```

Why this model is used:

 - fast to train,
 - lightweight,
 - reproducible,
 - easy to deploy,
 - stable inside Docker and Airflow.

This is the main model used by:

- FastAPI,
- Streamlit,
- Airflow live prediction pipeline,
- HuggingFace demo.

#### 10.2 Optional BERT Experiment

The original proposal listed BERT as the intended model.\
The project includes an optional BERT experiment using:

```
bert-base-uncased
```

The BERT model is implemented as an additional MLflow experiment.
Run:
```
python -m src.training.train_bert
```
BERT is not used as the default deployed model because the course evaluates the MLOps pipeline more than raw accuracy, and the baseline model is more stable for automation and deployment.

## 11. Installation
#### 11.1 Clone Repository
```
git clone <your-repository-url>
cd real-time-news-credibility
```

#### 11.2 Create Environment
```
python -m venv .venv
.venv\Scripts\activate
```

#### 11.3 Install Dependencies
```
pip install -r requirements.txt
```

## 12. Running the Project Locally
#### 12.1 Run Static Data Pipeline

```
python -m src.ingestion.load_static_data
python -m src.features.build_features
```

Expected outputs:
```
data/processed/combined_news_dataset.parquet
data/feature_store/news_features.parquet
```

#### 12.2 Train Baseline Model
```
python -m src.training.train_baseline
```

Expected outputs:
```
models/baseline_model.joblib
mlruns/
```

#### 12.3 Run Live News Pipeline
```
python -m src.ingestion.rss_ingest
python -m src.features.build_live_features
python -m src.inference.predict_live_news
python -m src.monitoring.prediction_monitor
```

Expected Outputs:
```
data/raw/live/live_news_latest.parquet
data/feature_store/live_news_features.parquet
data/processed/live_news_predictions.csv
reports/monitoring/prediction_monitoring_report.md
```

#### 12.4 Generate Evidently Drift Report
```
python -m src.monitoring.evidently_report
```
Output:
```
reports/evidently/data_drift_report.html
```

## 13. Run FastAPI Inference API
```
uvicorn src.inference.api:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

Example request
```
{
  "text": "The government confirmed the new policy in an official statement."
}
```
Response
```
{
  "prediction_label": "real",
  "credibility_score": 80,
  "risk_level": "Low"
}
```

## 14. Run Streamlit UI
```
streamlit run app/streamlit_app.py
```

check open:
```
http://localhost:8501
```

The UI supports:
 - custom text prediction,
 - latest live RSS article scores,
 - risk distribution visualization.

15. Run with Docker
Build and run Api and UI
```
docker compose up --build
```

Open:
```
FastAPI:   http://localhost:8000/docs
Streamlit: http://localhost:8501
```

## 16. Run Airflow Automation
```
docker compose -f docker-compose.airflow.yml up --build
```

```
http://localhost:8080
```

Login credentials:
```
Username: admin
Password: admin
```
Triger DAG
Dag runs
 - ingest_live_news
 - build_live_features
 - predict_live_news
 - monitor_predictions
 - evidently_report

## 17. MLflow Tracking
Start MLflow UI:

```
mlflow ui --backend-store-uri ./mlruns
```

open:
```
http://127.0.0.1:5000
```

Tracked information includes:
 - model parameters,
 - accuracy,
 - F1 score,
 - precision,
 - recall,
 - classification report,
 - model artifacts,
 - model versions.

## 18. Optional NewsAPI Ingestion
Create a .env file:
```
NEWSAPI_KEY=your_newsapi_key_here
```

```
python -m src.ingestion.newsapi_ingest
```
Output:
```
data/raw/newsapi/newsapi_latest.parquet
```
NewsAPI is optional. RSS feeds are used as the primary dynamic data source.

## 19. Optional Article Scraping
```
python -m src.ingestion.article_scraper "https://www.bbc.com/news"
```

Output:
```
data/raw/scraped/scraped_articles_latest.parquet
```

The scraper uses:
 - Requests,
 - BeautifulSoup,
 - polite user-agent headers.

## 20. Unit Tests
```
python -m pytest
```

The tests cover:

feature engineering utilities,\
prediction formatting,\
ingestion ID generation.

Test files:
```
tests/test_features.py
tests/test_prediction_format.py
tests/test_ingestion_utils.py
```

## 21. GitHub Actions CI
The project includes a CI workflow:
```
.github/workflows/ci.yml
```
The workflow:
 - installs dependencies,
 - checks imports,
 - runs unit tests,
 - builds Docker image.

This supports reproducibility, automation, and code quality.

## 22. HuggingFace Cloud Demo
A lightweight Streamlit-only demo is prepared for HuggingFace Spaces.\
The HuggingFace version uses:

```
app.py
Dockerfile
requirements.txt
models/baseline_model.joblib
```
Link to huggingface space:
```
https://huggingface.co/spaces/Realking46/news-credibility-scoring-system
```
The deployed Space demonstrates inference using the trained baseline model.
The full local system remains the main MLOps implementation because it includes:
 - FastAPI,
 - Airflow,
 - MLflow,
 - Docker Compose,
 - monitoring,
 - live pipelines.

## 23. Main Output Files
| Output               | Path                                                 |
|---|---|
| Combined dataset     | `data/processed/combined_news_dataset.parquet`       |
| Static feature store | `data/feature_store/news_features.parquet`           |
| Live RSS data        | `data/raw/live/live_news_latest.parquet`             |
| Live feature store   | `data/feature_store/live_news_features.parquet`      |
| Live predictions     | `data/processed/live_news_predictions.csv`           |
| Monitoring report    | `reports/monitoring/prediction_monitoring_report.md` |
| Evidently report     | `reports/evidently/data_drift_report.html`           |
| Local model          | `models/baseline_model.joblib`                       |
| MLflow runs          | `mlruns/`                                            |

## 24. Screenshots

| Screenshot              | Purpose                |
| ----------------------- | ---------------------- |
| Streamlit UI            | Live UI demo           |
| FastAPI `/docs`         | API endpoint proof     |
| MLflow experiment       | Tracking proof         |
| MLflow registered model | Registry proof         |
| Airflow green DAG       | Automation proof       |
| Docker containers       | Reproducibility proof  |
| HuggingFace Space       | Cloud demo proof       |
| Monitoring report       | Monitoring proof       |
| Evidently report        | Drift monitoring proof |

## 25. Known Limitations
 - The baseline model is simple and not optimized for maximum accuracy.
 - The live RSS data does not contain ground-truth labels.
 - The credibility score is derived from model output and confidence logic, not human fact-checking.
 - BERT is included as an optional experiment, not the default production model.
 - NewsAPI requires an external API key.
 - The HuggingFace Space is a lightweight inference demo, not the full Airflow/MLflow system.

## 26. Future Improvements

Possible extensions:
 - Deploy the full FastAPI + Streamlit system to a cloud service.
 - Use DistilBERT as the default production model.
 - Add a real feature store such as Feast or Hopsworks.
 - Store features in S3 or cloud object storage.
 - Add scheduled GitHub Actions retraining.
 - Add stronger drift monitoring with Evidently dashboards.
 - Add model promotion logic using MLflow aliases.
 - Add confidence calibration for credibility scores.
 - Add source-level credibility features.
 - Add more robust scraping with Playwright.

## 27. Project Summary

This project demonstrates a complete live MLOps system for news credibility scoring.

It includes:
 - dynamic data ingestion,
 - feature engineering,
 - model training,
 - MLflow experiment tracking,
 - model registry usage,
 - FastAPI inference,
 - Streamlit UI,
 - Airflow automation,
 - Docker reproducibility,
 - monitoring reports,
 - unit tests,
 - CI workflow,
 - HuggingFace cloud demo.

The project follows the course objective of moving from static notebooks to an automated, live, cloud-ready machine learning system.

## 28. Author
Nishant Singh\
HSLU MLOps Project\
Spring 2026
