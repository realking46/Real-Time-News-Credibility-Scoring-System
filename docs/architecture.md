# Project Architecture

```mermaid
flowchart TD

    A1[LIAR Dataset] --> B1[Static Data Loader]
    A2[FakeNewsNet Dataset] --> B1

    L1[RSS Feeds] --> B2[Live RSS Ingestion]
    L2[NewsAPI Optional] --> B3[NewsAPI Ingestion]
    L3[Article URLs] --> B4[BeautifulSoup Scraper]

    B1 --> C[Feature Pipeline]
    B2 --> C2[Live Feature Pipeline]
    B3 --> C2
    B4 --> C2

    C --> D[Feature Store<br/>Parquet / CSV]
    C2 --> D

    D --> E[Training Pipeline]
    E --> F[MLflow Tracking]
    E --> G[MLflow Model Registry]
    E --> H[Local Model Artifact<br/>baseline_model.joblib]

    H --> I[Inference Pipeline]
    G --> I

    I --> J[FastAPI API]
    I --> K[Streamlit UI]
    I --> M[Live News Predictions]

    M --> N[Prediction Monitoring Report]
    D --> O[Evidently Drift Report]

    P[Apache Airflow DAG] --> B2
    P --> C2
    P --> M
    P --> N
    P --> O

    Q[Docker Compose] --> J
    Q --> K
    Q --> P

    R[HuggingFace Space] --> K2[Cloud Streamlit Demo]