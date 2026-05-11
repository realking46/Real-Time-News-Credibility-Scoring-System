from pathlib import Path

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "live_news_predictions.csv"

API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="News Credibility Scoring System",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Real-Time News Credibility Scoring System")

tab1, tab2 = st.tabs(["Check Custom News", "Latest Live News Scores"])


with tab1:
    st.subheader("Check Custom News Text")

    text = st.text_area(
        "News text",
        height=220,
        placeholder="Paste news article text here..."
    )

    if st.button("Check Credibility"):
        if not text.strip():
            st.warning("Please enter some news text.")
        else:
            with st.spinner("Analyzing credibility..."):
                response = requests.post(
                    API_URL,
                    json={"text": text},
                    timeout=30,
                )

            if response.status_code == 200:
                result = response.json()

                col1, col2, col3 = st.columns(3)

                col1.metric("Credibility Score", result["credibility_score"])
                col2.metric("Risk Level", result["risk_level"])
                col3.metric("Prediction Label", result["prediction_label"])

                if result["risk_level"] == "Low":
                    st.success("This article appears relatively credible.")
                elif result["risk_level"] == "Medium":
                    st.warning("This article has moderate credibility risk.")
                else:
                    st.error("This article has high credibility risk.")
            else:
                st.error("API error. Make sure FastAPI is running.")


with tab2:
    st.subheader("Latest Live RSS News Predictions")

    if not PREDICTIONS_PATH.exists():
        st.warning(
            "No live predictions found. Run the live pipeline first:\n\n"
            "`python -m src.ingestion.rss_ingest`\n\n"
            "`python -m src.features.build_live_features`\n\n"
            "`python -m src.inference.predict_live_news`"
        )
    else:
        df = pd.read_csv(PREDICTIONS_PATH)

        st.write(f"Total live articles scored: **{len(df)}**")

        show_cols = [
            "title",
            "source_name",
            "credibility_score",
            "risk_level",
            "prediction_label",
            "url",
        ]

        existing_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[existing_cols], use_container_width=True)

        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        st.subheader("Risk Level Distribution")
        st.bar_chart(risk_counts.set_index("risk_level"))