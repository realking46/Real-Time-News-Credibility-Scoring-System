from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data" / "feature_store" / "news_features.parquet"

MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "news_credibility_baseline"


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}. "
            "Run src/features/build_features.py first."
        )

    df = pd.read_parquet(FEATURE_PATH)

    df = df[["text", "label_id", "label"]].dropna()
    df = df[df["text"].str.len() > 20]

    X = df["text"]
    y = df["label_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = {
        "model_type": "tfidf_logistic_regression",
        "max_features": 10000,
        "classifier": "LogisticRegression",
        "test_size": 0.2,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        model.fit(X_train, y_train)

        local_model_path = PROJECT_ROOT / "models" / "baseline_model.joblib"
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, local_model_path)
        print(f"Local model saved to: {local_model_path}")

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        report = classification_report(y_test, y_pred, target_names=["fake", "real"])
        report_path = PROJECT_ROOT / "models" / "baseline_classification_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)

        mlflow.log_artifact(str(report_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="news_credibility_model",
        )

        print("Training completed.")
        print(metrics)
        print(report)


if __name__ == "__main__":
    main()