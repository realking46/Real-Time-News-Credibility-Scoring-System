from datetime import datetime, timedelta
import subprocess
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")


def run_command(command: str) -> None:
    result = subprocess.run(
        command,
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {command}")


default_args = {
    "owner": "nishant",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="news_credibility_live_pipeline",
    description="Runs live news ingestion, feature generation, prediction, and monitoring.",
    default_args=default_args,
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "news", "credibility"],
) as dag:

    ingest_live_news = PythonOperator(
        task_id="ingest_live_news",
        python_callable=run_command,
        op_kwargs={"command": "python -m src.ingestion.rss_ingest"},
    )

    build_live_features = PythonOperator(
        task_id="build_live_features",
        python_callable=run_command,
        op_kwargs={"command": "python -m src.features.build_live_features"},
    )

    predict_live_news = PythonOperator(
        task_id="predict_live_news",
        python_callable=run_command,
        op_kwargs={"command": "python -m src.inference.predict_live_news"},
    )

    monitor_predictions = PythonOperator(
        task_id="monitor_predictions",
        python_callable=run_command,
        op_kwargs={"command": "python -m src.monitoring.prediction_monitor"},
    )

    evidently_report = PythonOperator(
        task_id="evidently_report",
        python_callable=run_command,
        op_kwargs={"command": "python -m src.monitoring.evidently_report"},
    )

    ingest_live_news >> build_live_features >> predict_live_news >> monitor_predictions>> evidently_report