
from datetime import datetime

from airflow import DAG
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator

PROJECT_ID = "graphic-outlook-489716-n6"
REGION = "europe-west1"
JOB_NAME = "news-credibility-live-pipeline"

with DAG(
    dag_id="news_credibility_cloud_pipeline",
    start_date=datetime(2026, 5, 30),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "news-credibility", "cloud-run"],
) as dag:

    run_live_pipeline = CloudRunExecuteJobOperator(
        task_id="run_cloud_run_live_pipeline_job",
        project_id=PROJECT_ID,
        region=REGION,
        job_name=JOB_NAME,
    )
