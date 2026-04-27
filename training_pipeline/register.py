"""
training_pipeline/register.py
──────────────────────────────
Compares all runs in the 'news-credibility' experiment and promotes
the best one (lowest val_loss) to the "Production" stage in the
MLflow Model Registry.

Usage:
    python -m training_pipeline.register
    python -m training_pipeline.register --metric val_mae   # use MAE instead
    python -m training_pipeline.register --dry-run          # report only
"""

import argparse
import logging
import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "news-credibility-scorer")
EXPERIMENT_NAME     = "news-credibility"


def get_best_run(metric: str = "val_loss") -> dict | None:
    """
    Search all FINISHED runs in the experiment and return the one
    with the lowest value of `metric`.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        log.error(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )

    if not runs:
        log.error("No finished runs found.")
        return None

    best = runs[0]
    log.info(
        f"Best run → ID: {best.info.run_id}  "
        f"{metric}: {best.data.metrics.get(metric, 'N/A')}"
    )
    return {
        "run_id":  best.info.run_id,
        "metrics": best.data.metrics,
        "params":  best.data.params,
    }


def promote_to_production(run_id: str, dry_run: bool = False) -> None:
    """
    Find the model version registered from `run_id` and transition it
    to the 'Production' stage. All other versions are moved to 'Archived'.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Find model version for this run
    versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    target_version = None

    for v in versions:
        if v.run_id == run_id:
            target_version = v
            break

    if target_version is None:
        log.error(
            f"No model version found for run_id={run_id}. "
            "Make sure train.py registered the model."
        )
        return

    log.info(
        f"Target version: {target_version.version}  "
        f"(current stage: {target_version.current_stage})"
    )

    if dry_run:
        log.info("[DRY RUN] Would promote version to Production. No changes made.")
        return

    # Archive all other Production versions first
    for v in versions:
        if v.current_stage == "Production" and v.version != target_version.version:
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=v.version,
                stage="Archived",
            )
            log.info(f"Archived previous Production version: {v.version}")

    # Promote target to Production
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=target_version.version,
        stage="Production",
        archive_existing_versions=False,
    )
    log.info(
        f"✅ Promoted '{MLFLOW_MODEL_NAME}' version {target_version.version} "
        f"→ Production"
    )

    # Add description tag
    client.update_model_version(
        name=MLFLOW_MODEL_NAME,
        version=target_version.version,
        description=(
            f"Best model from run {run_id}. "
            f"val_loss={target_version.run_id}"
        ),
    )


def run_registration(metric: str = "val_loss", dry_run: bool = False) -> None:
    best = get_best_run(metric=metric)
    if best is None:
        return

    log.info(f"Best run metrics: {best['metrics']}")
    promote_to_production(run_id=best["run_id"], dry_run=dry_run)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Promote best MLflow model to Production.")
    p.add_argument(
        "--metric",
        default="val_loss",
        help="Metric to rank runs by (lower = better). Default: val_loss",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the best run without making any registry changes.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_registration(metric=args.metric, dry_run=args.dry_run)
