from typing import Any

import mlflow


def start_run(experiment_name: str, run_name: str, params: dict[str, Any]) -> mlflow.ActiveRun:
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    return run


def log_epoch_metrics(metrics: dict[str, float], step: int) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)


def end_run() -> None:
    mlflow.end_run()
