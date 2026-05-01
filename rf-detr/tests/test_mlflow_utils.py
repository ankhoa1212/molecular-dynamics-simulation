import mlflow
import pytest

from mlflow_utils import end_run, log_artifact, log_epoch_metrics, start_run


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    yield
    if mlflow.active_run():
        mlflow.end_run()


def test_start_run_activates_a_run():
    start_run("test-exp", "test-run", {})
    assert mlflow.active_run() is not None
    end_run()


def test_start_run_sets_run_name():
    start_run("test-exp", "my-run", {})
    assert mlflow.active_run().info.run_name == "my-run"
    end_run()


def test_start_run_logs_params():
    start_run("test-exp", "test-run", {"lr": "0.001", "epochs": "50"})
    run_id = mlflow.active_run().info.run_id
    end_run()
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params
    assert params["lr"] == "0.001"
    assert params["epochs"] == "50"


def test_log_epoch_metrics_records_values():
    start_run("test-exp", "test-run", {})
    log_epoch_metrics({"loss": 0.42, "mAP50": 0.81}, step=3)
    run_id = mlflow.active_run().info.run_id
    end_run()
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    assert abs(metrics["loss"] - 0.42) < 1e-5
    assert abs(metrics["mAP50"] - 0.81) < 1e-5


def test_end_run_closes_active_run():
    start_run("test-exp", "test-run", {})
    assert mlflow.active_run() is not None
    end_run()
    assert mlflow.active_run() is None


def test_log_artifact_logs_file(tmp_path):
    artifact_file = tmp_path / "checkpoint.txt"
    artifact_file.write_text("weights")
    start_run("test-exp", "test-run", {})
    run_id = mlflow.active_run().info.run_id
    log_artifact(str(artifact_file))
    end_run()
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    assert any(a.path == "checkpoint.txt" for a in artifacts)
