import os
import argparse
from pathlib import Path

import yaml
import mlflow

from dataset import split_by_experiment


def load_config(path: str) -> dict:
    with open(path) as config_file:
        return yaml.safe_load(config_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF-DETR particle detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    # Point to the shared database in the data-setup directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "data-setup", "mlflow.db")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db_path}"

    splits = split_by_experiment(
        dataset_path=ds_cfg["path"],
        train_experiments=ds_cfg["train_experiments"],
        val_experiments=ds_cfg["val_experiments"],
        test_experiments=ds_cfg["test_experiments"],
    )

    # rfdetr expects dataset_dir to contain train/ and valid/ subdirectories
    dataset_dir = splits.train_dir.parent

    variant = model_cfg["variant"].lower()
    if variant == "base":
        from rfdetr import RFDETRBase

        model = RFDETRBase()
    elif variant == "large":
        from rfdetr import RFDETRLarge

        model = RFDETRLarge()
    else:
        raise ValueError(f"Unknown model variant {variant!r}. Choose 'base' or 'large'.")

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train using rfdetr's built-in MLflow support
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        grad_accum_steps=train_cfg["grad_accum_steps"],
        lr=train_cfg["learning_rate"],
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory", False),
        output_dir=str(checkpoint_dir),
        # Built-in MLflow integration
        mlflow=True,
        project=mlflow_cfg["experiment_name"],
        run=f"train-rfdetr-{model_cfg['variant']}",
    )


if __name__ == "__main__":
    main()
