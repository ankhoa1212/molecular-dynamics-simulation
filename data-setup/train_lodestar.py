"""Train a LodeSTAR model and save weights + companion JSON config."""
import argparse
import dataclasses
import glob
import json
import logging
import os
import random

import mlflow
import numpy as np
import torch
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger
from tqdm import tqdm

import deeplay as dl
import deeptrack as dt
from pytorch_lightning.callbacks import Callback

logging.getLogger("pint").setLevel(logging.ERROR)


class _DualEarlyStopping(Callback):
    """Stop training once both monitored metrics have stopped improving."""

    def __init__(self, metrics, patience=10, min_delta=1e-4):
        self._metrics = metrics
        self._patience = patience
        self._min_delta = min_delta
        self._best = {m: float("inf") for m in metrics}
        self._wait = {m: 0 for m in metrics}

    def on_train_epoch_end(self, trainer, pl_module):
        all_plateaued = True
        for metric in self._metrics:
            current = trainer.callback_metrics.get(metric)
            if current is None:
                all_plateaued = False
                continue
            current = float(current)
            if current < self._best[metric] - self._min_delta:
                self._best[metric] = current
                self._wait[metric] = 0
                all_plateaued = False
            else:
                self._wait[metric] += 1
                if self._wait[metric] < self._patience:
                    all_plateaued = False

        if all_plateaued:
            print(
                f"\nEarly stopping: both losses plateaued for {self._patience} epochs."
            )
            trainer.should_stop = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LodeSTAR model and save it for later inference."
    )
    parser.add_argument(
        "--input-dir", type=str,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--input-file", type=str,
        help="Path to a single input image.",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help=(
            "Where to save the trained model (.pt). "
            "Defaults to lodestar_model.pt next to the input folder."
        ),
    )
    parser.add_argument(
        "--num-outputs", type=int, default=3,
        help="Number of LodeSTAR output channels. 2=(x,y); 3=(x,y,radius).",
    )
    parser.add_argument(
        "--n-transforms", type=int, default=8,
        help="Number of geometric transforms for LodeSTAR equivariance.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of max epochs for training LodeSTAR.",
    )
    parser.add_argument(
        "--crop-size", type=int, default=64,
        help="Resize all loaded crops to this square size before training (default: 64).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of DataLoader worker processes. 0 is safest.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--experiment", type=str, default="lodestar",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="MLflow run name. Defaults to the model filename stem.",
    )
    parser.add_argument(
        "--mlflow-uri", type=str, default="mlruns",
        help="MLflow tracking URI (local path or remote).",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help=(
            "Early-stopping patience: stop when both within_image_disagreement "
            "and between_image_disagreement show no improvement for this many epochs."
        ),
    )
    parser.add_argument(
        "--min-delta", type=float, default=1e-4,
        help="Minimum decrease in a loss to count as an improvement.",
    )
    parser.add_argument(
        "--dataset-length", type=int, default=128,
        help="Number of augmented samples generated per crop per epoch.",
    )
    return parser.parse_args()


def _load_and_normalise(image_files):
    raw_images = []
    for fpath in image_files:
        img = Image.open(fpath).convert("L")
        raw_images.append(np.array(img))
    data_raw = np.array(raw_images)
    data = np.zeros_like(data_raw, dtype=np.float32)
    for idx, frame in enumerate(data_raw):
        frame_f = frame.astype(np.float32)
        f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
        data[idx] = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min
    return data


def _load_crops(image_files, crop_size=64):
    """Load, resize to crop_size×crop_size, and normalise crop images."""
    crops = []
    for fpath in image_files:
        img = Image.open(fpath).convert("L").resize((crop_size, crop_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr_ptp = np.ptp(arr)
        crops.append((arr - arr.min()) / (arr_ptp if arr_ptp else 1.0))
    return crops


def _make_pipeline(crop_array):
    channel_crop = np.expand_dims(crop_array, axis=-1)  # (H, W, 1)
    return (
        dt.Value(channel_crop)
        # Wider intensity range to cover dim/bright particle variation
        >> dt.Multiply(lambda: np.random.uniform(0.4, 1.6))
        >> dt.Add(lambda: np.random.uniform(-0.15, 0.15))
        # Rotation + scale jitter + small translation for tighter centering
        >> dt.Affine(
            rotation=lambda: np.random.uniform(0, 2 * np.pi),
            scale=lambda: np.random.uniform(0.8, 1.2),
            translate=lambda: (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)),
        )
        >> dt.FlipLR(p=0.5)
        >> dt.FlipUD(p=0.5)
        # Defocus blur variation
        >> dt.Gaussian(sigma=lambda: np.random.uniform(0, 2.0))
        # Shot noise (Poisson) — fundamental to optical microscopy
        >> dt.Poisson(snr=lambda: np.random.uniform(3, 20))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )


def _unique_model_path(path):
    """Return path unchanged if it doesn't exist; otherwise append _1, _2, … until free."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _collect_image_files(args):
    if args.input_file:
        if os.path.exists(args.input_file):
            return [args.input_file]
        print(f"File not found: {args.input_file}")
        return []
    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]


def _build_and_train(args, crops_data):
    if not crops_data:
        raise ValueError("No crops found — run crop_tool.py to create crops first.")

    print(f"Training LodeSTAR on {len(crops_data)} crops.")

    lodestar = dl.LodeSTAR(
        n_transforms=args.n_transforms,
        num_outputs=args.num_outputs,
        optimizer=dl.Adam(lr=1e-3),
    ).build()

    datasets = [
        dt.pytorch.Dataset(_make_pipeline(crop), length=args.dataset_length, replace=False)
        for crop in crops_data
    ]
    dataloader = dl.DataLoader(
        dataset=torch.utils.data.ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment,
        tracking_uri=args.mlflow_uri,
        run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None,
    )
    early_stop = _DualEarlyStopping(
        metrics=["within_image_disagreement", "between_image_disagreement"],
        patience=args.patience,
        min_delta=args.min_delta,
    )
    dl.Trainer(
        accelerator="auto",
        precision="16-mixed",
        max_epochs=args.epochs,
        log_every_n_steps=10,
        logger=mlf_logger,
        callbacks=[early_stop],
    ).fit(lodestar, dataloader)
    return lodestar


def main():
    args = parse_args()

    if not args.input_dir and not args.input_file:
        raise ValueError("Either --input-dir or --input-file must be provided.")

    if args.model_path is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        args.model_path = os.path.join(models_dir, "lodestar_model.pt")

    args.model_path = _unique_model_path(args.model_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = args.input_dir if args.input_dir else os.path.dirname(os.path.abspath(args.input_file))
    crops_dir = os.path.join(base_dir, "crops")
    if not os.path.isdir(crops_dir):
        print(f"Crops directory not found: {crops_dir}")
        print("Use crop_tool.py to manually create crops before training.")
        return

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    crop_files = sorted(
        os.path.join(crops_dir, fn)
        for fn in os.listdir(crops_dir)
        if os.path.splitext(fn)[1].lower() in valid_exts
    )
    if not crop_files:
        print(f"No images found in crops directory: {crops_dir}")
        return
    print(f"Found {len(crop_files)} crop image(s) in {crops_dir}")

    run_name = args.run_name or os.path.splitext(os.path.basename(args.model_path))[0]
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": args.epochs,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "n_transforms": args.n_transforms,
            "num_outputs": args.num_outputs,
            "seed": args.seed,
            "num_crops": len(crop_files),
            "crops_dir": crops_dir,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "dataset_length": args.dataset_length,
        })

        crops_data = _load_crops(crop_files, crop_size=args.crop_size)
        lodestar = _build_and_train(args, crops_data)

        # Save weights
        torch.save(lodestar.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

        # Save companion JSON so label_images.py can reconstruct the architecture
        config = {
            "n_transforms": args.n_transforms,
            "num_outputs": args.num_outputs,
        }
        config_path = os.path.splitext(args.model_path)[0] + ".json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")

        mlflow.log_artifact(args.model_path)
        mlflow.log_artifact(config_path)
        print(f"MLflow run '{run_name}' logged to experiment '{args.experiment}'.")


if __name__ == "__main__":
    main()
