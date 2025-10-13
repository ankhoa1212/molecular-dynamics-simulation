"""Starter script for training, evaluating, a YOLO11n model on a custom dataset"""

import json
import csv
from pathlib import Path
import torch
from ultralytics import YOLO

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the custom dataset
train_results = model.train(
    data="data/data.yaml",  # Path to dataset configuration file
    epochs=1,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cuda" if torch.cuda.is_available() else "cpu",  # Use cuda if available
    name="yolov11n-custom",
    task="detect",
)
print("Training finished")


def save_and_plot_metrics(metrics) -> None:
    """Save and plot metrics from training or evaluation."""
    # Create output directory
    out_dir = Path("runs/metrics/starter")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw metrics object as repr() in a text file for quick inspection
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(repr(metrics))

    # If metrics is a dict-like object, save as JSON/CSV and plot key scalars
    try:
        # Try to coerce to dict
        m = dict(metrics) if hasattr(metrics, "items") else metrics
    except (TypeError, ValueError, AttributeError):
        m = None

    if isinstance(m, dict):
        # Save JSON
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, default=str)

        # Save CSV (flat key,value rows)
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in m.items():
                # Try to stringify values that are lists/tuples/numpy arrays
                try:
                    val = (
                        float(v)
                        if (
                            isinstance(v, (int, float))
                            or (hasattr(v, "dtype") and getattr(v, "shape", None) == ())
                        )
                        else str(v)
                    )
                except (TypeError, ValueError, AttributeError):
                    val = str(v)
                writer.writerow([k, val])

        # Quick plot for numeric top-level metrics
        if plt is not None:
            keys = []
            vals = []
            for k, v in m.items():
                try:
                    # accept scalars only
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                keys.append(k)
                vals.append(val)

            if keys:
                plt.figure(figsize=(max(4, len(keys) * 0.5), 3))
                plt.bar(keys, vals, color="C0")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(out_dir / "metrics.png", dpi=150)
                plt.close()
    else:
        # If metrics isn't dict-like, skip plotting
        pass


# Evaluate the model's performance on the validation set
validation_results = model.val()

save_and_plot_metrics(validation_results)


# Perform object detection on an image
results = model("basic-test.jpeg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model
