import os
import sys
import yaml
import torch
import numpy as np


def run(config_path="config.yaml"):
    # Load config if exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Ensure submodule is in path
    submodule_path = os.path.join(os.path.dirname(__file__), "yolov12")
    if os.path.exists(submodule_path) and submodule_path not in sys.path:
        sys.path.append(submodule_path)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Could not import YOLO.")
        return

    weights = config.get("weights", "yolov12.pt")
    data_yaml = config.get("data_yaml", "processed_data/data.yaml")

    if not os.path.exists(weights):
        print(f"Error: Weights file not found: {weights}")
        return

    model = YOLO(weights)

    try:
        results = model.val(data=data_yaml)
        print(f"Evaluation complete. mAP50: {results.box.map50:.4f}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    run()
