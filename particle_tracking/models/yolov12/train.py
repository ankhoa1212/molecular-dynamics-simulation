import os
import sys
import yaml


def run(config_path="config.yaml"):
    # Load config if exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Path to your dataset (default to processed_data/data.yaml)
    output_dir = config.get("output_dir", "processed_data")
    data_yaml = config.get("data_yaml", os.path.join(output_dir, "data.yaml"))

    # Ensure submodule is in path
    submodule_path = os.path.join(os.path.dirname(__file__), "yolov12")
    if os.path.exists(submodule_path) and submodule_path not in sys.path:
        sys.path.append(submodule_path)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Error: Could not import YOLO. Ensure ultralytics is installed or submodule is present."
        )
        return

    # Create YOLOv12 model instance
    model_name = config.get("model", "yolov12n-cls.pt")
    model = YOLO(model_name)

    # Ensure the output directory exists
    train_output_dir = os.path.join(os.path.dirname(__file__), "runs", "train")
    os.makedirs(train_output_dir, exist_ok=True)

    if not os.path.exists(data_yaml):
        print(f"Error: Dataset YAML file not found: {data_yaml}")
        return

    try:
        # Train the model
        model.train(
            data=data_yaml,
            epochs=config.get("epochs", 100),
            imgsz=config.get("imgsz", 640),
            batch=config.get("batch", 16),
            project=train_output_dir,
            name=config.get("name", "yolov12n-custom"),
            task=config.get("task", "detect"),
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")


if __name__ == "__main__":
    run()
