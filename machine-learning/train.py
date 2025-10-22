import os
from ultralytics import YOLO

def main():
    # Path to your dataset
    data_yaml = os.path.join(os.path.dirname(__file__), 'data', 'data.yaml')

    # Create YOLOv12 model instance
    model = YOLO('yolov12n-cls.pt')

    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), 'runs', 'train')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML file not found: {data_yaml}")
    else:
        with open(data_yaml, 'r') as f:
            print(f"Found dataset YAML file at {data_yaml}:\n")
            print(f.read())

    try:
        # Train the model
        model.train(
            data=data_yaml,
            epochs=100,
            imgsz=640,
            batch=16,
            project=output_dir,
            name='yolov12n-custom',
            task='detect'
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
    # # Save the trained model
    # model.export(format='pt', imgsz=640)
    # model.save(os.path.join(output_dir, 'yolov12n-custom.pt')))
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
    