import os
from ultralytics import YOLO

# Path to your dataset
data_yaml = os.path.join(os.path.dirname(__file__), 'data', 'data.yaml')

# Create YOLOv12 model (replace 'yolov12.pt' with your pretrained weights if available)
model = YOLO('yolov12.pt')  # Make sure you have the correct weights file

# Train the model
model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    project='runs/train',
    name='yolov12-custom'
)