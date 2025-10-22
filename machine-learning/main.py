import torch
from util import tifFrameToPng, scatterplot, process_frames
import pandas

def main():
    model = torch.hub.load("yolov5", "custom", path="yolov5/runs/train/exp35/weights/best.pt", source="local", force_reload=True)
    # model = torch.load(f="yolov5/runs/train/exp35/weights/best.pt", weights_only=True)
    # Preview model predictions
    # path = "60% Intensity PS 5um Video Trial 1.tif"
    path = "70% Intensity PS 5um Video Trial 1_1_MMStack_Default.ome-0.png"
    results = model(path)
    # Draw only boxes, hide labels
    results.show(labels=False)

    # Get the guesses
    df = results.pandas().xyxy[0] if hasattr(results, "pandas") else results.xyxy[0]

    output_path = path
    # Convert a frame from the TIFF file to PNG and save it
    tifFrameToPng(0, output_path)

    # Create a scatterplot of detected object centers
    if not df.empty:
        scatterplot(df, output_path)
    
    # Process all frames in the TIFF file (example usage)
    frames = process_frames(path)

if __name__ == "__main__":
    main()
    