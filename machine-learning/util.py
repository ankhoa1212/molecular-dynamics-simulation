import pims
import os

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def tifFrameToPng(i: int, path: str):
    """
    Converts a specific frame from a TIFF stack to a PNG image file.

    Args:
        i (int): The index of the frame to extract from the TIFF stack.
        path (str): The file path to the TIFF stack.

    Returns:
        str: The file path to the generated PNG image. If the PNG already exists, returns the existing file path.

    Notes:
        - The output PNG file will be named using the original TIFF path with the frame index appended before the '.png' extension.
        - The function suppresses axes and tightens the layout for the saved image.
        - The image is saved in grayscale colormap.
    """
    v = pims.TiffStack(path)
    path = f"{path[:-4]}-{i}.png"
    if not os.path.isfile(path):
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(v[i], cmap="gray")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
    return path

def scatterplot(results, path):
    """
    Generates scatterplot data from object detection results and loads the corresponding image.

    Args:
        results: An object containing detection results, expected to have a `pandas().xyxy[0]` method/attribute
            that returns a DataFrame with columns 'xmin', 'xmax', 'ymin', 'ymax', and 'confidence'.
        path (str): The file path to the image associated with the detection results.

    Returns:
        tuple: A tuple containing:
            - x (list of float): The x-coordinates (center points) of detected bounding boxes.
            - y (list of float): The y-coordinates (center points) of detected bounding boxes.
            - conf (list of float): The confidence scores (as percentages) for each detection.
            - img (ndarray): The image loaded from the specified path.
    """
    x = []
    y = []
    conf = []
    stf = results.pandas().xyxy[0] if hasattr(results, "pandas") else results.xyxy[0]
    for i in range(len(stf["xmin"])):
        x.append((stf["xmin"][i] + stf["xmax"][i]) / 2)
        y.append((stf["ymin"][i] + stf["ymax"][i]) / 2)
        conf.append(stf["confidence"][i] * 100)
    # Generate the scatterplot
    img = mpimg.imread(path)
    # You can add plotting code here if needed
    return x, y, conf, img

def process_frames(fileName, model):
    """
    Processes frames from a multi-frame image file, applies a model to each frame, and collects detection counts.

    Args:
        fileName (str): Path to the multi-frame image file (e.g., TIFF).
        model (callable): A model function that takes an image path as input and returns detection results with a pandas DataFrame accessible via results.pandas().xyxy[0].

    Returns:
        tuple: Two lists:
            - x (list of int): Frame indices.
            - y (list of int): Number of detections per frame.
    """
    frames = Image.open(fileName)
    y = []
    x = []
    for i in range(frames.n_frames):
        path = tifFrameToPng(i, fileName)
        results = model(path)
        os.remove(path)
        y.append(len(results.pandas().xyxy[0]["xmin"]) if hasattr(results, "pandas") else len(results.xyxy[0]["xmin"]))
        x.append(i)
    return x, y
