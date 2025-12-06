"""
This module provides utilities for labeling and visualizing particle detection
in image sequences, particularly for molecular dynamics simulations. It includes
functions to generate YOLO labels using a pre-trained model, create pseudo-labels
via particle tracking, and overlay bounding boxes on images for visualization.
"""

import os

import numpy as np
import trackpy as tp
from PIL import Image, ImageDraw
from ultralytics import YOLO


def _generate_yolo_labels_for_image(model, image_path, label_file):
    """Generate YOLO labels for a single image using the provided model."""
    with open(label_file, "w", encoding="utf-8") as file:
        for result in model.predict(image_path, save=True, verbose=True):
            for box in result.boxes:
                x_center, y_center, w, h = box.xywhn[0].tolist()
                file.write(
                    f"{int(box.cls[0])} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                )


def generate_yolo_labels(input_path, output_label_folder, model_path="yolov8m.pt"):
    """Generate YOLO-format labels for images using a pre-trained YOLO model."""
    model = YOLO(model_path)

    if os.path.isfile(input_path) and input_path.lower().endswith(".png"):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".png")
        )
        if not files:
            raise ValueError(f"No PNG files found in folder: {input_path}")
    else:
        raise ValueError(
            "input_path must be a .png file or a folder containing .png files"
        )

    os.makedirs(output_label_folder, exist_ok=True)

    for i, fp in enumerate(files):
        label_file = os.path.join(output_label_folder, f"frame_{i:05d}.txt")
        _generate_yolo_labels_for_image(model, fp, label_file)


def _generate_pseudo_labels_for_image(
    group, label_file, img_width, img_height, box_size
):
    """Generate YOLO-format pseudo-labels for a single image based on trackpy output."""
    with open(label_file, "w", encoding="utf-8") as file:
        for _, row in group.iterrows():
            x_center = row["x"] / img_width
            y_center = row["y"] / img_height
            w = box_size / img_width
            h = box_size / img_height
            file.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def generate_pseudo_labels(
    input_path, output_label_folder, diameter=41, minmass=200, box_size=40
):
    """Generate pseudo-labels for images using trackpy."""
    if os.path.isfile(input_path) and input_path.lower().endswith(".png"):
        files = [input_path]
        stack = np.array([np.array(Image.open(input_path).convert("L"))])
    elif os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".png")
        )
        if not files:
            raise ValueError(f"No PNG files found in folder: {input_path}")
        stack = np.array([np.array(Image.open(fp).convert("L")) for fp in files])
    else:
        raise ValueError(
            "input_path must be a .png file or a folder containing .png files"
        )

    f = tp.batch(stack, diameter=diameter, minmass=minmass)

    os.makedirs(output_label_folder, exist_ok=True)

    img_height, img_width = stack.shape[1], stack.shape[2]

    for frame_id, group in f.groupby("frame"):
        label_file = os.path.join(output_label_folder, f"frame_{int(frame_id):05d}.txt")
        _generate_pseudo_labels_for_image(
            group, label_file, img_width, img_height, box_size
        )


def _overlay_labels(label_file, img):
    """Overlay bounding boxes from YOLO-format label file onto the given image."""
    with open(label_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    draw = ImageDraw.Draw(img)
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            _, x_c, y_c, ww, hh = map(float, parts)
            iw, ih = img.size
            draw.rectangle(
                [
                    (x_c - ww / 2) * iw,
                    (y_c - hh / 2) * ih,
                    (x_c + ww / 2) * iw,
                    (y_c + hh / 2) * ih,
                ],
                outline="red",
                width=2,
            )


def _save_label_overlay(i, image_path, label_folder, output_path):
    """Load an image, overlay labels, and save the result."""
    img = Image.open(image_path).convert("RGB")
    label_file = os.path.join(label_folder, f"frame_{i:05d}.txt")
    if os.path.exists(label_file):
        _overlay_labels(label_file, img)
    overlay_fp = os.path.join(output_path, f"frame_{i:05d}_overlay.png")
    img.save(overlay_fp)


def overlay_labels_on_images(input_path, output_label_folder, output_overlay_folder):
    """Overlay bounding boxes from YOLO-format labels onto images and save as new PNGs."""
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".png")
        )
    else:
        files = [input_path]

    os.makedirs(output_overlay_folder, exist_ok=True)

    for i, fp in enumerate(files):
        _save_label_overlay(i, fp, output_label_folder, output_overlay_folder)


INPUT_PATH = (
    "/home/ankhoa1212/git/molecular-dynamics-simulation/data-setup/"
    "Au Cit+1% of 2um PS+NaCl 20% Light Intensity Test Video 300 ms Trial 17_1_frames/"
    "Au Cit+1% of 2um PS+NaCl 20% Light Intensity Test Video 300 ms Trial 17_1_MMStack_Default.ome"
    "_frames"
)

# create output directory in current working directory based on input_path name
base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
output_dir = os.path.join(os.getcwd(), f"{base_name}_frames")

# custom_yolo_model = 'yolov8n.pt'  # replace with path to custom-trained model if available
# generate_yolo_labels(INPUT_PATH, output_dir, model_path=custom_yolo_model)
generate_pseudo_labels(INPUT_PATH, output_dir)

# Overlay labels on images and save as new PNGs
output_overlay_dir = os.path.join(os.getcwd(), f"{base_name}_overlays")
overlay_labels_on_images(INPUT_PATH, output_dir, output_overlay_dir)
