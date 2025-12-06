import trackpy as tp
import pandas as pd
import os
from PIL import Image, ImageDraw
import numpy as np

def generate_pseudo_labels(input_path, output_label_folder, diameter=41, minmass=200, box_size=40):
    # Load single PNG or all PNGs in a folder (sorted)
    frames = []
    if os.path.isfile(input_path) and input_path.lower().endswith('.png'):
        img = Image.open(input_path).convert('L')
        frames.append(np.array(img))
    elif os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith('.png')
        )
        if not files:
            raise ValueError(f"No PNG files found in folder: {input_path}")
        for fp in files:
            img = Image.open(fp).convert('L')
            frames.append(np.array(img))
    else:
        raise ValueError("input_path must be a .png file or a folder containing .png files")

    # Stack into shape (n_frames, height, width)
    stack = np.stack(frames, axis=0)

    # Detect particles across the stack
    f = tp.batch(stack, diameter=diameter, minmass=minmass)

    os.makedirs(output_label_folder, exist_ok=True)

    img_height, img_width = stack.shape[1], stack.shape[2]

    # Group by frame to save YOLO-format txt files
    for frame_id, group in f.groupby('frame'):
        label_file = os.path.join(output_label_folder, f"frame_{int(frame_id):05d}.txt")
        with open(label_file, 'w') as file:
            for _, row in group.iterrows():
                x_center = row['x'] / img_width
                y_center = row['y'] / img_height
                w = box_size / img_width
                h = box_size / img_height
                file.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def overlay_labels_on_images(input_path, output_label_folder, output_overlay_folder):
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith('.png')
        )
    else:
        files = [input_path]
    
    os.makedirs(output_overlay_folder, exist_ok=True)
    
    for i, fp in enumerate(files):
        img = Image.open(fp).convert('RGB')
        label_file = os.path.join(output_label_folder, f"frame_{i:05d}.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            draw = ImageDraw.Draw(img)
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x_center, y_center, w, h = map(float, parts)
                    img_width, img_height = img.size
                    x1 = (x_center - w/2) * img_width
                    y1 = (y_center - h/2) * img_height
                    x2 = (x_center + w/2) * img_width
                    y2 = (y_center + h/2) * img_height
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        overlay_fp = os.path.join(output_overlay_folder, f"frame_{i:05d}_overlay.png")
        img.save(overlay_fp)

input_path = "/home/ankhoa1212/git/molecular-dynamics-simulation/data-setup/Au Cit+1% of 2um PS+NaCl 20% Light Intensity Test Video 300 ms Trial 17_1_frames/Au Cit+1% of 2um PS+NaCl 20% Light Intensity Test Video 300 ms Trial 17_1_MMStack_Default.ome_frames"

# create output directory in current working directory based on input_path name
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = os.path.join(os.getcwd(), f"{base_name}_frames")

# generate_pseudo_labels("path/to/image.png", "dataset/labels/train")
generate_pseudo_labels(input_path, output_dir)

# Overlay labels on images and save as new PNGs
output_overlay_dir = os.path.join(os.getcwd(), f"{base_name}_overlays")
overlay_labels_on_images(input_path, output_dir, output_overlay_dir)