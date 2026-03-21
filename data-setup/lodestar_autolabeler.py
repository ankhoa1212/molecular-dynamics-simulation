"""
Use LodeSTAR for generating YOLO labels on TIFF files.
Recursively searches for .tif/.tiff files, extracts nth frames, and generates YOLO labels.
Output structure mimics RoboFlow: <filename>_dataset/{images, labels}/
"""

import argparse
import json
import os
import glob
import logging
import dataclasses

import numpy as np
import torch
import cv2
import tifffile
from PIL import Image
from tqdm import tqdm

import deeplay as dl

# Suppress pint logging
logging.getLogger("pint").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(
        description="LodeSTAR Autolabeler for TIFF files."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the saved LodeSTAR .pt weights file.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Root directory to search for .tif/.tiff files recursively.",
    )
    parser.add_argument(
        "--nth", type=int, default=10,
        help="Save every nth frame (default: 10).",
    )
    parser.add_argument(
        "--box-size", type=int, default=40,
        help="Fixed bounding box size in pixels.",
    )
    parser.add_argument(
        "--detect-batch-size", type=int, default=4,
        help="Batch size for detection to prevent OOM.",
    )

    parser.add_argument(
        "--png-frames", type=str, default=None,
        help="Directory containing PNG frames to label (alternative to --input for TIFFs).",
    )
    return parser.parse_args()

def load_lodestar_model(model_path):
    """Load LodeSTAR model from .pt and companion .json config."""
    config_path = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        n_transforms = config.get("n_transforms", 8)
        num_outputs = config.get("num_outputs", 3)
        print(f"Loaded config: n_transforms={n_transforms}, num_outputs={num_outputs}")
    else:
        print(f"Warning: no companion JSON found. Using defaults (8, 3).")
        n_transforms = 8
        num_outputs = 3

    lodestar = dl.LodeSTAR(
        n_transforms=n_transforms,
        num_outputs=num_outputs,
    ).build()
    lodestar.load_state_dict(torch.load(model_path, map_location="cpu"))
    lodestar.eval()
    return lodestar, num_outputs

def extract_and_labels(tif_path, model, args, num_outputs):
    """Process a single TIFF file: extract frames and generate labels."""
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    output_base = os.path.join(os.path.dirname(tif_path), f"{base_name}_dataset")
    img_dir = os.path.join(output_base, "images")
    lbl_dir = os.path.join(output_base, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    try:
        tiff_stack = tifffile.imread(tif_path)
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    frame_indices = range(0, len(tiff_stack), args.nth)
    print(f"Processing {tif_path}: {len(frame_indices)} frames...")

    for i in tqdm(frame_indices, desc=f"Frames from {base_name}"):
        lbl_path = os.path.join(lbl_dir, f"frame_{i:05d}.txt")
        if os.path.exists(lbl_path):
            # Skip already labeled frames
            continue

        frame_raw = tiff_stack[i]

        # Normalize for saving and inference
        if frame_raw.dtype != np.uint8:
            frame_norm = cv2.normalize(frame_raw, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        else:
            frame_norm = frame_raw

        # Save image (RoboFlow style: frame_00000.png)
        img_name = f"frame_{i:05d}.png"
        img_path = os.path.join(img_dir, img_name)
        if len(frame_norm.shape) == 2:
            cv2.imwrite(img_path, frame_norm)
        else:
            cv2.imwrite(img_path, cv2.cvtColor(frame_norm, cv2.COLOR_RGB2BGR))

        # Prepare for inference (normalized 0-1)
        frame_f = frame_norm.astype(np.float32)
        f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
        frame_in = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min

        # Skip frames with NaN or Inf values after normalization
        if np.isnan(frame_in).any() or np.isinf(frame_in).any():
            print(f"Warning: NaN or Inf in frame {i} of {tif_path}, skipping.")
            continue

        # inference needs (1, 1, H, W)
        if len(frame_in.shape) == 2:
            input_tensor = torch.from_numpy(frame_in).unsqueeze(0).unsqueeze(0).to(device)
        else:  # assuming grayscale even if 3D
            input_tensor = torch.from_numpy(frame_in[:,:,0]).unsqueeze(0).unsqueeze(0).to(device)

        # Check for NaN/Inf in input tensor before detection
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            print(f"Warning: NaN or Inf in input tensor for frame {i} of {tif_path}, skipping.")
            continue

        with torch.inference_mode():
            detections = model.detect(input_tensor, alpha=0.5, beta=0.5, cutoff=0.5, mode="ratio")

        # detections is a list of arrays (one per batch item). We have batch size 1.
        if isinstance(detections, list):
            frame_dets = detections[0]
        else:
            frame_dets = detections

        # Check for NaN/Inf in detection output before writing labels
        if np.isnan(frame_dets).any() or np.isinf(frame_dets).any():
            print(f"Warning: NaN or Inf in detections for frame {i} of {tif_path}, skipping label writing.")
            continue

        h, w = frame_norm.shape[:2]

        with open(lbl_path, "w") as f:
            for det in frame_dets:
                det_y, det_x = det[0], det[1]
                # Normalize coordinates 0-1
                x_c = det_x / w
                y_c = det_y / h
                w_n = args.box_size / w
                h_n = args.box_size / h
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model}")
    model, num_outputs = load_lodestar_model(args.model)
    

    if args.png_frames:
        # Process PNG frames in the specified directory
        png_dir = args.png_frames
        png_files = sorted(glob.glob(os.path.join(png_dir, '*.png')))
        if not png_files:
            print(f"No PNG files found in {png_dir}")
            return
        print(f"Found {len(png_files)} PNG files in {png_dir}.")
        process_png_frames(png_files, model, args, num_outputs, png_dir)
        print("Done.")
        return

    # Default: process TIFFs
    search_pattern = os.path.join(args.input, "**", "*.tif*")
    tif_files = glob.glob(search_pattern, recursive=True)
    if not tif_files:
        print(f"No TIFF files found in {args.input}")
        return
    print(f"Found {len(tif_files)} TIFF files.")
    for tif_path in tif_files:
        extract_and_labels(tif_path, model, args, num_outputs)
    print("Done.")

def process_png_frames(png_files, model, args, num_outputs, png_dir):
    """Process a list of PNG files: run detection and generate YOLO labels."""
    base_name = os.path.basename(os.path.normpath(png_dir))
    output_base = os.path.join(os.path.dirname(png_dir), f"{base_name}_dataset")
    img_dir = os.path.join(output_base, "images")
    lbl_dir = os.path.join(output_base, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for idx, png_path in enumerate(tqdm(png_files, desc=f"Frames from {base_name}")):
        img_name = f"frame_{idx:05d}.png"
        img_out_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, f"frame_{idx:05d}.txt")
        if os.path.exists(lbl_path):
            continue

        # Read and normalize image
        frame_raw = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if frame_raw is None:
            print(f"Warning: Could not read {png_path}, skipping.")
            continue
        if frame_raw.dtype != np.uint8:
            frame_norm = cv2.normalize(frame_raw, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        else:
            frame_norm = frame_raw

        # Save normalized image to output dir (optional, for consistency)
        if len(frame_norm.shape) == 2:
            cv2.imwrite(img_out_path, frame_norm)
        else:
            cv2.imwrite(img_out_path, cv2.cvtColor(frame_norm, cv2.COLOR_RGB2BGR))

        # Prepare for inference (normalized 0-1)
        frame_f = frame_norm.astype(np.float32)
        f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
        frame_in = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min

        if np.isnan(frame_in).any() or np.isinf(frame_in).any():
            print(f"Warning: NaN or Inf in {png_path}, skipping.")
            continue

        if len(frame_in.shape) == 2:
            input_tensor = torch.from_numpy(frame_in).unsqueeze(0).unsqueeze(0).to(device)
        else:
            input_tensor = torch.from_numpy(frame_in[:,:,0]).unsqueeze(0).unsqueeze(0).to(device)

        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            print(f"Warning: NaN or Inf in input tensor for {png_path}, skipping.")
            continue

        with torch.inference_mode():
            detections = model.detect(input_tensor, alpha=0.5, beta=0.5, cutoff=0.5, mode="ratio")

        if isinstance(detections, list):
            frame_dets = detections[0]
        else:
            frame_dets = detections

        if np.isnan(frame_dets).any() or np.isinf(frame_dets).any():
            print(f"Warning: NaN or Inf in detections for {png_path}, skipping label writing.")
            continue

        h, w = frame_norm.shape[:2]
        with open(lbl_path, "w") as f:
            for det in frame_dets:
                det_y, det_x = det[0], det[1]
                x_c = det_x / w
                y_c = det_y / h
                w_n = args.box_size / w
                h_n = args.box_size / h
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

if __name__ == "__main__":
    main()
