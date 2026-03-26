"""
Use LodeSTAR for generating YOLO labels on TIFF files.
Recursively searches for .tif/.tiff files, extracts nth frames, and generates YOLO labels.
Output structure mimics RoboFlow: <filename>_dataset/{images, labels}/
"""

import argparse
import os
import glob
import logging

import numpy as np
import torch
import cv2
import tifffile
from tqdm import tqdm

# Reuse shared utilities from label_images
from label_images import (
    _nms,
    _load_model,
    _SaveConfig,
    _write_frame,
    _print_radius_stats,
)

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
        "--input", type=str, default=None,
        help="Root directory to search for .tif/.tiff files recursively.",
    )
    parser.add_argument(
        "--nth", type=int, default=5,
        help="Save every nth frame (default: 5).",
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
        "--alpha", type=float, default=0.5,
        help="Alpha parameter for LodeSTAR detect (default: 0.5).",
    )
    parser.add_argument(
        "--cutoff", type=float, default=0.5,
        help="Cutoff parameter for LodeSTAR detect (default: 0.5).",
    )
    parser.add_argument(
        "--nms-distance", type=float, default=0.0,
        help="Minimum pixel distance between detections (NMS). 0 disables NMS.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Overlay detections on the input image and save as <name>_overlay.png.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Directory to write YOLO label files (and overlays). "
            "Defaults to <name>_dataset/labels/ next to the input. "
            "For TIFF mode with multiple files, each TIFF gets a sub-folder."
        ),
    )
    parser.add_argument(
        "--use-radius", action="store_true",
        help=(
            "Use the model's per-detection radius as box size instead of --box-size. "
            "Requires num_outputs >= 3 in the saved model."
        ),
    )
    parser.add_argument(
        "--radius-scale", type=float, default=1.0,
        help="Multiplier applied to the raw radius output to convert it to pixels.",
    )
    parser.add_argument(
        "--min-box-size", type=float, default=0.0,
        help="Minimum box size in pixels when --use-radius is active. 0 = use --box-size as the floor.",
    )
    parser.add_argument(
        "--png-frames", type=str, default=None,
        help="Directory containing PNG frames to label (alternative to --input for TIFFs).",
    )
    return parser.parse_args()


def _detect_frame(frame_norm, model, args, device):
    """Normalise a uint8 frame to float, build a tensor, run detect, return detections array.

    Returns None if the frame or tensor contains NaN/Inf.
    """
    frame_f = frame_norm.astype(np.float32)
    f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
    frame_in = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min

    if np.isnan(frame_in).any() or np.isinf(frame_in).any():
        return None

    # inference needs (1, 1, H, W)
    if len(frame_in.shape) == 2:
        input_tensor = torch.from_numpy(frame_in).unsqueeze(0).unsqueeze(0).to(device)
    else:
        input_tensor = torch.from_numpy(frame_in[:, :, 0]).unsqueeze(0).unsqueeze(0).to(device)

    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        return None

    with torch.inference_mode():
        detections = model.detect(
            input_tensor,
            alpha=args.alpha,
            beta=1.0 - args.alpha,
            cutoff=args.cutoff,
            mode="ratio",
        )

    frame_dets = detections[0] if isinstance(detections, list) else detections

    if np.isnan(frame_dets).any() or np.isinf(frame_dets).any():
        return None

    if args.nms_distance > 0:
        frame_dets = _nms(list(frame_dets), args.nms_distance)

    return frame_dets


def _make_cfg(args, output_dir, frame_shape):
    """Build a _SaveConfig from autolabeler args."""
    min_box_px = args.min_box_size if args.min_box_size > 0 else float(args.box_size)
    return _SaveConfig(
        output_dir=output_dir,
        frame_shape=frame_shape,
        use_radius=args.use_radius,
        radius_scale=args.radius_scale,
        min_box_px=min_box_px,
        box_size=args.box_size,
        do_plot=args.plot,
    )


def extract_and_labels(tif_path, model, args):
    """Process a single TIFF file: extract frames and generate YOLO labels."""
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    output_base = os.path.join(os.path.dirname(tif_path), f"{base_name}_dataset")
    img_dir = os.path.join(output_base, "images")
    # --output-dir: each TIFF gets its own sub-folder to avoid name collisions
    if args.output_dir:
        lbl_dir = os.path.join(args.output_dir, base_name)
    else:
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
            continue

        frame_raw = tiff_stack[i]

        # Normalize for saving and inference
        if frame_raw.dtype != np.uint8:
            frame_norm = cv2.normalize(frame_raw, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        else:
            frame_norm = frame_raw

        # Save image (RoboFlow style: frame_00000.png)
        img_path = os.path.join(img_dir, f"frame_{i:05d}.png")
        if len(frame_norm.shape) == 2:
            cv2.imwrite(img_path, frame_norm)
        else:
            cv2.imwrite(img_path, cv2.cvtColor(frame_norm, cv2.COLOR_RGB2BGR))

        frame_dets = _detect_frame(frame_norm, model, args, device)
        if frame_dets is None:
            print(f"Warning: skipping frame {i} of {tif_path} (NaN/Inf or empty).")
            continue

        cfg = _make_cfg(args, lbl_dir, frame_norm.shape[:2])
        _write_frame(lbl_path, frame_norm, frame_dets, cfg)


def process_png_frames(png_files, model, args, png_dir):
    """Process a list of PNG files: run detection and generate YOLO labels."""
    base_name = os.path.basename(os.path.normpath(png_dir))
    output_base = os.path.join(os.path.dirname(png_dir), f"{base_name}_dataset")
    img_dir = os.path.join(output_base, "images")
    lbl_dir = args.output_dir if args.output_dir else os.path.join(output_base, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for idx, png_path in enumerate(tqdm(png_files, desc=f"Frames from {base_name}")):
        img_out_path = os.path.join(img_dir, f"frame_{idx:05d}.png")
        lbl_path = os.path.join(lbl_dir, f"frame_{idx:05d}.txt")
        if os.path.exists(lbl_path):
            continue

        frame_raw = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if frame_raw is None:
            print(f"Warning: Could not read {png_path}, skipping.")
            continue
        if frame_raw.dtype != np.uint8:
            frame_norm = cv2.normalize(frame_raw, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        else:
            frame_norm = frame_raw

        # Save normalized image to output dir
        if len(frame_norm.shape) == 2:
            cv2.imwrite(img_out_path, frame_norm)
        else:
            cv2.imwrite(img_out_path, cv2.cvtColor(frame_norm, cv2.COLOR_RGB2BGR))

        frame_dets = _detect_frame(frame_norm, model, args, device)
        if frame_dets is None:
            print(f"Warning: skipping {png_path} (NaN/Inf or empty).")
            continue

        cfg = _make_cfg(args, lbl_dir, frame_norm.shape[:2])
        _write_frame(lbl_path, frame_norm, frame_dets, cfg)


def main():
    args = parse_args()

    if not args.input and not args.png_frames:
        raise ValueError("Either --input or --png-frames must be provided.")

    # Bridge: _load_model() from label_images expects args.model_path
    args.model_path = args.model
    args.detect_mode = "ratio"
    # _load_model sets args.num_outputs from the companion JSON
    args.num_outputs = 3  # default; overwritten by _load_model if JSON exists

    print(f"Loading model: {args.model}")
    lodestar = _load_model(args)

    if args.png_frames:
        png_dir = args.png_frames
        png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))
        if not png_files:
            print(f"No PNG files found in {png_dir}")
            return
        print(f"Found {len(png_files)} PNG files in {png_dir}.")
        process_png_frames(png_files, lodestar, args, png_dir)
        print("Done.")
        return

    # Default: process TIFFs recursively
    search_pattern = os.path.join(args.input, "**", "*.tif*")
    tif_files = glob.glob(search_pattern, recursive=True)
    if not tif_files:
        print(f"No TIFF files found in {args.input}")
        return
    print(f"Found {len(tif_files)} TIFF files.")
    for tif_path in tif_files:
        extract_and_labels(tif_path, lodestar, args)
    print("Done.")


if __name__ == "__main__":
    main()
