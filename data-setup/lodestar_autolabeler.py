"""
Use LodeSTAR for generating YOLO labels on TIFF files.
Recursively searches for .tif/.tiff files, extracts nth frames, and generates YOLO labels.
Output structure mimics RoboFlow: <filename>_dataset/{images, labels}/
"""

import argparse
import os
import glob
import logging
from typing import NamedTuple

import numpy as np
import torch
import cv2  # pylint: disable=no-member
import tifffile
from tqdm import tqdm

# Reuse shared utilities from label_images
from label_images import _nms, _load_model, _SaveConfig, _write_frame

# Suppress pint logging
logging.getLogger("pint").setLevel(logging.ERROR)


class _FrameCtx(NamedTuple):
    """Bundles inference state passed through the frame-processing pipeline."""

    model: object
    args: object
    device: str


def parse_args():
    """Parse command-line arguments for the LodeSTAR autolabeler."""
    parser = argparse.ArgumentParser(description="LodeSTAR Autolabeler for TIFF files.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the saved LodeSTAR .pt weights file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Root directory to search for .tif/.tiff files recursively.",
    )
    parser.add_argument("--nth", type=int, default=5, help="Save every nth frame (default: 5).")
    parser.add_argument(
        "--box-size", type=int, default=40, help="Fixed bounding box size in pixels."
    )
    parser.add_argument(
        "--detect-batch-size", type=int, default=4, help="Batch size for detection to prevent OOM."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for LodeSTAR detect (default: 0.5).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="Cutoff parameter for LodeSTAR detect (default: 0.5).",
    )
    parser.add_argument(
        "--nms-distance",
        type=float,
        default=0.0,
        help="Minimum pixel distance between detections (NMS). 0 disables NMS.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Overlay detections on the input image and save as <n>_overlay.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write YOLO label files (and overlays). "
            "Defaults to <n>_dataset/labels/ next to the input. "
            "For TIFF mode with multiple files, each TIFF gets a sub-folder."
        ),
    )
    parser.add_argument(
        "--use-radius",
        action="store_true",
        help=(
            "Use the model's per-detection radius as box size instead of --box-size. "
            "Requires num_outputs >= 3 in the saved model."
        ),
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the raw radius output to convert it to pixels.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=0.0,
        help=(
            "Minimum box size in pixels when --use-radius is active. "
            "0 = use --box-size as the floor."
        ),
    )
    parser.add_argument(
        "--png-frames",
        type=str,
        default=None,
        help=("Directory containing PNG frames to label " "(alternative to --input for TIFFs)."),
    )
    return parser.parse_args()


def _detect_frame(frame_norm, ctx):
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
        input_tensor = torch.from_numpy(frame_in).unsqueeze(0).unsqueeze(0).to(ctx.device)
    else:
        input_tensor = torch.from_numpy(frame_in[:, :, 0]).unsqueeze(0).unsqueeze(0).to(ctx.device)

    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        return None

    with torch.inference_mode():
        detections = ctx.model.detect(
            input_tensor,
            alpha=ctx.args.alpha,
            beta=1.0 - ctx.args.alpha,
            cutoff=ctx.args.cutoff,
            mode="ratio",
        )

    frame_dets = detections[0] if isinstance(detections, list) else detections

    if np.isnan(frame_dets).any() or np.isinf(frame_dets).any():
        return None

    if ctx.args.nms_distance > 0:
        frame_dets = _nms(list(frame_dets), ctx.args.nms_distance)

    return frame_dets


def _make_cfg(ctx, output_dir, frame_shape):
    """Build a _SaveConfig from autolabeler args."""
    min_box_px = ctx.args.min_box_size if ctx.args.min_box_size > 0 else float(ctx.args.box_size)
    return _SaveConfig(
        output_dir=output_dir,
        frame_shape=frame_shape,
        use_radius=ctx.args.use_radius,
        radius_scale=ctx.args.radius_scale,
        min_box_px=min_box_px,
        box_size=ctx.args.box_size,
        do_plot=ctx.args.plot,
    )


def _normalize_frame(frame_raw):
    """Normalize a raw frame to uint8, returning the normalized array."""
    if frame_raw.dtype != np.uint8:
        return cv2.normalize(  # pylint: disable=no-member
            frame_raw, None, 0, 255, cv2.NORM_MINMAX  # pylint: disable=no-member
        ).astype("uint8")
    return frame_raw


def _save_image(img_path, frame_norm):
    """Write a normalized frame to disk in BGR format."""
    if len(frame_norm.shape) == 2:
        cv2.imwrite(img_path, frame_norm)  # pylint: disable=no-member
    else:
        bgr = cv2.cvtColor(frame_norm, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
        cv2.imwrite(img_path, bgr)  # pylint: disable=no-member


def _process_single_png(idx, png_path, img_dir, lbl_dir, ctx):
    """Detect objects in one PNG frame and write the YOLO label file.

    Skips the frame if the label already exists, the file is unreadable,
    or the detection output contains NaN/Inf values.
    """
    img_out_path = os.path.join(img_dir, f"frame_{idx:05d}.png")
    lbl_path = os.path.join(lbl_dir, f"frame_{idx:05d}.txt")
    if os.path.exists(lbl_path):
        return

    frame_raw = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member
    if frame_raw is None:
        print(f"Warning: Could not read {png_path}, skipping.")
        return

    frame_norm = _normalize_frame(frame_raw)
    _save_image(img_out_path, frame_norm)

    frame_dets = _detect_frame(frame_norm, ctx)
    if frame_dets is None:
        print(f"Warning: skipping {png_path} (NaN/Inf or empty).")
        return

    cfg = _make_cfg(ctx, lbl_dir, frame_norm.shape[:2])
    _write_frame(lbl_path, frame_norm, frame_dets, cfg)


def _process_single_tif_frame(frame_index, tiff_stack, img_dir, lbl_dir, ctx):
    """Normalize, save, and label one frame from a TIFF stack.

    Skips the frame if the label already exists or detection yields NaN/Inf.
    """
    lbl_path = os.path.join(lbl_dir, f"frame_{frame_index:05d}.txt")
    if os.path.exists(lbl_path):
        return
    frame_norm = _normalize_frame(tiff_stack[frame_index])
    _save_image(os.path.join(img_dir, f"frame_{frame_index:05d}.png"), frame_norm)
    frame_dets = _detect_frame(frame_norm, ctx)
    if frame_dets is None:
        print(f"Warning: skipping frame {frame_index} (NaN/Inf or empty).")
        return
    cfg = _make_cfg(ctx, lbl_dir, frame_norm.shape[:2])
    _write_frame(lbl_path, frame_norm, frame_dets, cfg)


def extract_and_labels(tif_path, model, args):
    """Process a single TIFF file: extract frames and generate YOLO labels."""
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    tif_dir = os.path.dirname(tif_path)
    # Inline dataset root to avoid an extra local variable
    img_dir = os.path.join(tif_dir, f"{base_name}_dataset", "images")
    # --output-dir: each TIFF gets its own sub-folder to avoid name collisions
    if args.output_dir:
        lbl_dir = os.path.join(args.output_dir, base_name)
    else:
        lbl_dir = os.path.join(tif_dir, f"{base_name}_dataset", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    try:
        tiff_stack = tifffile.imread(tif_path)
    except OSError as e:
        print(f"Error reading {tif_path}: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = _FrameCtx(model=model.to(device), args=args, device=device)

    # Inline range into the loop to avoid an extra local variable
    print(f"Processing {tif_path}: {len(range(0, len(tiff_stack), args.nth))} frames...")
    for frame_index in tqdm(range(0, len(tiff_stack), args.nth), desc=f"Frames from {base_name}"):
        _process_single_tif_frame(frame_index, tiff_stack, img_dir, lbl_dir, ctx)


def process_png_frames(png_files, model, args, png_dir):
    """Process a list of PNG files: run detection and generate YOLO labels."""
    base_name = os.path.basename(os.path.normpath(png_dir))
    # Inline dataset root to avoid an extra local variable
    img_dir = os.path.join(os.path.dirname(png_dir), f"{base_name}_dataset", "images")
    lbl_dir = (
        args.output_dir
        if args.output_dir
        else os.path.join(os.path.dirname(png_dir), f"{base_name}_dataset", "labels")
    )
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = _FrameCtx(model=model.to(device), args=args, device=device)

    # Per-frame logic is delegated to _process_single_png to keep locals under limit
    for idx, png_path in enumerate(tqdm(png_files, desc=f"Frames from {base_name}")):
        _process_single_png(idx, png_path, img_dir, lbl_dir, ctx)


def main():
    """Entry point: load model and dispatch to TIFF or PNG processing."""
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
