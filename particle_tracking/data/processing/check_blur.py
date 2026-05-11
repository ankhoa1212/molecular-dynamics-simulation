#!/usr/bin/env python3
"""
Blur detection utility using the Variance of Laplacian method.
Higher score = Sharper image. Lower score = Blurrier image.

Usage:
    # Basic quality check for a folder of images
    python check_blur.py --input data/frames/ --threshold 80

    # Exploratory analysis of a TIFF video (Plot focus drift + Identify peaks)
    python check_blur.py --input video.tif --plot --window-size 200 --top-in-window

    # Selective sampling: Check every 10th frame and save the sharpest ones as PNGs
    python check_blur.py --input video.tif --nth 10 --save-samples --top-in-window

    # Tracker training: Save 20-frame bursts around focal peaks
    python check_blur.py --input video.tif --save-sequences 20 --window-size 300
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_blur_score(image: np.ndarray, blur_size: int = 3, grid_size: int = 1) -> float:
    """
    Compute the focal quality of an image using the Variance of Laplacian method.
    If grid_size > 1, returns the average of the top 10% sharpest tiles (robust for sparse images).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalize exotic bit-depths to 8-bit for consistent scoring
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Denoise to avoid high-frequency sensor noise being counted as "sharpness"
    if blur_size > 0:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if grid_size <= 1:
        return laplacian.var()

    # Tiled mode: find the SHARPEST parts of the image
    h, w = laplacian.shape
    th, tw = h // grid_size, w // grid_size
    tile_vars = []

    for i in range(grid_size):
        for j in range(grid_size):
            tile = laplacian[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            tile_vars.append(tile.var())

    # Sort descending and take top 10% (at least 1)
    tile_vars.sort(reverse=True)
    num_to_avg = max(1, len(tile_vars) // 10)
    return float(np.mean(tile_vars[:num_to_avg]))


def process_file(
    file_path: Path,
    threshold: Optional[float],
    nth: int = 1,
    blur_size: int = 3,
    grid_size: int = 1,
) -> tuple[list[int], list[float]]:
    """Process a single image or TIFF stack. Returns (indices, scores)."""
    ext = file_path.suffix.lower()
    indices = []
    scores = []

    if ext in [".tif", ".tiff"]:
        try:
            stack = tifffile.imread(file_path)
            if len(stack.shape) == 2:  # Single frame TIFF
                stack = [stack]

            print(
                f"Checking TIFF stack: {file_path.name} ({len(stack)} frames, checking every {nth}th)"
            )
            blurry_frames = []
            for i in tqdm(range(0, len(stack), nth), desc="Frames"):
                frame = stack[i]
                score = calculate_blur_score(frame, blur_size, grid_size)
                indices.append(i)
                scores.append(score)
                if threshold is not None and score < threshold:
                    blurry_frames.append((i, score))

            if threshold is not None:
                if blurry_frames:
                    print(f"\nFound {len(blurry_frames)} blurry frames (Score < {threshold}):")
                    for i, score in blurry_frames[:10]:
                        print(f"  Frame {i:04d}: {score:.2f}")
                    if len(blurry_frames) > 10:
                        print(f"  ... and {len(blurry_frames) - 10} more.")
                else:
                    print(f"No frames were below the threshold ({threshold}).")

        except Exception as e:
            print(f"Error reading TIFF {file_path}: {e}")
            return [], []
    else:
        # Standard image
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"Warning: Could not read {file_path}")
            return []

        score = calculate_blur_score(img, blur_size, grid_size)
        indices.append(0)
        scores.append(score)
        if threshold is not None:
            status = "BLURRY" if score < threshold else "SHARP"
            print(f"{file_path.name}: {score:.2f} ({status})")
        else:
            print(f"{file_path.name}: {score:.2f}")

    return indices, scores


def plot_scores(
    indices: list[int],
    scores: list[float],
    threshold: Optional[float],
    title: str,
    is_stack: bool,
    output_path: str,
    moving_median: Optional[list[float]] = None,
    peak_indices: Optional[list[int]] = None,
):
    """Generate and save a plot of blur scores."""
    plt.figure(figsize=(12, 6))

    if is_stack:
        # Time-series plot for stacks
        plt.plot(indices, scores, label="Raw Sharpness", color="#bdc3c7", linewidth=1, alpha=0.6)

        if moving_median:
            plt.plot(
                indices, moving_median, label="Moving Median (Trend)", color="#2ecc71", linewidth=2
            )

        if peak_indices:
            peak_vals = [scores[i] for i in peak_indices]
            peak_frames = [indices[i] for i in peak_indices]
            plt.scatter(
                peak_frames,
                peak_vals,
                color="#f1c40f",
                s=30,
                label="Local Peaks (Best Focus)",
                zorder=5,
            )

        if threshold is not None:
            plt.axhline(
                y=threshold, color="#e74c3c", linestyle="--", label=f"Threshold ({threshold})"
            )

        plt.xlabel("Frame Index")
        plt.ylabel("Sharpness Score (Var of Laplacian)")
        plt.title(f"Focus Stability Analysis: {title}")
    else:
        # Histogram for directories
        plt.hist(scores, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
        if threshold is not None:
            plt.axvline(
                x=threshold, color="#e74c3c", linestyle="--", label=f"Threshold ({threshold})"
            )
        plt.xlabel("Sharpness Score (Var of Laplacian)")
        plt.ylabel("Frequency")
        plt.title(f"Dataset Quality Distribution: {title}")

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Analysis plot saved to: {output_path}")
    plt.close()


def calculate_moving_median(scores: list[float], window_size: int) -> list[float]:
    """Calculate the moving median of a list of scores."""
    if window_size <= 1:
        return scores

    medians = []
    half = window_size // 2
    for i in range(len(scores)):
        start = max(0, i - half)
        end = min(len(scores), i + half + 1)
        medians.append(float(np.median(scores[start:end])))
    return medians


def find_local_peaks(scores: list[float], window_size: int) -> list[int]:
    """Identify indices of local maximums within a sliding window."""
    if window_size <= 1:
        return list(range(len(scores)))

    peaks = []
    half = window_size // 2
    for i in range(len(scores)):
        start = max(0, i - half)
        end = min(len(scores), i + half + 1)
        if scores[i] == max(scores[start:end]):
            # Avoid adjacent duplicates
            if not peaks or peaks[-1] != i - 1:
                peaks.append(i)
    return peaks


def save_sample_frames(
    input_path: Path,
    indices: list[int],
    scores: list[float],
    threshold: Optional[float],
    peak_indices: Optional[list[int]] = None,
    seq_length: int = 0,
):
    """Save the sharpest, blurriest, and transition frames/sequences for inspection."""
    if not indices or not scores:
        return

    # 1. Identify interesting frames
    idx_max = np.argmax(scores)
    idx_min = np.argmin(scores)

    samples = {
        "global_sharpest": (indices[idx_max], scores[idx_max]),
        "global_blurriest": (indices[idx_min], scores[idx_min]),
    }

    # If we have local peaks, save the first and middle one as well
    if peak_indices and len(peak_indices) > 1:
        mid_peak = peak_indices[len(peak_indices) // 2]
        samples["local_peak_mid"] = (indices[mid_peak], scores[mid_peak])

    # Find sudden drops (rate of change)
    if len(scores) > 5:
        diffs = np.diff(scores)
        drop_thresh = np.mean(scores) * 0.2
        idx_drop = np.argmin(diffs)
        if diffs[idx_drop] < -drop_thresh:
            samples["sudden_drop_after"] = (indices[idx_drop + 1], scores[idx_drop + 1])

    # 2. Extract and save
    out_dir = Path(f"blur_samples_{input_path.stem}")
    out_dir.mkdir(exist_ok=True)

    try:
        if input_path.is_file() and input_path.suffix.lower() in [".tif", ".tiff"]:
            stack = tifffile.imread(input_path)

            # Normal single-frame samples
            for name, (f_idx, score) in samples.items():
                frame = stack[f_idx]
                if frame.dtype != np.uint8:
                    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                out_path = out_dir / f"{name}_frame{f_idx:04d}_score{score:.1f}.png"
                cv2.imwrite(str(out_path), frame)

            # Sequence bursts (for tracker training)
            if seq_length > 0 and peak_indices:
                print(f"Saving {len(peak_indices)} sequences of length {seq_length}...")
                for p_idx in peak_indices:
                    seq_dir = out_dir / f"sequence_peak{indices[p_idx]:04d}"
                    seq_dir.mkdir(exist_ok=True)

                    start_f = max(0, indices[p_idx] - seq_length // 2)
                    end_f = min(len(stack), start_f + seq_length)

                    for f_idx in range(start_f, end_f):
                        frame = stack[f_idx]
                        if frame.dtype != np.uint8:
                            frame = cv2.normalize(
                                frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                            )
                        out_path = seq_dir / f"frame_{f_idx:04d}.png"
                        cv2.imwrite(str(out_path), frame)
        else:
            print("Note: --save-samples/sequences currently only implemented for TIFF stacks.")

        print(f"Extraction complete. Results in: {out_dir}")
    except Exception as e:
        print(f"Failed to extract samples: {e}")


def main():
    parser = argparse.ArgumentParser(description="Check for blurriness in images or TIFF stacks.")
    parser.add_argument("--input", "-i", required=True, help="Path to image, TIFF, or directory.")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Threshold for blurriness. If not provided, classification is skipped.",
    )
    parser.add_argument(
        "--nth",
        "-n",
        type=int,
        default=1,
        help="Check every nth frame of TIFF stacks (default: 1).",
    )
    parser.add_argument(
        "--blur-size",
        "-b",
        type=int,
        default=3,
        help="Gaussian blur size to remove noise before check (default: 3).",
    )
    parser.add_argument(
        "--grid",
        "-g",
        type=int,
        default=4,
        help="Divide image into NxN grid and take MAX sharpness (default: 4). Helps for sparse images.",
    )
    parser.add_argument(
        "--window-size",
        "-w",
        type=int,
        default=50,
        help="Window size for moving median and peak detection (default: 50).",
    )
    parser.add_argument(
        "--top-in-window",
        action="store_true",
        help="Only output/highlight the best frames within each window.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate a visualization plot.")
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save sharpest/blurriest/drop frames as PNGs for inspection.",
    )
    parser.add_argument(
        "--save-sequences",
        "-s",
        type=int,
        default=0,
        help="Save N-frame sequences around each local peak (for tracker training).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Path {args.input} does not exist.")
        return

    all_indices = []
    all_scores = []
    is_stack = False

    if input_path.is_file():
        all_indices, all_scores = process_file(
            input_path, args.threshold, args.nth, args.blur_size, args.grid
        )
        is_stack = input_path.suffix.lower() in [".tif", ".tiff"]
    elif input_path.is_dir():
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in image_exts])
        print(f"Checking {len(files)} files in {input_path.name}...")
        for f in tqdm(files, desc="Files"):
            idx, sc = process_file(f, args.threshold, args.nth, args.blur_size, args.grid)
            all_indices.extend(idx)
            all_scores.extend(sc)
        is_stack = False

    if args.plot and all_scores:
        out_name = f"blur_analysis_{input_path.stem}.png"

        # Calculate moving median if we have enough points
        moving_median = None
        if len(all_scores) > args.window_size:
            moving_median = calculate_moving_median(all_scores, args.window_size)

        peak_indices = None
        if args.top_in_window and len(all_scores) > args.window_size:
            peak_indices = find_local_peaks(all_scores, args.window_size)
            print(f"Identified {len(peak_indices)} local peaks across the recording.")

        plot_scores(
            all_indices,
            all_scores,
            args.threshold,
            input_path.name,
            is_stack,
            out_name,
            moving_median,
            peak_indices,
        )

    if args.save_samples and is_stack:
        save_sample_frames(
            input_path,
            all_indices,
            all_scores,
            args.threshold,
            peak_indices,
            seq_length=args.save_sequences,
        )


if __name__ == "__main__":
    main()
