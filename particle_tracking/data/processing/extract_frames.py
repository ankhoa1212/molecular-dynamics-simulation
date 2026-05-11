import argparse
import os
import concurrent.futures
import multiprocessing

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


def _process_frame(args):
    """Helper function for parallel processing of a single frame."""
    frame_data, output_path, image_format = args

    # Normalization
    if frame_data.dtype != np.uint8:
        frame_data = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Handle Color Channels
    if len(frame_data.shape) == 2:
        frame_data = cv2.merge([frame_data, frame_data, frame_data])

    # Save Frame
    cv2.imwrite(output_path, frame_data)
    return True


def convert_jpg_to_frames(input_path, output_folder, image_format="png", num_workers=None):
    """Convert JPG images to different image format in parallel."""
    if os.path.isdir(input_path):
        jpg_files = sorted(
            [f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".jpeg"))]
        )
        if not jpg_files:
            print(f"No .jpg files found in directory: {input_path}")
            return

        tasks = []
        for fname in jpg_files:
            src = os.path.join(input_path, fname)
            dst = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}.{image_format}")
            img = cv2.imread(src)
            if img is not None:
                tasks.append((img, dst, image_format))

        os.makedirs(output_folder, exist_ok=True)

        if num_workers == 1:
            for t in tqdm(tasks, desc="Converting JPGs"):
                _process_frame(t)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(
                    tqdm(
                        executor.map(_process_frame, tasks),
                        total=len(tasks),
                        desc="Converting JPGs",
                    )
                )
        return

    os.makedirs(output_folder, exist_ok=True)
    try:
        img = cv2.imread(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        cv2.imwrite(os.path.join(output_folder, f"{base}.{image_format}"), img)
    except cv2.error as e:
        print(f"Error converting JPG '{input_path}': {e}")


def convert_tif_to_frames(input_path, output_folder, image_format="png", nth=10, num_workers=None):
    """Convert multi-page TIFF files to individual image frames in parallel."""

    if os.path.isdir(input_path):
        tif_files = sorted(
            [f for f in os.listdir(input_path) if f.lower().endswith((".tif", ".tiff"))]
        )
        if not tif_files:
            print(f"No .tif/.tiff files found in directory: {input_path}")
            return
        for fname in tif_files:
            base = os.path.splitext(fname)[0]
            out_subdir = os.path.join(output_folder, f"{base}_frames")
            convert_tif_to_frames(
                os.path.join(input_path, fname),
                out_subdir,
                image_format=image_format,
                nth=nth,
                num_workers=num_workers,
            )
        return

    os.makedirs(output_folder, exist_ok=True)

    try:
        tiff_stack = tifffile.imread(input_path)
        print(f"Data shape: {tiff_stack.shape} - reading {input_path}")
    except (OSError, ValueError) as e:
        print(f"Error reading TIF '{input_path}': {e}")
        return

    tasks = []
    saved_count = 0
    frame_indices = range(0, len(tiff_stack), nth)

    for i in frame_indices:
        dst = os.path.join(output_folder, f"frame_{saved_count:05d}.{image_format}")
        tasks.append((tiff_stack[i], dst, image_format))
        saved_count += 1

    if num_workers == 1:
        for t in tqdm(tasks, desc="Saving frames", unit="frame"):
            _process_frame(t)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(
                    executor.map(_process_frame, tasks),
                    total=len(tasks),
                    desc="Saving frames",
                    unit="frame",
                )
            )

    print(f"Successfully converted {saved_count} frames (every {nth}th) to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert multi-page TIFF to image frames.")
    parser.add_argument(
        "input_path", help="Path to input .tif file or a directory containing .tif files"
    )
    parser.add_argument("output_dir", nargs="?", default=None, help="Directory to save frames")
    parser.add_argument(
        "--jpg", action="store_true", dest="convert_jpg", help="Convert JPG images instead of TIFF"
    )
    parser.add_argument(
        "-n", "--nth", type=int, default=10, help="Save every nth frame (default: 10)"
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="image_format",
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel workers (default: CPUs-1)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        args.output_dir = os.path.join(os.getcwd(), f"{base_name}_frames")

    if args.convert_jpg:
        convert_jpg_to_frames(
            args.input_path, args.output_dir, args.image_format, num_workers=args.workers
        )
    else:
        convert_tif_to_frames(
            args.input_path, args.output_dir, args.image_format, args.nth, num_workers=args.workers
        )
