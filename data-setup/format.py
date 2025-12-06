"""Convert multi-page TIFF to image frames."""

import argparse
import os

import cv2
import numpy as np
import tifffile


def convert_jpg_to_frames(input_path, output_folder, image_format="png"):
    """Convert a JPG image to different image format."""
    if os.path.isdir(input_path):
        jpg_files = sorted(
            [f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".jpeg"))]
        )
        if not jpg_files:
            print(f"No .jpg files found in directory: {input_path}")
            return
        for fname in jpg_files:
            base = os.path.splitext(fname)[0]
            convert_jpg_to_frames(
                os.path.join(input_path, fname),
                output_folder,
                image_format=image_format,
            )
        return

    os.makedirs(output_folder, exist_ok=True)

    try:
        print(input_path)
        img = cv2.imread(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        cv2.imwrite(os.path.join(output_folder, f"{base}.{image_format}"), img)
    except cv2.error as e:
        print(f"Error converting JPG '{input_path}': {e}")


def convert_tif_to_frames(input_path, output_folder, image_format="png", nth=10):
    """Convert multi-page TIFF files to individual image frames saved in output_folder."""

    # If input_path is a directory, find .tif files and process each
    if os.path.isdir(input_path):
        tif_files = sorted(
            [f for f in os.listdir(input_path) if f.lower().endswith(".tif")]
        )
        if not tif_files:
            print(f"No .tif files found in directory: {input_path}")
            return
        for fname in tif_files:
            base = os.path.splitext(fname)[0]
            out_subdir = os.path.join(output_folder, f"{base}_frames")
            os.makedirs(out_subdir, exist_ok=True)
            convert_tif_to_frames(
                os.path.join(input_path, fname),
                out_subdir,
                image_format=image_format,
                nth=nth,
            )
        return

    # 1. Create output directory (for single file)
    os.makedirs(output_folder, exist_ok=True)

    # 2. Read the TIF file
    try:
        tiff_stack = tifffile.imread(input_path)
        print(
            f"Data shape: {tiff_stack.shape} - reading {input_path}"
        )  # Usually (Frames, Height, Width)
    except (OSError, ValueError) as e:
        print(f"Error reading TIF '{input_path}': {e}")
        return

    # 3. Iterate through frames and save every nth frame
    saved_count = 0
    for i in range(0, len(tiff_stack), nth):
        frame = tiff_stack[i]

        # 4. Normalization
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        # 5. Handle Color Channels
        if len(frame.shape) == 2:
            frame = cv2.merge([frame, frame, frame])

        # 6. Save Frame
        cv2.imwrite(
            os.path.join(output_folder, f"frame_{saved_count:05d}.{image_format}"),
            frame,
        )
        saved_count += 1

    print(
        f"Successfully converted {saved_count} frames (every {nth}th) to {output_folder}"
    )


INPUT_PATH = (
    "/mnt/c/Users/ankho/git/molecular-dynamics-simulation/raw_data/2024.07.02/"
    "Trial 1 Au Citrate Best Trials/"
    "Au Cit+1% of 2um PS+NaCl 20% Light Intensity Test Video 300 ms Trial 17_1"
)

# create output directory in current working directory based on input_path name
base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
output_dir = os.path.join(os.getcwd(), f"{base_name}_frames")
os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert multi-page TIFF to image frames."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=INPUT_PATH,
        help="Path to input .tif file or a directory containing .tif files",
    )
    parser.add_argument(
        "output_dir", nargs="?", default=output_dir, help="Directory to save frames"
    )
    parser.add_argument(
        "convert_jpg",
        type=bool,
        default=False,
        help="Set to True to convert JPG images instead of TIFF",
    )
    parser.add_argument(
        "-n",
        "--nth",
        dest="nth",
        type=int,
        default=10,
        help="Save every nth frame (default: 10)",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="image_format",
        default="png",
        help="Output image format (default: png)",
    )
    args = parser.parse_args()

    if args.convert_jpg:
        convert_jpg_to_frames(args.input_path, args.output_dir, args.image_format)
    else:
        convert_tif_to_frames(
            args.input_path, args.output_dir, args.image_format, args.nth
        )
