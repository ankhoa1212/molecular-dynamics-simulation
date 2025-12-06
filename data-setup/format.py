import tifffile
import cv2
import numpy as np
import os
import argparse

def convert_tif_to_frames(input_path, output_folder, image_format='png', nth=10):
    # ensure nth is valid
    try:
        nth = int(nth)
    except Exception:
        nth = 1
    if nth <= 0:
        nth = 1

    # If input_path is a directory, find .tif files and process each
    if os.path.isdir(input_path):
        tif_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith('.tif')])
        if not tif_files:
            print(f"No .tif files found in directory: {input_path}")
            return
        for fname in tif_files:
            file_path = os.path.join(input_path, fname)
            base = os.path.splitext(fname)[0]
            out_subdir = os.path.join(output_folder, f"{base}_frames")
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir, exist_ok=True)
            convert_tif_to_frames(file_path, out_subdir, image_format=image_format, nth=nth)
        return

    # 1. Create output directory (for single file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 2. Read the TIF file
    try:
        tiff_stack = tifffile.imread(input_path)
        print(f"Data shape: {tiff_stack.shape} - reading {input_path}")  # Usually (Frames, Height, Width)
    except Exception as e:
        print(f"Error reading TIF '{input_path}': {e}")
        return

    # 3. Iterate through frames and save every nth frame
    saved_count = 0
    for i, frame in enumerate(tiff_stack):
        if (i % nth) != 0:
            continue

        # 4. Normalization
        if frame.dtype != np.uint8:
            frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame_8bit = frame_norm.astype('uint8')
        else:
            frame_8bit = frame

        # 5. Handle Color Channels
        if len(frame_8bit.shape) == 2:
            frame_output = cv2.merge([frame_8bit, frame_8bit, frame_8bit])
        else:
            frame_output = frame_8bit

        # 6. Save Frame
        file_name = f"frame_{saved_count:05d}.{image_format}"
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, frame_output)
        saved_count += 1

    print(f"Successfully converted {saved_count} frames (every {nth}th) to {output_folder}")
    return os.path.abspath(output_folder)

input_path = "/mnt/c/Users/ankho/git/molecular-dynamics-simulation/raw_data/2024.07.02/Trial 1 Au Citrate Best Trials/Au Cit+1% of 2um PS+NaCl 20% Light Intensity Test Video 300 ms Trial 17_1"

# create output directory in current working directory based on input_path name
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = os.path.join(os.getcwd(), f"{base_name}_frames")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert multi-page TIFF to image frames.")
    parser.add_argument("input_path", nargs="?", default=input_path, help="Path to input .tif file or a directory containing .tif files")
    parser.add_argument("output_dir", nargs="?", default=output_dir, help="Directory to save frames")
    parser.add_argument("-n", "--nth", dest="nth", type=int, default=10, help="Save every nth frame (default: 10)")
    parser.add_argument("-f", "--format", dest="image_format", default="png", help="Output image format (default: png)")
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    image_format = args.image_format

    convert_tif_to_frames(input_path, output_dir, image_format=image_format)

