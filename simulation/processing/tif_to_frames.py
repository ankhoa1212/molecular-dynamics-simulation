#!/usr/bin/env python3
"""
Extract individual frames from a multi-page TIFF file and save them as PNGs.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageSequence


def extract_frames(tif_path, output_dir=None, format="png"):
    """
    Extract frames from a TIFF file.

    Args:
        tif_path: Path to the .tif or .tiff file
        output_dir: Directory to save the frames
        format: Image format to save as (default: png)
    """
    tif_file = Path(tif_path)
    if not tif_file.exists():
        print(f"Error: File '{tif_path}' not found.")
        return False

    # Set up output directory
    if output_dir is None:
        output_dir = tif_file.parent / f"{tif_file.stem}_frames"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening: {tif_file}")
    try:
        img = Image.open(tif_file)

        # Count total frames (optional but nice for progress)
        total_frames = getattr(img, "n_frames", 1)
        print(f"Found {total_frames} frames. Extracting to {output_dir}...")

        # Extract each frame
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            output_path = output_dir / f"frame_{i:04d}.{format}"
            # Convert to RGB if necessary (some TIFFs are in weird formats)
            if frame.mode not in ("L", "RGB", "RGBA"):
                frame = frame.convert("RGB")

            frame.save(output_path)

            if (i + 1) % 10 == 0 or (i + 1) == total_frames:
                print(f"Extracted {i + 1}/{total_frames} frames...", end="\r")

        print(f"\nSuccessfully extracted all frames to: {output_dir}")
        return True

    except Exception as e:
        print(f"\nError processing {tif_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual frames from a multi-page TIFF file."
    )
    parser.add_argument("input", help="Path to the multi-page TIFF file")
    parser.add_argument("-o", "--output", help="Output directory (default: input_filename_frames)")
    parser.add_argument("--format", default="png", help="Output image format (default: png)")

    args = parser.parse_args()

    if extract_frames(args.input, args.output, args.format):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
