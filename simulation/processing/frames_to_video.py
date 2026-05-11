#!/usr/bin/env python3
"""
Convert a folder of PNG frames into an MP4 video using FFmpeg.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def natural_sort_key(s):
    """
    Key function for natural sorting (e.g., frame_1.png, frame_2.png, ..., frame_10.png).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def create_video_from_frames(input_dir, output_file, fps=5, pattern="*.png", crf=23):
    """
    Create a video from a folder of images using FFmpeg.

    Args:
        input_dir: Directory containing the images
        output_file: Path to the output video file
        fps: Frames per second
        pattern: Glob pattern to match images
        crf: Constant Rate Factor for quality (0-51, lower is better, 23 is default)
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return False

    # Get all matching files and sort them naturally
    frames = sorted(list(input_path.glob(pattern)), key=natural_sort_key)

    if not frames:
        print(f"Error: No files matching '{pattern}' found in '{input_dir}'.")
        return False

    print(f"Found {len(frames)} frames. Starting conversion...")

    # Create a temporary file list for FFmpeg 'concat' demuxer
    # This avoids issues with shell globbing limits or complex filename patterns
    list_file = input_path / "ffmpeg_frames_list.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for frame in frames:
                # FFmpeg concat demuxer needs escaped single quotes and paths relative to the list file
                # or absolute paths. We'll use absolute paths for simplicity.
                f.write(f"file '{frame.absolute()}'\n")
                f.write(f"duration {1/fps}\n")

        # FFmpeg command
        # -y: overwrite output
        # -f concat: use concat demuxer
        # -safe 0: allow absolute paths in list file
        # -i: input list file
        # -vcodec libx264: use H.264 codec
        # -pix_fmt yuv420p: ensure compatibility with most players
        # -crf: quality setting
        # -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2": ensure dimensions are divisible by 2 (required for yuv420p)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(output_file),
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created: {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e.stderr}")
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        if list_file.exists():
            os.remove(list_file)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of PNG frames into an MP4 video using FFmpeg."
    )
    parser.add_argument("input", help="Directory containing the PNG frames")
    parser.add_argument("-o", "--output", help="Output MP4 file path (default: input_dir.mp4)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30.0)")
    parser.add_argument("--pattern", default="*.png", help="File pattern to match (default: *.png)")
    parser.add_argument(
        "--crf", type=int, default=23, help="Quality (CRF 0-51, lower is better, default 23)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_dir.with_suffix(".mp4")

    if create_video_from_frames(input_dir, output_file, args.fps, args.pattern, args.crf):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
