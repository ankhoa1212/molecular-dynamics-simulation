"""Auto-label microscopy image frames using trackpy particle tracking."""

import argparse
import glob
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
from PIL import Image
from tqdm import tqdm

# Suppress trackpy's verbose per-frame output by default
tp.quiet()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _collect_image_files(input_dir):
    """Return sorted list of image paths from input_dir."""
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.*")))
    return [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]


def _load_frames(image_files):
    """Load grayscale images and normalise each frame to [0, 255] uint8."""
    frames = []
    for fpath in tqdm(image_files, desc="Loading frames"):
        img = np.array(Image.open(fpath).convert("L"))
        if img.dtype != np.uint8:
            lo, hi = img.min(), img.max()
            img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Detection & tracking
# ---------------------------------------------------------------------------


def _detect_all_frames(frames, diameter, min_mass, separation, percentile):
    """Run trackpy.locate on every frame and return a list of DataFrames."""
    results = []
    for frame in tqdm(frames, desc="Detecting particles"):
        df = tp.locate(
            frame,
            diameter=diameter,
            minmass=min_mass,
            separation=separation,
            percentile=percentile,
        )
        results.append(df)
    return results


def _track_particles(per_frame_dfs, search_range, memory):
    """Link detections across frames with trackpy.link_df."""
    import pandas as pd

    # Add frame index column
    tagged = []
    for idx, df in enumerate(per_frame_dfs):
        df = df.copy()
        df["frame"] = idx
        tagged.append(df)

    all_dets = pd.concat(tagged, ignore_index=True)
    if all_dets.empty:
        return all_dets

    linked = tp.link_df(all_dets, search_range=search_range, memory=memory)
    return linked


# ---------------------------------------------------------------------------
# YOLO output
# ---------------------------------------------------------------------------


def _to_yolo(x, y, box_px, frame_w, frame_h):
    """Convert pixel (x, y) centre + box size to normalised YOLO values."""
    x_c = max(0.0, min(1.0, x / frame_w))
    y_c = max(0.0, min(1.0, y / frame_h))
    n_w = max(0.0, min(1.0, box_px / frame_w))
    n_h = max(0.0, min(1.0, box_px / frame_h))
    return x_c, y_c, n_w, n_h


def _write_labels(image_files, frames, per_frame_dfs, output_dir, box_size, do_plot):
    """Write one YOLO .txt label file per frame and optionally save overlay PNGs."""
    os.makedirs(output_dir, exist_ok=True)

    for idx, (fpath, frame, df) in enumerate(
        tqdm(
            zip(image_files, frames, per_frame_dfs),
            total=len(image_files),
            desc="Writing labels",
        )
    ):
        frame_h, frame_w = frame.shape[:2]
        base = os.path.splitext(os.path.basename(fpath))[0]
        txt_path = os.path.join(output_dir, f"{base}.txt")

        fig, ax = plt.subplots(1) if do_plot else (None, None)
        if do_plot:
            ax.imshow(frame, cmap="gray")

        with open(txt_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                x_c, y_c, n_w, n_h = _to_yolo(row["x"], row["y"], box_size, frame_w, frame_h)
                f.write(f"0 {x_c:.6f} {y_c:.6f} {n_w:.6f} {n_h:.6f}\n")

                if do_plot:
                    rect = patches.Rectangle(
                        (row["x"] - box_size / 2, row["y"] - box_size / 2),
                        box_size,
                        box_size,
                        linewidth=1.5,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

        if do_plot:
            ax.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"{base}_overlay.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-label microscopy frames with trackpy → YOLO format."
    )
    parser.add_argument(
        "input_dir",
        help="Directory of image frames to label.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for YOLO .txt labels (default: <input_dir>_labels).",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=11,
        help="Expected particle diameter in pixels (must be odd). Default: 11.",
    )
    parser.add_argument(
        "--min-mass",
        type=float,
        default=None,
        help="Minimum integrated brightness to keep a detection. "
        "If omitted, trackpy's default heuristic is used.",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=None,
        help="Minimum centre-to-centre distance between particles (pixels). "
        "Defaults to diameter + 1.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=64.0,
        help="Features must be brighter than this percentile of pixels. Default: 64.",
    )
    parser.add_argument(
        "--search-range",
        type=float,
        default=15.0,
        help="Max displacement between frames for linking (pixels). Default: 15.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=3,
        help="Frames a particle may vanish and still be linked. Default: 3.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=40,
        help="Fixed bounding box size in pixels for YOLO labels. Default: 40.",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Skip linking; detect independently per frame (faster, no track IDs).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save overlay PNGs alongside label files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip("/\\") + "_labels"

    image_files = _collect_image_files(args.input_dir)
    if not image_files:
        print(f"No valid images found in {args.input_dir}")
        return
    print(f"Found {len(image_files)} frame(s).")

    frames = _load_frames(image_files)

    separation = args.separation if args.separation is not None else args.diameter + 1
    min_mass = args.min_mass  # None → trackpy default

    per_frame_dfs = _detect_all_frames(
        frames,
        diameter=args.diameter,
        min_mass=min_mass,
        separation=separation,
        percentile=args.percentile,
    )

    total_dets = sum(len(df) for df in per_frame_dfs)
    print(
        f"Detections — total: {total_dets}  "
        f"min/frame: {min(len(d) for d in per_frame_dfs)}  "
        f"max/frame: {max(len(d) for d in per_frame_dfs)}  "
        f"mean/frame: {total_dets / max(1, len(per_frame_dfs)):.1f}"
    )

    if not args.no_track:
        print("Linking particles across frames...")
        linked = _track_particles(per_frame_dfs, args.search_range, args.memory)
        # Rebuild per-frame list from linked DataFrame for label writing
        if not linked.empty:
            per_frame_dfs = [linked[linked["frame"] == i] for i in range(len(frames))]
        print(
            f"Unique particle tracks: " f"{linked['particle'].nunique() if not linked.empty else 0}"
        )

    _write_labels(
        image_files,
        frames,
        per_frame_dfs,
        output_dir=args.output_dir,
        box_size=args.box_size,
        do_plot=args.plot,
    )
    print(f"Done. Labels saved to {args.output_dir}")


if __name__ == "__main__":
    main()
