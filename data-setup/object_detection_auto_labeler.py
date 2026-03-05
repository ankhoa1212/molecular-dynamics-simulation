"""Auto-label microscopy images using LodeSTAR unsupervised object detection."""
import argparse
import dataclasses
import glob
import logging
import os
import random

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import deeplay as dl
import deeptrack as dt

logging.getLogger("pint").setLevel(logging.ERROR)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label images using LodeSTAR."
    )
    parser.add_argument(
        "--input-dir", type=str,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--input-file", type=str,
        help="Path to a single input image.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save YOLO label txt files.",
    )
    parser.add_argument(
        "--box-size", type=int, default=40,
        help="Fixed bounding box size in pixels. Used when --use-radius is not set.",
    )
    parser.add_argument(
        "--num-outputs", type=int, default=3,
        help="Number of LodeSTAR output channels. 2 = (x,y) only; 3 = (x,y,radius).",
    )
    parser.add_argument(
        "--use-radius", action="store_true",
        help=(
            "Use LodeSTAR\'s per-detection radius estimate as box size "
            "instead of --box-size. Requires --num-outputs >= 3."
        ),
    )
    parser.add_argument(
        "--radius-scale", type=float, default=1.0,
        help=(
            "Multiplier applied to the raw radius output to convert it to "
            "pixels (box half-width). Tune this for your particle size."
        ),
    )
    parser.add_argument(
        "--min-box-size", type=float, default=0.0,
        help=(
            "Minimum box size in pixels when --use-radius is active. "
            "0 = use --box-size as the floor."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of max epochs for training LodeSTAR.",
    )
    parser.add_argument(
        "--crop-size", type=int, default=64,
        help="Size of the random crop for training.",
    )
    parser.add_argument(
        "--num-crops", type=int, default=8,
        help="Number of training crops to sample (from multiple frames).",
    )
    parser.add_argument(
        "--nms-distance", type=float, default=0.0,
        help=(
            "Minimum pixel distance between detections (NMS). "
            "0 disables NMS. Set to ~box_size to suppress duplicates."
        ),
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Alpha parameter for LodeSTAR detect.",
    )
    parser.add_argument(
        "--cutoff", type=float, default=0.5,
        help="Cutoff parameter for LodeSTAR detect. Meaning depends on --detect-mode.",
    )
    parser.add_argument(
        "--detect-mode", type=str, default="ratio",
        choices=["quantile", "ratio", "constant"],
        help=(
            "Thresholding mode for LodeSTAR detect. "
            "\'ratio\': cutoff * max_score (recommended for large images). "
            "\'quantile\': score quantile as h_maxima height (strict on large images). "
            "\'constant\': fixed threshold."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of DataLoader worker processes. 0 is safest.",
    )
    parser.add_argument(
        "--detect-batch-size", type=int, default=4,
        help="Batch size for detection. Keep low (e.g. 1-4) to prevent OOM.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Overlay bounding boxes on the input image and save as PNG.",
    )
    return parser.parse_args()


def _load_and_normalise(image_files):
    """Load grayscale images and apply per-frame [0, 1] normalisation."""
    raw_images = []
    for fpath in image_files:
        img = Image.open(fpath).convert("L")
        raw_images.append(np.array(img))
    data_raw = np.array(raw_images)
    data = np.zeros_like(data_raw, dtype=np.float32)
    for idx, frame in enumerate(data_raw):
        frame_f = frame.astype(np.float32)
        f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
        data[idx] = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min
    return raw_images, data


def _variance_guided_crop(frame, crop_size, top_k=32):
    """Return (y0, x0) of the highest-variance crop_size x crop_size patch."""
    max_y = max(0, frame.shape[0] - crop_size)
    max_x = max(0, frame.shape[1] - crop_size)
    candidates = [
        (random.randint(0, max_y), random.randint(0, max_x))
        for _ in range(top_k)
    ]
    best_y, best_x, best_var = candidates[0][0], candidates[0][1], -1.0
    for cy, cx in candidates:
        patch = frame[cy:cy + crop_size, cx:cx + crop_size]
        variance = float(np.var(patch))
        if variance > best_var:
            best_var, best_y, best_x = variance, cy, cx
    return best_y, best_x


def _make_pipeline(crop_array):
    """Build a deeptrack augmentation pipeline for a single crop."""
    channel_crop = np.expand_dims(crop_array, axis=-1)  # (H, W, 1)
    return (
        dt.Value(channel_crop)
        >> dt.Multiply(lambda: np.random.uniform(0.85, 1.15))
        >> dt.Add(lambda: np.random.uniform(-0.05, 0.05))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )


def _collect_detections(batch_detections, out_list):
    """Move detections to CPU numpy and append per-frame results to out_list."""
    if isinstance(batch_detections, (list, tuple)):
        for frame_dets in batch_detections:
            if isinstance(frame_dets, torch.Tensor):
                out_list.append(frame_dets.detach().cpu().numpy())
            elif isinstance(frame_dets, (list, tuple)):
                out_list.append([
                    d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d
                    for d in frame_dets
                ])
            else:
                out_list.append(frame_dets)
    else:
        if isinstance(batch_detections, torch.Tensor):
            batch_detections = batch_detections.detach().cpu().numpy()
            for frame_dets in batch_detections:
                out_list.append(frame_dets)
        else:
            out_list.append(batch_detections)


def _nms(detections, min_dist):
    """Non-maximum suppression: remove detections within min_dist pixels."""
    if min_dist <= 0 or len(detections) == 0:
        return detections
    arr = np.array(detections)
    keep = []
    suppressed = np.zeros(len(arr), dtype=bool)
    for idx_i, det_i in enumerate(detections):
        if suppressed[idx_i]:
            continue
        keep.append(det_i)
        yi, xi = arr[idx_i, 0], arr[idx_i, 1]
        for idx_j in range(idx_i + 1, len(arr)):
            if suppressed[idx_j]:
                continue
            dist = np.sqrt(
                (arr[idx_j, 0] - yi) ** 2 + (arr[idx_j, 1] - xi) ** 2
            )
            if dist < min_dist:
                suppressed[idx_j] = True
    return keep


def _print_radius_stats(all_detections, radius_scale):
    """Print per-detection radius statistics to aid --radius-scale calibration."""
    all_radii = [
        float(det[2])
        for frame_dets in all_detections
        for det in (
            frame_dets
            if not isinstance(frame_dets, np.ndarray)
            else [frame_dets]
        )
        if hasattr(det, "__len__") and len(det) >= 3
    ]
    if not all_radii:
        return
    scaled_min = min(abs(r) * radius_scale for r in all_radii)
    scaled_max = max(abs(r) * radius_scale for r in all_radii)
    print(
        f"Raw radius channel stats (before scaling): "
        f"min={min(all_radii):.3f}  max={max(all_radii):.3f}  "
        f"mean={sum(all_radii) / len(all_radii):.3f}  "
        f"-> box_px range with scale={radius_scale}: "
        f"{scaled_min:.1f} - {scaled_max:.1f} px"
    )


@dataclasses.dataclass
class _SaveConfig:
    """Parameters controlling label output and overlay rendering."""
    output_dir: str
    frame_shape: tuple  # (height, width)
    use_radius: bool
    radius_scale: float
    min_box_px: float
    box_size: int
    do_plot: bool

    @property
    def frame_h(self):
        """Frame height in pixels."""
        return self.frame_shape[0]

    @property
    def frame_w(self):
        """Frame width in pixels."""
        return self.frame_shape[1]


def _det_box_px(det, cfg):
    """Return the box size in pixels for a single detection."""
    if cfg.use_radius and len(det) >= 3:
        return max(cfg.min_box_px, abs(float(det[2])) * cfg.radius_scale)
    return float(cfg.box_size)


def _yolo_coords(det_y, det_x, box_px, frame_h, frame_w):
    """Return normalised (x_c, y_c, w, h) YOLO values for one detection."""
    x_c = max(0.0, min(1.0, det_x / frame_w))
    y_c = max(0.0, min(1.0, det_y / frame_h))
    n_w = max(0.0, min(1.0, box_px / frame_w))
    n_h = max(0.0, min(1.0, box_px / frame_h))
    return x_c, y_c, n_w, n_h


def _write_detection(label_file, det, cfg, ax):
    """Write one detection to the label file and optionally annotate ax."""
    # LodeSTAR forward: ch0 = X (row -> image y), ch1 = Y (col -> image x)
    det_y, det_x = det[0], det[1]
    box_px = _det_box_px(det, cfg)
    x_c, y_c, n_w, n_h = _yolo_coords(det_y, det_x, box_px, cfg.frame_h, cfg.frame_w)
    label_file.write(f"0 {x_c:.6f} {y_c:.6f} {n_w:.6f} {n_h:.6f}\n")
    if ax is not None:
        rect = patches.Rectangle(
            (det_x - box_px / 2, det_y - box_px / 2),
            box_px, box_px,
            linewidth=2, edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)
        ax.plot(det_x, det_y, "+", color="cyan", markersize=4, markeredgewidth=1)


def _write_frame(frame_file, image, frame_dets, cfg):
    """Write one YOLO label file and optionally save an overlay PNG."""
    base_name = os.path.splitext(os.path.basename(frame_file))[0]
    txt_path = os.path.join(cfg.output_dir, f"{base_name}.txt")

    fig, ax = (plt.subplots(1) if cfg.do_plot else (None, None))
    if cfg.do_plot:
        ax.imshow(image, cmap="gray")

    with open(txt_path, "w", encoding="utf-8") as label_file:
        for det in frame_dets:
            _write_detection(label_file, det, cfg, ax if cfg.do_plot else None)

    if cfg.do_plot:
        ax.axis("off")
        plot_path = os.path.join(cfg.output_dir, f"{base_name}_overlay.png")
        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _save_labels_and_plots(image_files, images, all_detections, cfg):
    """Write YOLO .txt label files and optionally save overlay PNGs."""
    print("Saving YOLO labels...")
    for frame_idx, (frame_file, frame_dets) in enumerate(
        tqdm(zip(image_files, all_detections), total=len(image_files), desc="Processing files")
    ):
        _write_frame(frame_file, images[frame_idx], frame_dets, cfg)


def _run_inference(lodestar, data_tensor, args, beta):
    """Run batched detection with GPU OOM fallback."""
    detect_device = "cuda" if torch.cuda.is_available() else "cpu"
    lodestar = lodestar.to(detect_device)

    def _run_detect(frames_tensor, device):
        return lodestar.detect(
            frames_tensor.to(device),
            alpha=args.alpha,
            beta=beta,
            cutoff=args.cutoff,
            mode=args.detect_mode,
        )

    all_detections = []
    with torch.inference_mode():
        for i in tqdm(
            range(0, len(data_tensor), args.detect_batch_size),
            desc="Detecting targets",
        ):
            batch = data_tensor[i: i + args.detect_batch_size]
            try:
                batch_dets = _run_detect(batch, detect_device)
                if detect_device == "cuda":
                    torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                print(
                    f"\nOOM at batch {i // args.detect_batch_size}. "
                    "Retrying frame-by-frame on GPU..."
                )
                if detect_device == "cuda":
                    torch.cuda.empty_cache()
                batch_dets = []
                for j in range(len(batch)):
                    single = batch[j: j + 1]
                    try:
                        det = _run_detect(single, detect_device)
                        if detect_device == "cuda":
                            torch.cuda.empty_cache()
                    except torch.OutOfMemoryError:
                        print(f"  Still OOM for frame {i + j}. Falling back to CPU...")
                        torch.cuda.empty_cache()
                        lodestar = lodestar.to("cpu")
                        det = _run_detect(single, "cpu")
                        lodestar = lodestar.to(detect_device)
                    _collect_detections(det, batch_dets)
                all_detections.extend(batch_dets)
                continue
            _collect_detections(batch_dets, all_detections)
    return all_detections


def _collect_image_files(args):
    """Return the sorted list of valid image paths from args."""
    if args.input_file:
        if os.path.exists(args.input_file):
            return [args.input_file]
        print(f"File not found: {args.input_file}")
        return []
    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]


def _build_and_train(args, data):
    """Build crops, train LodeSTAR, and return the trained model."""
    img_h, img_w = data.shape[1], data.shape[2]
    crop_size = min(args.crop_size, img_h, img_w)

    frame_indices = list(range(len(data)))
    random.shuffle(frame_indices)
    crops = []
    for fi in frame_indices[: args.num_crops]:
        y0, x0 = _variance_guided_crop(data[fi], crop_size)
        crops.append((fi, y0, x0, data[fi, y0:y0 + crop_size, x0:x0 + crop_size]))

    print(f"Training LodeSTAR on {len(crops)} variance-guided {crop_size}x{crop_size} crops.")
    for fi, y0, x0, _ in crops:
        print(f"  frame={fi}  x={x0}  y={y0}")

    if args.use_radius and args.num_outputs < 3:
        print("WARNING: --use-radius requires --num-outputs >= 3. Setting to 3.")
        args.num_outputs = 3

    lodestar = dl.LodeSTAR(
        n_transforms=4, num_outputs=args.num_outputs, optimizer=dl.Adam(lr=1e-3),
    ).build()
    datasets = [
        dt.pytorch.Dataset(_make_pipeline(c[3]), length=128, replace=False)
        for c in crops
    ]
    dataloader = dl.DataLoader(
        dataset=torch.utils.data.ConcatDataset(datasets),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    dl.Trainer(accelerator="auto", max_epochs=args.epochs, log_every_n_steps=10).fit(
        lodestar, dataloader
    )
    return lodestar


def _print_detection_summary(all_detections):
    """Print a one-line count summary across all frames."""
    total_dets = sum(len(d) for d in all_detections)
    print(
        f"Detections per frame: "
        f"min={min(len(d) for d in all_detections)}  "
        f"max={max(len(d) for d in all_detections)}  "
        f"mean={total_dets / max(1, len(all_detections)):.1f}  "
        f"total={total_dets}"
    )


def main():
    """Entry point."""
    args = parse_args()

    if not args.input_dir and not args.input_file:
        raise ValueError("Either --input-dir or --input-file must be provided.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = _collect_image_files(args)
    if not image_files:
        print("No valid images found.")
        return
    print(f"Loaded {len(image_files)} image(s).")

    images, data = _load_and_normalise(image_files)
    lodestar = _build_and_train(args, data)

    data_tensor = torch.from_numpy(data).to(torch.float32).unsqueeze(1)
    print(len(data_tensor))
    print("Detecting objects across all frames...")
    lodestar.eval()
    all_detections = _run_inference(lodestar, data_tensor, args, 1.0 - args.alpha)

    if args.nms_distance > 0:
        all_detections = [_nms(list(d), args.nms_distance) for d in all_detections]

    _print_detection_summary(all_detections)

    if args.use_radius and args.num_outputs >= 3:
        _print_radius_stats(all_detections, args.radius_scale)

    min_box_px = args.min_box_size if args.min_box_size > 0 else float(args.box_size)
    cfg = _SaveConfig(
        output_dir=args.output_dir,
        frame_shape=(data.shape[1], data.shape[2]),
        use_radius=args.use_radius,
        radius_scale=args.radius_scale,
        min_box_px=min_box_px,
        box_size=args.box_size,
        do_plot=args.plot,
    )
    _save_labels_and_plots(image_files, images, all_detections, cfg)
    print(f"Finished auto-labeling. Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
