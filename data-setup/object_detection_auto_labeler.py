import argparse
import glob
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import deeplay as dl
import deeptrack as dt
import numpy as np
import torch
from PIL import Image
import logging
logging.getLogger("pint").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label images using LodeSTAR.")
    parser.add_argument("--input-dir", type=str, help="Directory containing input images.")
    parser.add_argument("--input-file", type=str, help="Path to a single input image.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save YOLO label txt files.")
    parser.add_argument("--box-size", type=int, default=40, help="Fixed bounding box size in pixels. Used when --use-radius is not set.")
    parser.add_argument("--num-outputs", type=int, default=3, help="Number of LodeSTAR output channels. 2 = (x,y) only; 3 = (x,y,radius).")
    parser.add_argument("--use-radius", action="store_true", help="Use LodeSTAR's per-detection radius estimate as box size instead of --box-size. Requires --num-outputs >= 3.")
    parser.add_argument("--radius-scale", type=float, default=1.0, help="Multiplier applied to the raw radius output to convert it to pixels (box half-width). Tune this for your particle size.")
    parser.add_argument("--min-box-size", type=float, default=0.0, help="Minimum box size in pixels when --use-radius is active. 0 = use --box-size as the floor.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of max epochs for training LodeSTAR.")
    parser.add_argument("--crop-size", type=int, default=64, help="Size of the random crop for training.")
    parser.add_argument("--num-crops", type=int, default=8, help="Number of training crops to sample (from multiple frames).")
    parser.add_argument("--nms-distance", type=float, default=0.0, help="Minimum pixel distance between detections (NMS). 0 disables NMS. Set to ~box_size to suppress duplicates in dense fields.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for LodeSTAR detect.")
    parser.add_argument("--cutoff", type=float, default=0.5, help="Cutoff parameter for LodeSTAR detect. Meaning depends on --detect-mode.")
    parser.add_argument("--detect-mode", type=str, default="ratio", choices=["quantile", "ratio", "constant"], help="Thresholding mode for LodeSTAR detect. 'ratio': cutoff * max_score (recommended for large images). 'quantile': score quantile value as h_maxima height (strict on large images). 'constant': fixed threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes. 0 is safest to avoid multiprocessing issues.")
    parser.add_argument("--detect-batch-size", type=int, default=4, help="Batch size for detection. Keep low (e.g., 4 or 8) to prevent OOM since LodeSTAR loops frames sequentially on CPU.")
    parser.add_argument("--plot", action="store_true", help="Overlay bounding box labels over the corresponding input image and save as PNG.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.input_dir and not args.input_file:
        raise ValueError("Either --input-dir or --input-file must be provided.")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    image_files = []
    if args.input_file:
        if os.path.exists(args.input_file):
            image_files.append(args.input_file)
        else:
            print(f"File not found: {args.input_file}")
            return
    elif args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        
    if not image_files:
        print(f"No valid images found.")
        return
        
    print(f"Loaded {len(image_files)} image(s).")
    
    images = []
    for f in image_files:
        img = Image.open(f).convert("L") # Convert to grayscale
        images.append(np.array(img))

    data_raw = np.array(images)

    # Per-frame normalization: each frame is independently scaled to [0, 1].
    # This preserves intra-frame contrast regardless of global illumination drift.
    data = np.zeros_like(data_raw, dtype=np.float32)
    for idx in range(len(data_raw)):
        frame = data_raw[idx].astype(np.float32)
        f_min, f_ptp = frame.min(), np.ptp(frame)
        data[idx] = (frame - f_min) / f_ptp if f_ptp != 0 else frame - f_min
        
    H, W = data.shape[1], data.shape[2]
    crop_size = min(args.crop_size, H, W)

    def _variance_guided_crop(frame, cs, top_k=32):
        """Return the (y0, x0) of the cs×cs patch with the highest local variance.
        Samples top_k candidate positions and picks the best to balance diversity
        with particle richness."""
        max_y = max(0, frame.shape[0] - cs)
        max_x = max(0, frame.shape[1] - cs)
        candidates = [
            (random.randint(0, max_y), random.randint(0, max_x))
            for _ in range(top_k)
        ]
        best_y, best_x, best_var = candidates[0][0], candidates[0][1], -1.0
        for cy, cx in candidates:
            patch = frame[cy:cy+cs, cx:cx+cs]
            v = float(np.var(patch))
            if v > best_var:
                best_var, best_y, best_x = v, cy, cx
        return best_y, best_x

    # Collect num_crops training crops from diverse frames
    frame_indices = list(range(len(data)))
    random.shuffle(frame_indices)
    crop_frame_indices = frame_indices[:args.num_crops]

    crops = []
    for fi in crop_frame_indices:
        y0, x0 = _variance_guided_crop(data[fi], crop_size)
        crop = data[fi, y0:y0+crop_size, x0:x0+crop_size]
        crops.append((fi, y0, x0, crop))

    print(f"Training LodeSTAR on {len(crops)} variance-guided {crop_size}×{crop_size} crops.")
    for fi, y0, x0, _ in crops:
        print(f"  frame={fi}  x={x0}  y={y0}")

    # Build a pipeline that randomly draws from one of the selected crops each step
    def _make_pipeline(crop_array):
        c = np.expand_dims(crop_array, axis=-1)  # (H, W, 1)
        return (
            dt.Value(c)
            >> dt.Multiply(lambda: np.random.uniform(0.85, 1.15))
            >> dt.Add(lambda: np.random.uniform(-0.05, 0.05))
            >> dt.MoveAxis(-1, 0)
            >> dt.pytorch.ToTensor(dtype=torch.float32)
        )

    # Use the first crop as the primary pipeline; remaining crops augment with extra datasets
    training_pipeline = _make_pipeline(crops[0][3])

    if args.use_radius and args.num_outputs < 3:
        print("WARNING: --use-radius requires --num-outputs >= 3. Setting --num-outputs to 3.")
        args.num_outputs = 3

    lodestar = dl.LodeSTAR(
        n_transforms=4,
        num_outputs=args.num_outputs,
        optimizer=dl.Adam(lr=1e-3),
    ).build()

    # Build datasets for each crop and concatenate them
    datasets = [dt.pytorch.Dataset(_make_pipeline(c[3]), length=128, replace=False) for c in crops]
    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    # Use a safe number of workers (0 is safest to avoid multiprocessing issues with tensors here)
    dataloader = dl.DataLoader(
        dataset=combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Train LodeSTAR
    lodestar_trainer = dl.Trainer(accelerator="auto", max_epochs=args.epochs, log_every_n_steps=10)
    lodestar_trainer.fit(lodestar, dataloader)

    # Detect particles
    beta = 1.0 - args.alpha
    
    # Expected format for LodeSTAR input is Tensor of shape [N, C, H, W]
    # Current data shape: [N, H, W] -> unsqueeze(1) becomes [N, 1, H, W]
    data_tensor = torch.from_numpy(data).to(torch.float32).unsqueeze(1)

    print(len(data_tensor))

    print("Detecting objects across all frames...")
    all_detections = []
    
    # Put model into evaluation mode and disable gradient computation
    lodestar.eval()
    
    # Lightning Trainer automatically moves the model back to CPU after fit()
    # We must explicitly move it back to GPU for manual batched inference
    detect_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lodestar = lodestar.to(detect_device)

    def _collect_detections(batch_detections, out_list):
        """Move detections to CPU numpy and append to out_list."""
        if isinstance(batch_detections, (list, tuple)):
            for frame_detections in batch_detections:
                if isinstance(frame_detections, torch.Tensor):
                    out_list.append(frame_detections.detach().cpu().numpy())
                elif isinstance(frame_detections, (list, tuple)):
                    out_list.append([
                        d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d
                        for d in frame_detections
                    ])
                else:
                    out_list.append(frame_detections)
        else:
            if isinstance(batch_detections, torch.Tensor):
                batch_detections = batch_detections.detach().cpu().numpy()
                for frame_detections in batch_detections:
                    out_list.append(frame_detections)
            else:
                out_list.append(batch_detections)

    def _run_detect(frames_tensor, device):
        return lodestar.detect(
            frames_tensor.to(device),
            alpha=args.alpha,
            beta=beta,
            cutoff=args.cutoff,
            mode=args.detect_mode
        )

    with torch.inference_mode():
        # Process frames in batches
        for i in tqdm(range(0, len(data_tensor), args.detect_batch_size), desc="Detecting targets"):
            batch = data_tensor[i:i + args.detect_batch_size]

            try:
                batch_detections = _run_detect(batch, detect_device)
                if detect_device == 'cuda':
                    torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                # Fall back: process one frame at a time on GPU, then retry on CPU if still OOM
                print(f"\nOOM at batch {i // args.detect_batch_size}. Retrying frame-by-frame on GPU...")
                if detect_device == 'cuda':
                    torch.cuda.empty_cache()
                batch_detections = []
                for j in range(len(batch)):
                    single = batch[j:j+1]
                    try:
                        det = _run_detect(single, detect_device)
                        if detect_device == 'cuda':
                            torch.cuda.empty_cache()
                    except torch.OutOfMemoryError:
                        print(f"  Still OOM for frame {i + j}. Falling back to CPU...")
                        torch.cuda.empty_cache()
                        lodestar_cpu = lodestar.to('cpu')
                        det = _run_detect(single, 'cpu')
                        lodestar = lodestar_cpu.to(detect_device)
                    _collect_detections(det, batch_detections)
                all_detections.extend(batch_detections)
                continue

            _collect_detections(batch_detections, all_detections)

    
    def _nms(detections, min_dist):
        """Remove detections whose (y, x) centre is within min_dist pixels of a
        higher-scored detection.  Detections are expected to be sorted by
        descending score (det[2]) when a score channel is present, otherwise
        they are kept in order."""
        if min_dist <= 0 or len(detections) == 0:
            return detections
        arr = np.array(detections)          # shape (N, >=2)
        keep = []
        suppressed = np.zeros(len(arr), dtype=bool)
        for i in range(len(arr)):
            if suppressed[i]:
                continue
            keep.append(detections[i])
            yi, xi = arr[i, 0], arr[i, 1]
            for j in range(i + 1, len(arr)):
                if suppressed[j]:
                    continue
                dist = np.sqrt((arr[j, 0] - yi) ** 2 + (arr[j, 1] - xi) ** 2)
                if dist < min_dist:
                    suppressed[j] = True
        return keep

    # Apply NMS per frame before saving
    if args.nms_distance > 0:
        all_detections = [
            _nms(list(dets), args.nms_distance) for dets in all_detections
        ]

    total_dets = sum(len(d) for d in all_detections)
    print(f"Detections per frame: min={min(len(d) for d in all_detections)}  "
          f"max={max(len(d) for d in all_detections)}  "
          f"mean={total_dets / max(1, len(all_detections)):.1f}  "
          f"total={total_dets}")

    # Print radius statistics to help calibrate --radius-scale
    if args.use_radius and args.num_outputs >= 3:
        all_radii = [
            float(det[2])
            for frame_dets in all_detections
            for det in (frame_dets if not isinstance(frame_dets, np.ndarray) else [frame_dets])
            if hasattr(det, '__len__') and len(det) >= 3
        ]
        if all_radii:
            print(f"Raw radius channel stats (before scaling): "
                  f"min={min(all_radii):.3f}  max={max(all_radii):.3f}  "
                  f"mean={sum(all_radii)/len(all_radii):.3f}  "
                  f"→ box_px range with scale={args.radius_scale}: "
                  f"{min(abs(r)*args.radius_scale for r in all_radii):.1f} – "
                  f"{max(abs(r)*args.radius_scale for r in all_radii):.1f} px")

    # Effective minimum box size for --use-radius mode
    min_box_px = args.min_box_size if args.min_box_size > 0 else float(args.box_size)

    print("Saving YOLO labels...")
    for frame_idx, (frame_file, frame_detections) in enumerate(tqdm(zip(image_files, all_detections), total=len(image_files), desc="Processing files")):
        base_name = os.path.splitext(os.path.basename(frame_file))[0]
        txt_path = os.path.join(args.output_dir, f"{base_name}.txt")
        
        if args.plot:
            fig, ax = plt.subplots(1)
            ax.imshow(images[frame_idx], cmap='gray')
            
        with open(txt_path, 'w', encoding="utf-8") as f:
            for det in frame_detections:
                # lodestar forward: channel 0 = X (row = image y), channel 1 = Y (col = image x)
                # channel 2 (when num_outputs>=3) is the radius estimate
                y, x = det[0], det[1]

                if args.use_radius and len(det) >= 3:
                    raw_radius = float(det[2])
                    box_px = max(min_box_px, abs(raw_radius) * args.radius_scale)
                else:
                    box_px = float(args.box_size)

                # Normalize and clamp YOLO properties
                x_center = max(0.0, min(1.0, x / W))
                y_center = max(0.0, min(1.0, y / H))
                w = max(0.0, min(1.0, box_px / W))
                h = max(0.0, min(1.0, box_px / H))
                
                # Class 0 by default
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
                if args.plot:
                    # Draw bounding box (un-normalized coordinates for matplotlib)
                    rect = patches.Rectangle(
                        (x - box_px/2, y - box_px/2),
                        box_px, box_px,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    # Centre dot for small particles
                    ax.plot(x, y, '+', color='cyan', markersize=4, markeredgewidth=1)
        
        if args.plot:
            ax.axis('off')
            plot_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
    print(f"Finished auto-labeling. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
