import argparse
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageSequence

SCRIPT_DIR = Path(__file__).parent

# RF-DETR variant name → class name in the rfdetr package
RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",  # kept for backward compatibility
}


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("Warning: 'pyyaml' not installed — config file ignored. Run 'pip install pyyaml'.")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg, *keys, default=None):
    """Walk a nested dict by keys, returning default if any key is missing."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def resolve_path(p):
    """Resolve a path relative to particle-tracking/ if not already absolute."""
    p = Path(p)
    return p if p.is_absolute() else SCRIPT_DIR / p


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def get_rfdetr_model(variant, checkpoint, device):
    """Load RF-DETR from the local venv in models/rf_detr/."""
    rf_detr_venv = SCRIPT_DIR / "models" / "rf_detr" / ".venv"
    site_packages = list(rf_detr_venv.glob("lib/python*/site-packages"))
    if site_packages and str(site_packages[0]) not in sys.path:
        sys.path.insert(0, str(site_packages[0]))

    try:
        import rfdetr as _rfdetr

        cls_name = RFDETR_VARIANTS.get(variant)
        if cls_name is None:
            print(
                f"Error: unknown RF-DETR variant '{variant}'. Choose from: {', '.join(RFDETR_VARIANTS)}"
            )
            sys.exit(1)

        cls = getattr(_rfdetr, cls_name, None)
        if cls is None:
            print(f"Error: '{cls_name}' not found in installed rfdetr package.")
            sys.exit(1)

        model = cls(pretrain_weights=str(checkpoint))
        model.to(device)
        model.eval()
        return model
    except ImportError:
        print("Error: 'rfdetr' not found. Run 'uv sync' inside models/rf_detr/.")
        sys.exit(1)


def get_yolo_model(checkpoint):
    try:
        from ultralytics import YOLO

        return YOLO(str(checkpoint))
    except ImportError:
        print("Error: 'ultralytics' not found. Run 'pip install ultralytics'.")
        sys.exit(1)


def get_lodestar_model(checkpoint, device):
    try:
        import deeplay as dl
        import torch
        import json

        config_path = Path(checkpoint).with_suffix(".json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            n_transforms = config.get("n_transforms", 8)
            num_outputs = config.get("num_outputs", 3)
        else:
            n_transforms, num_outputs = 8, 3

        model = dl.LodeSTAR(n_transforms=n_transforms, num_outputs=num_outputs).build()
        model.load_state_dict(torch.load(str(checkpoint), map_location=device))
        model.to(device)
        model.eval()
        return model
    except ImportError:
        print("Error: 'deeplay' or 'torch' not found. Run 'pip install deeplay deeptrack'.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_lodestar(model, frame, threshold, device, box_size=40):
    import torch
    import supervision as sv

    frame_f = frame.astype(np.float32)
    if frame.ndim == 3:
        frame_f = np.mean(frame_f, axis=2)

    f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
    frame_norm = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min

    tensor = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        detections_raw = model.detect(tensor, alpha=0.5, beta=0.5, cutoff=threshold, mode="ratio")

    if isinstance(detections_raw, list):
        detections_raw = detections_raw[0]

    if detections_raw is None or len(detections_raw) == 0:
        return sv.Detections.empty()

    xyxy, confidences = [], []
    for det in detections_raw:
        y, x = det[0], det[1]
        r = abs(det[2]) if len(det) >= 3 else box_size / 2
        xyxy.append([x - r, y - r, x + r, y + r])
        confidences.append(1.0)

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.zeros(len(xyxy), dtype=int),
    )


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def load_frames(input_path):
    """Load frames from a video file, image directory, or multi-page TIFF."""
    input_path = Path(input_path)
    frames = []

    if input_path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted([f for f in input_path.glob("*.*") if f.suffix.lower() in valid_exts])
        for f in files:
            frames.append(np.array(Image.open(f).convert("RGB")))
    elif input_path.suffix.lower() in {".tif", ".tiff"}:
        img = Image.open(input_path)
        for frame in ImageSequence.Iterator(img):
            frames.append(np.array(frame.convert("RGB")))
    elif input_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        frames.append(np.array(Image.open(input_path).convert("RGB")))
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Particle Tracking with RF-DETR, YOLOv12, or LodeSTAR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Path to YAML config file",
    )
    # Model
    parser.add_argument("--model-type", choices=["rf-detr", "yolo", "lodestar"])
    parser.add_argument("--checkpoint", help="Path to model weights (.pth, .pt, or .ckpt)")
    parser.add_argument("--variant", choices=list(RFDETR_VARIANTS), help="RF-DETR model size")
    parser.add_argument("--device", help="Inference device (e.g. 0 or cpu)")
    parser.add_argument("--threshold", type=float, help="Detection confidence threshold")
    # I/O
    parser.add_argument("--input", help="Path to video, image folder, or TIFF stack")
    parser.add_argument("--output-dir", help="Directory to save results")
    # Tracking
    parser.add_argument("--tracker", choices=["trackpy", "bytetrack"])
    parser.add_argument("--search-range", type=float, help="Trackpy: max pixel distance per frame")
    parser.add_argument("--memory", type=int, help="Trackpy: frames a particle may be missing")
    parser.add_argument("--stub-filter", type=int, help="Trackpy: min track length to keep")
    # Video output
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=int, help="FPS for output video")

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Resolve final values: CLI arg → config → built-in default
    model_type = args.model_type or cfg_get(cfg, "model", "type", default="rf-detr")
    checkpoint = args.checkpoint or cfg_get(
        cfg, "model", "checkpoint", default="models/rf_detr/checkpoints/checkpoint_best_ema.pth"
    )
    variant = args.variant or cfg_get(cfg, "model", "variant", default="large")
    device = args.device or cfg_get(cfg, "model", "device", default="0")
    threshold = args.threshold or cfg_get(cfg, "detection", "threshold", default=0.25)
    input_path = args.input or cfg_get(cfg, "input")
    output_dir = Path(
        args.output_dir
        or cfg_get(cfg, "output", "dir", default="evaluation/results/tracking_output")
    )
    tracker = args.tracker or cfg_get(cfg, "tracking", "tracker", default="trackpy")
    search_range = (
        args.search_range
        if args.search_range is not None
        else cfg_get(cfg, "tracking", "search_range", default=10.0)
    )
    memory = (
        args.memory if args.memory is not None else cfg_get(cfg, "tracking", "memory", default=3)
    )
    stub_filter = (
        args.stub_filter
        if args.stub_filter is not None
        else cfg_get(cfg, "tracking", "stub_filter", default=5)
    )
    save_video = args.save_video or cfg_get(cfg, "output", "save_video", default=False)
    fps = args.fps or cfg_get(cfg, "output", "fps", default=30)

    if input_path is None:
        parser.error("--input is required (or set 'input' in config.yaml)")

    checkpoint = resolve_path(checkpoint)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:    {args.config}")
    print(f"Model:     {model_type} ({checkpoint})")
    print(f"Tracker:   {tracker}")
    print(f"Input:     {input_path}")
    print(f"Output:    {output_dir}")

    print(f"\nLoading frames from {input_path}...")
    frames = load_frames(input_path)
    if not frames:
        print("No frames found. Exiting.")
        return
    print(f"Found {len(frames)} frames. Initializing {model_type} model...")

    try:
        import supervision as sv
    except ImportError:
        print("Error: 'supervision' not found. Run 'pip install supervision'.")
        sys.exit(1)

    if tracker == "trackpy":
        try:
            import trackpy as tp
        except ImportError:
            print("Error: 'trackpy' not found. Run 'pip install trackpy'.")
            sys.exit(1)

    # Initialize detection model
    if model_type == "rf-detr":
        model = get_rfdetr_model(variant, checkpoint, device)
    elif model_type == "yolo":
        model = get_yolo_model(checkpoint)
    elif model_type == "lodestar":
        model = get_lodestar_model(checkpoint, device)

    # 1. Detection phase
    all_detections = []
    raw_tracking_data = []

    for i, frame in enumerate(tqdm(frames, desc="Detecting")):
        if model_type == "rf-detr":
            detections = model.predict(frame, threshold=threshold)
        elif model_type == "yolo":
            results = model.predict(frame, conf=threshold, device=device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
        elif model_type == "lodestar":
            detections = detect_lodestar(model, frame, threshold, device)

        all_detections.append(detections)

        for j in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[j]
            raw_tracking_data.append(
                {
                    "frame": i,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": detections.confidence[j] if detections.confidence is not None else 1.0,
                }
            )

    # 2. Tracking phase
    df = pd.DataFrame(raw_tracking_data)
    tracking_data = []

    if tracker == "trackpy":
        print("Applying Trackpy (offline)...")
        if not df.empty:
            df = tp.link_df(df, search_range=search_range, memory=memory)
            if stub_filter > 0:
                df = tp.filter_stubs(df, stub_filter)
            df = df.rename(columns={"particle": "track_id"})
            tracking_data = df.to_dict("records")
        else:
            print("No detections to track.")

    elif tracker == "bytetrack":
        print("Applying ByteTrack (online)...")
        byte_tracker = sv.ByteTrack()
        tracked_frames_detections = []

        for i, detections in enumerate(tqdm(all_detections, desc="Tracking")):
            detections = byte_tracker.update_with_detections(detections)
            tracked_frames_detections.append(detections)

            if detections.tracker_id is not None:
                for j in range(len(detections.tracker_id)):
                    x1, y1, x2, y2 = detections.xyxy[j]
                    tracking_data.append(
                        {
                            "frame": i,
                            "track_id": int(detections.tracker_id[j]),
                            "x": (x1 + x2) / 2,
                            "y": (y1 + y2) / 2,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "conf": (
                                detections.confidence[j]
                                if detections.confidence is not None
                                else 1.0
                            ),
                        }
                    )
            else:
                tracked_frames_detections[-1] = sv.Detections.empty()

    # 3. Visualization phase
    if save_video:
        print("Annotating video...")
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        trace_annotator = sv.TraceAnnotator()
        df_tracked = pd.DataFrame(tracking_data)
        annotated_frames = []

        for i, frame in enumerate(tqdm(frames, desc="Visualizing")):
            if not df_tracked.empty and "track_id" in df_tracked.columns:
                frame_df = df_tracked[df_tracked["frame"] == i]
                if not frame_df.empty:
                    xyxy, tracker_ids = [], []
                    for _, row in frame_df.iterrows():
                        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                        xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                        tracker_ids.append(int(row["track_id"]))
                    detections = sv.Detections(
                        xyxy=np.array(xyxy, dtype=np.float32),
                        tracker_id=np.array(tracker_ids, dtype=int),
                        class_id=np.zeros(len(xyxy), dtype=int),
                    )
                else:
                    detections = sv.Detections.empty()
            else:
                detections = sv.Detections.empty()

            annotated_frame = frame.copy()
            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                labels = [f"#{tid}" for tid in detections.tracker_id]
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            annotated_frames.append(annotated_frame)

        video_path = output_dir / "tracking_visualization.mp4"
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        for f in annotated_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved annotated video to {video_path}")

    # 4. Save results
    df_final = pd.DataFrame(tracking_data)
    csv_path = output_dir / "tracks.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"Saved tracking data to {csv_path}")


if __name__ == "__main__":
    main()
