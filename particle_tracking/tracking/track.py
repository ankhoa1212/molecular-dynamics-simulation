import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageSequence


# Optional imports handled inside to allow running with different environments
def get_rfdetr_model(variant, checkpoint, device):
    # Add rf-detr to path to allow importing its modules
    # Updated for new structure: pipeline/tracking/track.py -> pipeline/models/rf-detr
    rf_detr_path = Path(__file__).parent.parent / "models" / "rf-detr"
    if str(rf_detr_path) not in sys.path:
        sys.path.append(str(rf_detr_path))

    try:
        import torch

        if variant == "base":
            from rfdetr import RFDETRBase

            model = RFDETRBase(pretrain_weights=checkpoint)
        elif variant == "large":
            from rfdetr import RFDETRLarge

            model = RFDETRLarge(pretrain_weights=checkpoint)
        else:
            raise ValueError(f"Unknown RF-DETR variant: {variant}")
        model.to(device)
        model.eval()
        return model
    except ImportError:
        print("Error: 'rfdetr' library not found. Please run 'uv sync' in the rf-detr directory.")
        sys.exit(1)


def get_yolo_model(checkpoint):
    try:
        from ultralytics import YOLO

        return YOLO(checkpoint)
    except ImportError:
        print(
            "Error: 'ultralytics' library not found. Please install it with 'pip install ultralytics'."
        )
        sys.exit(1)


def get_lodestar_model(checkpoint, device):
    try:
        import deeplay as dl
        import torch
        import json

        # Look for companion JSON
        config_path = Path(checkpoint).with_suffix(".json")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            n_transforms = config.get("n_transforms", 8)
            num_outputs = config.get("num_outputs", 3)
        else:
            n_transforms, num_outputs = 8, 3

        model = dl.LodeSTAR(n_transforms=n_transforms, num_outputs=num_outputs).build()
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        model.eval()
        return model
    except ImportError:
        print("Error: 'deeplay' or 'torch' not found. Please install them to use LodeSTAR.")
        sys.exit(1)


def detect_lodestar(model, frame, threshold, device, box_size=40):
    import torch
    import supervision as sv

    # Normalize frame as in label_images.py
    frame_f = frame.astype(np.float32)
    if frame.ndim == 3:
        frame_f = np.mean(frame_f, axis=2)  # Convert to grayscale

    f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
    frame_norm = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min

    tensor = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        detections_raw = model.detect(tensor, alpha=0.5, beta=0.5, cutoff=threshold, mode="ratio")

    if isinstance(detections_raw, list):
        detections_raw = detections_raw[0]

    if detections_raw is None or len(detections_raw) == 0:
        return sv.Detections.empty()

    # detections_raw is [y, x, (radius)]
    xyxy = []
    confidences = []
    for det in detections_raw:
        y, x = det[0], det[1]
        r = abs(det[2]) if len(det) >= 3 else box_size / 2
        xyxy.append([x - r, y - r, x + r, y + r])
        confidences.append(
            1.0
        )  # LodeSTAR doesn't provide explicit confidence in the same way YOLO does

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.zeros(len(xyxy), dtype=int),
    )


def load_frames(input_path):
    """Load frames from a video file, a directory of images, or a multi-page TIFF."""
    input_path = Path(input_path)
    frames = []

    if input_path.is_dir():
        # Directory of images
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted([f for f in input_path.glob("*.*") if f.suffix.lower() in valid_exts])
        for f in files:
            frames.append(np.array(Image.open(f).convert("RGB")))
    elif input_path.suffix.lower() in {".tif", ".tiff"}:
        # Multi-page TIFF
        img = Image.open(input_path)
        for frame in ImageSequence.Iterator(img):
            frames.append(np.array(frame.convert("RGB")))
    elif input_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        # Single image file
        frames.append(np.array(Image.open(input_path).convert("RGB")))
    else:
        # Video file
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


def main():
    parser = argparse.ArgumentParser(
        description="Particle Tracking with RF-DETR, YOLO, or LodeSTAR"
    )
    parser.add_argument("--model-type", choices=["rf-detr", "yolo", "lodestar"], required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model weights (.pth, .pt, or .ckpt)"
    )
    parser.add_argument("--variant", default="base", help="RF-DETR variant (base/large)")
    parser.add_argument("--input", required=True, help="Path to video or folder of frames")
    parser.add_argument(
        "--output-dir", default="tracking_results", help="Directory to save results"
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection threshold")
    parser.add_argument("--device", default="0", help="Device to run inference on (e.g. 0 or cpu)")

    # Tracking options
    parser.add_argument(
        "--tracker",
        choices=["bytetrack", "trackpy"],
        default="bytetrack",
        help="Tracking algorithm",
    )
    parser.add_argument(
        "--search-range", type=float, default=10.0, help="Trackpy: Max distance between frames"
    )
    parser.add_argument(
        "--memory", type=int, default=3, help="Trackpy: Number of frames a particle can be missing"
    )
    parser.add_argument(
        "--stub-filter", type=int, default=0, help="Trackpy: Minimum number of frames for a track"
    )

    parser.add_argument("--save-video", action="store_true", help="Save an annotated video")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output video")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frames from {args.input}...")
    frames = load_frames(args.input)
    if not frames:
        print("No frames found. Exiting.")
        return

    print(f"Found {len(frames)} frames. Initializing {args.model_type} model...")

    # Import supervision and trackpy
    try:
        import supervision as sv
    except ImportError:
        print("Error: 'supervision' library not found. Please install it.")
        sys.exit(1)

    if args.tracker == "trackpy":
        try:
            import trackpy as tp
        except ImportError:
            print("Error: 'trackpy' library not found. Please install it.")
            sys.exit(1)

    # Initialize model
    if args.model_type == "rf-detr":
        model = get_rfdetr_model(args.variant, args.checkpoint, args.device)
    elif args.model_type == "yolo":
        model = get_yolo_model(args.checkpoint)
    elif args.model_type == "lodestar":
        model = get_lodestar_model(args.checkpoint, args.device)

    # Tracking State
    all_detections = []
    tracking_data = []

    # 1. Detection Phase
    for i, frame in enumerate(tqdm(frames, desc="Detecting")):
        if args.model_type == "rf-detr":
            detections = model.predict(frame, threshold=args.threshold)
        elif args.model_type == "yolo":
            results = model.predict(frame, conf=args.threshold, device=args.device, verbose=False)[
                0
            ]
            detections = sv.Detections.from_ultralytics(results)
        elif args.model_type == "lodestar":
            detections = detect_lodestar(model, frame, args.threshold, args.device)

        all_detections.append(detections)

        # Log raw detections
        for j in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[j]
            tracking_data.append(
                {
                    "frame": i,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": detections.confidence[j] if detections.confidence is not None else 1.0,
                    "tracker_id": -1,
                }
            )

    # 2. Tracking Phase
    df = pd.DataFrame(tracking_data)

    if args.tracker == "bytetrack":
        print("Applying ByteTrack (online)...")
        tracker = sv.ByteTrack()
        tracked_frames_detections = []

        # Reset tracking data to populate with tracker IDs
        tracking_data = []

        for i, detections in enumerate(tqdm(all_detections, desc="Tracking")):
            detections = tracker.update_with_detections(detections)
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

    elif args.tracker == "trackpy":
        print("Applying Trackpy (offline)...")
        if len(df) > 0:
            # Trackpy link_df
            df = tp.link_df(df, search_range=args.search_range, memory=args.memory)

            if args.stub_filter > 0:
                df = tp.filter_stubs(df, args.stub_filter)

            df = df.rename(columns={"particle": "track_id"})
            tracking_data = df.to_dict("records")
        else:
            print("No detections to track.")
            tracking_data = []

    # 3. Visualization Phase
    if args.save_video:
        print("Annotating video...")
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        trace_annotator = sv.TraceAnnotator()

        annotated_frames = []
        df_tracked = pd.DataFrame(tracking_data)

        for i, frame in enumerate(tqdm(frames, desc="Visualizing")):
            if not df_tracked.empty and "track_id" in df_tracked.columns:
                frame_df = df_tracked[df_tracked["frame"] == i]

                if not frame_df.empty:
                    # Convert back to supervision detections for easy annotation
                    xyxy = []
                    tracker_ids = []
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

            labels = (
                [f"#{id}" for id in detections.tracker_id]
                if detections.tracker_id is not None
                else []
            )
            annotated_frame = frame.copy()
            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
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
        out = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        for f in annotated_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved annotated video to {video_path}")

    # 4. Save Results
    df_final = pd.DataFrame(tracking_data)
    csv_path = output_dir / "tracks.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"Saved tracking data to {csv_path}")


if __name__ == "__main__":
    main()
