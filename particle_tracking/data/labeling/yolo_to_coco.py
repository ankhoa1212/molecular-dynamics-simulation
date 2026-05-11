#!/usr/bin/env python3
"""
Convert YOLO format detection data to COCO JSON format.
Specifically designed for compatibility with RF-DETR training pipelines.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height


def convert_yolo_to_coco(images_dir, labels_dir, output_file, class_names):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    categories = []
    for i, name in enumerate(class_names):
        categories.append({"id": i, "name": name, "supercategory": "none"})

    coco_data = {
        "info": {
            "description": "Converted from YOLO format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    label_files = list(labels_dir.glob("*.txt"))

    # In case labels and images are in the same folder or different structures
    # We prioritize matching images to existing labels

    ann_id = 0
    img_id = 0

    print(f"Found {len(label_files)} label files in {labels_dir}")

    for label_path in tqdm(label_files, desc="Converting"):
        # Find corresponding image
        image_path = None
        for ext in image_extensions:
            potential_path = images_dir / f"{label_path.stem}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break

        if not image_path:
            continue

        try:
            width, height = get_image_size(image_path)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue

        coco_data["images"].append(
            {
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": image_path.name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            # Convert to COCO (absolute x_min, y_min, width, height)
            w_abs = w_norm * width
            h_abs = h_norm * height
            x_min_abs = (x_center_norm - w_norm / 2) * width
            y_min_abs = (y_center_norm - h_norm / 2) * height

            # Clamp values to image dimensions
            x_min_abs = max(0, min(x_min_abs, width))
            y_min_abs = max(0, min(y_min_abs, height))
            w_abs = min(w_abs, width - x_min_abs)
            h_abs = min(h_abs, height - y_min_abs)

            coco_data["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": [],
                    "area": w_abs * h_abs,
                    "bbox": [x_min_abs, y_min_abs, w_abs, h_abs],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"\nConversion complete!")
    print(f"Total images: {img_id}")
    print(f"Total annotations: {ann_id}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO labels to COCO format.")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--labels-dir", type=str, required=True, help="Path to labels directory")
    parser.add_argument(
        "--output", type=str, default="annotations.json", help="Path to output COCO JSON file"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="particle",
        help="Comma-separated list of class names or path to classes.txt",
    )

    args = parser.parse_args()

    # Handle classes
    if os.path.exists(args.classes):
        with open(args.classes, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = [c.strip() for c in args.classes.split(",")]

    convert_yolo_to_coco(args.images_dir, args.labels_dir, args.output, class_names)


if __name__ == "__main__":
    main()
