import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "processed_data"
IMAGES_EXT = (".jpg", ".jpeg", ".png")

# Check if data directory is already in YOLO format
# YOLO format: data/images and data/labels directories exist and contain files
# Check if data directory is already split into train, test, or validation directories
SPLIT_DIRS = ["train", "test", "validation"]
FOUND_SPLIT = any(os.path.isdir(os.path.join(DATA_DIR, d)) for d in SPLIT_DIRS)

if FOUND_SPLIT:
    print("Data directory already contains train/test/validation splits. No processing needed.")
    exit(0)
# Check if already in YOLO format
PROCESSING_NEEDED = True
for split_name in SPLIT_DIRS:
    images_dir = os.path.join(DATA_DIR, split_name, "images")
    labels_dir = os.path.join(DATA_DIR, split_name, "labels")
    has_images_dir = os.path.isdir(images_dir)
    has_labels_dir = os.path.isdir(labels_dir)
    has_images = len(glob(os.path.join(images_dir, "*"))) > 0
    has_labels = len(glob(os.path.join(labels_dir, "*"))) > 0
    if not (has_images_dir and has_labels_dir and has_images and has_labels):
        PROCESSING_NEEDED = False
        break
if PROCESSING_NEEDED:
    print("Data directory already in YOLO format. No processing needed.")
    exit(0)
# Create output directories
for split in ["train", "validation"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Collect image files
IMAGE_FILES = []
for ext in IMAGES_EXT:
    IMAGE_FILES.extend(glob(os.path.join(DATA_DIR, f"*{ext}")))

# Split into train/validation (80/20)
TRAIN_IMGS, VALIDATION_IMGS = train_test_split(IMAGE_FILES, test_size=0.2, random_state=42)


def copy_files(img_list, split):
    for img_path in img_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(DATA_DIR, base + ".txt")
        # Copy image
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, split, "images", os.path.basename(img_path)))
        # Copy label if exists
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(OUTPUT_DIR, split, "labels", base + ".txt"))


copy_files(TRAIN_IMGS, "train")
copy_files(VALIDATION_IMGS, "validation")

print("Preprocessing complete. Data is ready for YOLOv12 fine-tuning.")
