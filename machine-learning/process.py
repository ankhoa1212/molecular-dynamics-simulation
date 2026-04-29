import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = 'data'
OUTPUT_DIR = 'processed_data'
IMAGES_EXT = ('.jpg', '.jpeg', '.png')

# Check if data directory is already in YOLO format
# YOLO format: data/images and data/labels directories exist and contain files
# Check if data directory is already split into train, test, or validation directories
split_dirs = ['train', 'test', 'validation']
found_split = any(os.path.isdir(os.path.join(DATA_DIR, d)) for d in split_dirs)

if found_split:
    print("Data directory already contains train/test/validation splits. No processing needed.")
    exit(0)
# Check if already in YOLO format
processing_needed = True
for dir in split_dirs:
    if not (
        os.path.isdir(os.path.join(DATA_DIR, dir, 'images')) and
        os.path.isdir(os.path.join(DATA_DIR, dir, 'labels')) and
        len(glob(os.path.join(DATA_DIR, dir, 'images', '*'))) > 0 and
        len(glob(os.path.join(DATA_DIR, dir, 'labels', '*'))) > 0
    ):
        processing_needed = False
        break
if not processing_needed:
    print("Data directory already in YOLO format. No processing needed.")
    exit(0)
# Create output directories
for split in ['train', 'validation']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# Collect image files
image_files = []
for ext in IMAGES_EXT:
    image_files.extend(glob(os.path.join(DATA_DIR, f'*{ext}')))

# Split into train/validation (80/20)
train_imgs, validation_imgs = train_test_split(image_files, test_size=0.2, random_state=42)

def copy_files(img_list, split):
    for img_path in img_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(DATA_DIR, base + '.txt')
        # Copy image
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, split, 'images', os.path.basename(img_path)))
        # Copy label if exists
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(OUTPUT_DIR, split, 'labels', base + '.txt'))

copy_files(train_imgs, 'train')
copy_files(validation_imgs, 'validation')

print("Preprocessing complete. Data is ready for YOLOv12 fine-tuning.")