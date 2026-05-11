import os
import shutil
import yaml
from glob import glob
from sklearn.model_selection import train_test_split


def run(config_path="config.yaml"):
    # Load config if exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Paths
    data_dir = config.get("data_dir", "data")
    output_dir = config.get("output_dir", "processed_data")
    images_ext = (".jpg", ".jpeg", ".png")

    print(f"Processing data from {data_dir} to {output_dir}...")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return

    # Create output directories
    for split_name in ["train", "validation"]:
        os.makedirs(os.path.join(output_dir, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_name, "labels"), exist_ok=True)

    # Collect image files
    image_files = []
    for ext in images_ext:
        image_files.extend(glob(os.path.join(data_dir, f"*{ext}")))

    if not image_files:
        print(f"No images found in {data_dir}")
        return

    # Split into train/validation (80/20)
    train_imgs, val_imgs = train_test_split(image_files, test_size=0.2, random_state=42)

    def copy_files(img_list, split_name):
        count = 0
        for img_path in img_list:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(data_dir, base + ".txt")

            # Copy image
            shutil.copy(
                img_path, os.path.join(output_dir, split_name, "images", os.path.basename(img_path))
            )
            # Copy label if exists
            if os.path.exists(label_path):
                shutil.copy(
                    label_path, os.path.join(output_dir, split_name, "labels", base + ".txt")
                )
                count += 1
        return count

    train_labels = copy_files(train_imgs, "train")
    val_labels = copy_files(val_imgs, "validation")

    print(f"Copied {len(train_imgs)} train images ({train_labels} labels)")
    print(f"Copied {len(val_imgs)} val images ({val_labels} labels)")

    # Generate data.yaml
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "validation/images",
        "nc": config.get("nc", 1),
        "names": config.get("names", ["particle"]),
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"Generated {yaml_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    run()
