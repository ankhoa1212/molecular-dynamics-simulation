# Data Setup — LodeSTAR Auto-Labeling

Unsupervised particle detection pipeline using [LodeSTAR](https://github.com/softmatterlab/DeepTrack2).
Detects particles in microscopy images and outputs YOLO-format `.txt` label files.

## Scripts

| Script | Purpose |
|---|---|
| `extract_frames.py` | Extract individual PNG frames from multi-page TIFF (or JPG) files |
| `crop_tool.py` | GUI tool to manually select crops from frames for training |
| `train_lodestar.py` | Train a LodeSTAR model on crops and save it for reuse |
| `label_images.py` | Run inference with a saved model to produce YOLO labels |
| `lodestar_autolabeler.py` | Batch label raw TIFF stacks or PNG frames with a saved model |
| `preview_augmentations.py` | Visualize how brightness, contrast, and noise affect your training crops |

For help on script usage:
```bash
python name_of_script.py --help
```
---

## Installation

```bash
pip install -r requirements.txt
```

---

## Recommended Workflow: Extract → Crop → Train → Label

### 1. Extract Frames from TIFF
LodeSTAR training and manual cropping work best with individual image files. Convert your raw TIFF stacks into PNG frames first:

```bash
python extract_frames.py video.tif frames/ --nth 5
```
This will save every 5th frame into the `frames/` directory.

### 2. Create Crops for Training
Use `crop_tool.py` to open your extracted PNG frames and draw bounding boxes around representative particles.

```bash
python crop_tool.py frames/
```
Crops are saved to a `crops/` subdirectory inside the frame folder. These will be used for training.

### 3. Train Autolabeler

```bash
python train_lodestar.py \
  --input-dir "frames/" \
  --model-path models/lodestar_model_15/
```

Accepts one or more `--input-dir` paths. Each directory is searched directly and inside its
`crops/` subdirectory. Pass multiple folders to train on a combined dataset:

```bash
python train_lodestar.py \
  --input-dir "trial_1_frames/" "trial_2_frames/" \
  --model-path models/lodestar_model_15/
```

Saves a model directory containing:
- `models/lodestar_model_15/model.pt` — model weights
- `models/lodestar_model_15/model.json` — architecture config + training parameters (see below)
- `models/lodestar_model_15/crops/` — a copy of the source images used to train this model

If `--model-path` is omitted, the model is saved as `model.pt` in a folder `lodestar_model_` + number inside `models/`.

Example `model.json`:
```json
{
  "n_transforms": 8,
  "num_outputs": 3,
  "training_params": {
    "epochs": 100,
    "batch_size": 8,
    "seed": 42,
    "source_crops": ["/path/to/frame1_crop.png", "..."]
  }
}
```

Training uses early stopping and is logged to MLflow automatically (see [MLflow](#mlflow) below).

### 4. Label Images (Test Inference)

```bash
python label_images.py \
  --input-dir frames/ \
  --model-path models/lodestar_model_15/ \
  --output-dir labels/
```

Writes one `.txt` YOLO label file per input image.


### 5. Run Autolabeler (Production)

To batch-label a folder of PNG frames using a trained model, run:

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_10/ \
  --input "/folder/path/of/tif/file" \
  --use-radius \
  --alpha 0.9 --cutoff 0.001 \
  --nms-distance 35
```

This will label all PNG frames in the specified directory using the provided model, with radius-based bounding boxes and the given detection parameters.

---

## Arguments

### `train_lodestar.py`

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | — | One or more directories of crop images (space-separated) |
| `--input-file` | — | One or more individual crop image paths |
| `--model-path` | auto (see above) | Where to save the `.pt` weights |
| `--epochs` | `100` | Maximum training epochs (early stopping usually fires earlier) |
| `--crop-size` | `64` | Crop size (px); images are centre-padded or centre-cropped to this |
| `--n-transforms` | `8` | Equivariance transforms (higher = more rotation-robust) |
| `--num-outputs` | `3` | `2` = (x,y); `3` = (x,y,radius) |
| `--batch-size` | `8` | Training batch size |
| `--num-workers` | `0` | DataLoader workers (0 is safest) |
| `--patience` | `15` | Early stopping: epochs with no improvement before stopping |
| `--min-delta` | `0.005` | Minimum loss decrease to count as improvement |
| `--dataset-length` | auto | Augmented samples per crop per epoch (auto-scaled by crop count) |
| `--seed` | `42` | Random seed for reproducibility |
| `--experiment` | `lodestar` | MLflow experiment name |
| `--run-name` | model filename stem | MLflow run name |
| `--mlflow-uri` | `sqlite:///mlflow.db` | MLflow tracking URI |
| `--brightness` | `-0.05 0.05` | Brightness offset range |
| `--contrast` | `0.25 1.0` | Contrast multiplier range |
| `--noise` | `0.001 0.01` | Gaussian noise range (sigma) |
| `--rotation` | `0 2π` | Rotation range (radians) |
| `--scale` | `0.8 1.2` | Scale jitter range |
| `--translate` | `-0.1 0.1` | Translation range (fraction of image size) |
| `--flip-lr` | `0.5` | Probability of left-right flip |
| `--flip-ud` | `0.5` | Probability of up-down flip |
| `--config` | — | Path to a JSON configuration file |

### `label_images.py`

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | — | Directory of input images |
| `--input-file` | — | Single input image |
| `--config` | — | Path to a JSON configuration file |
| `--model-path` | **required** | Path to saved `.pt` weights |
| `--output-dir` | `output_<input_folder_name>/` | Where to write YOLO `.txt` files |
| `--alpha` | `0.5` | Blend between equivariance score (0) and detection score (1) |
| `--cutoff` | `0.5` | Detection threshold (interpretation depends on `--detect-mode`) |
| `--detect-mode` | `ratio` | `ratio` / `quantile` / `constant` (see [Detection Modes](#detection-modes)) |
| `--nms-distance` | `0` | Min pixel distance between detections; 0 disables NMS |
| `--box-size` | `40` | Fixed bounding box size in pixels |
| `--use-radius` | off | Use per-detection radius from the model instead of `--box-size` |
| `--radius-scale` | `1.0` | Multiplier on raw radius output to convert to pixels |
| `--min-box-size` | `0` | Minimum box size when `--use-radius` is active |
| `--detect-batch-size` | `4` | Frames per GPU batch; lower to avoid OOM |
| `--plot` | off | Save `*_overlay.png` with bounding boxes drawn |

### `lodestar_autolabeler.py`

Batch-labels raw TIFF stacks or a folder of PNG frames using a saved model.
By default, output is written in a RoboFlow-compatible structure: `<name>_dataset/{images,labels}/` next to the input. Use `--output-dir` to redirect labels to a specific directory.

Either `--input` (TIFF search) or `--png-frames` must be provided.

| Argument | Default | Description |
|---|---|---|
| `--model` | **required** | Path to saved LodeSTAR model folder (or .pt file) |
| `--input` | — | Root directory to search for `.tif`/`.tiff` files recursively |
| `--png-frames` | — | Directory of PNG frames to label (alternative to `--input`) |
| `--output-dir` | `<name>_dataset/labels/` | Directory to write YOLO label files and overlays. For TIFF mode, each TIFF gets a sub-folder. |
| `--nth` | `5` | Save every nth frame from TIFF stacks |
| `--alpha` | `0.5` | Blend between equivariance score (0) and detection score (1) |
| `--cutoff` | `0.5` | Detection threshold (`ratio` mode: keep scores ≥ `cutoff × max`) |
| `--nms-distance` | `0` | Min pixel distance between detections; 0 disables NMS |
| `--box-size` | `40` | Fixed bounding box size in pixels |
| `--use-radius` | off | Use per-detection radius from the model instead of `--box-size` |
| `--radius-scale` | `1.0` | Multiplier on raw radius output to convert to pixels |
| `--min-box-size` | `0` | Minimum box size in pixels when `--use-radius` is active |
| `--detect-batch-size` | `4` | Frames per GPU batch |
| `--plot` | off | Save `*_overlay.png` with detections drawn |
| `--config` | — | Path to a JSON configuration file |
| `--num-workers` | `4` | DataLoader workers for prefetching |
| `--fp16` | off | Use 16-bit mixed precision (faster on GPU) |
| `--compile` | off | Use torch.compile for kernel optimization (PyTorch 2.0+) |

**Label TIFF stacks:**

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_15/ \
  --input data/raw_tiffs/ \
  --nth 5 \
  --cutoff 0.4 \
  --nms-distance 15 \
  --output-dir /mnt/results/labels \
  --plot
```

**Label a folder of PNG frames (with radius-based boxes):**

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_15/ \
  --png-frames data/frames/ \
  --output-dir /mnt/results/labels \
  --use-radius --radius-scale 2.0 \
  --alpha 0.9 --cutoff 0.001 \
  --nms-distance 35 \
  --plot
```

## Configuring with JSON

Most scripts in this pipeline support a `--config` flag to simplify command-line usage. This is especially useful for maintaining reproducible training runs or specific inference settings for different experiments.

### 1. Training Configuration
You can define all hyperparameters for `train_lodestar.py` in a JSON file. This is the recommended way to manage complex augmentation ranges.

```bash
python train_lodestar.py --config configs/default_lodestar.json --input-dir frames/
```

Example `configs/default_lodestar.json` (partial):
```json
{
  "n_transforms": 8,
  "num_outputs": 3,
  "training_params": {
    "epochs": 100,
    "batch_size": 8,
    "brightness": [-0.05, 0.05],
    "contrast": [0.25, 1.0],
    "noise": [0.001, 0.01]
  }
}
```

### 2. Autolabeling Configuration
For batch processing, you can save your model path and detection thresholds (`cutoff`, `alpha`, etc.) in a config file.

```bash
python lodestar_autolabeler.py --config configs/autolabel_2um_lodestar_model_15.json
```

We provide several pre-tuned configs in the `configs/` directory:
- `configs/autolabel_2um_lodestar_model_10.json`: Optimized for Model 10 on 2um particles.
- `configs/autolabel_2um_lodestar_model_15.json`: Optimized for Model 15 on 2um particles.

> **Tip:** CLI arguments always override JSON values. For example, `python lodestar_autolabeler.py --config configs/autolabel_2um.json --cutoff 0.01` will use the JSON settings but apply a different cutoff.

---

## Detection Modes

| Mode | Behavior |
|---|---|
| `ratio` | Keep detections with score ≥ `cutoff × max_score`. Good default for large images where background is variable. |
| `quantile` | Use the `cutoff` quantile of scores as the threshold. Stricter on dense images. |
| `constant` | Keep detections with score ≥ `cutoff` (absolute value). Use when scores are calibrated. |

> **Note:** All modes use the same underlying peak-finding algorithm. `ratio` with `--cutoff 0.3`
> is a good starting point. Adjust `--cutoff` up (fewer detections) or down (more) without retraining.

---

## Bounding Box Size

By default every detection gets a square box of `--box-size` pixels.

To use the model's own radius estimate instead:

```bash
python label_images.py \
  --model-path models/exp1.pt \
  --input-dir frames/ \
  --use-radius \
  --radius-scale 2.0 \
  --min-box-size 10
```

Run once with `--plot` to inspect the overlay and calibrate `--radius-scale`.

---

## Manual Verification & Correction

You can use the `crop_tool.py` to inspect and correct the autolabeler's output. 

**Using a configuration file:**
If you have an autolabeler config, you can use it to automatically open the generated images:
```bash
python crop_tool.py --config configs/autolabel_2um.json
```
This will automatically open the `images/` folder inside your specified `output_dir`.

**Manual Folder Mode:**
```bash
python crop_tool.py path/to/your/images_dataset/images
```

### `crop_tool.py`

| Argument | Default | Description |
|---|---|---|
| `folder` | — | Positional: folder containing images to browse |
| `--config` | — | Path to a JSON configuration file (autolabeler config supported) |

**Advanced Controls:**
*   **Edit Mode (E)**: Toggle to select and modify existing boxes.
*   **Multi-Select**: Hold **Control** while clicking to select multiple boxes.
*   **Batch Move**: Dragging any selected box moves the entire selection.
*   **Bulk Actions**: `Delete` and `Convert` (T) work on all selected boxes at once.

---

## MLflow

Training runs are automatically tracked with MLflow to help you compare different models and augmentation settings.

### Viewing Runs Locally
To start the MLflow dashboard and inspect your training history:

1. Navigate to the `data-setup/` directory.
2. Activate the virtual environment for data-setup:
   ```bash
   source .venv/bin/activate
   ```
3. Run the UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
4. Open [http://localhost:5000](http://localhost:5000) in your browser.

Each run records:
- **Parameters**: All training hyperparameters (epochs, crop size, batch size, etc.).
- **Metrics**: Loss curves per step (logged automatically by the Lightning trainer).
- **Artifacts**: A copy of the saved `.pt` weights, the `.json` config, and a copy of the **source crops** used for that specific run.

To track runs locally (now defaults to `sqlite:///mlflow.db`):

```bash
python train_lodestar.py \
  --input-dir frames/ \
  --model-path models/lodestar_model_15/ \
  --experiment particle-detection \
  --run-name trial-1
```

---

## Augmentation Preview

Before training, use `preview_augmentations.py` to tune your brightness, contrast, and noise settings. This script picks random particles from your crops and applies unique random augmentations to each subplot in a square-like grid.

```bash
python preview_augmentations.py models/lodestar_model_15/crops/ --count 25 --brightness -0.1 0.4 --contrast 0.1 0.5 --noise 0.01 0.03
```

| Argument | Default | Description |
|---|---|---|
| `--count` | `12` | Number of samples to generate in the grid |
| `--brightness` | `-0.15 0.15` | Range for brightness offset |
| `--contrast` | `0.4 1.6` | Range for contrast multiplier |
| `--noise` | `0.0 0.05` | Range for Gaussian noise (sigma) |
| `--size` | `64` | Crop size (px) |

---

## YOLO Label Format

Each `.txt` file contains one line per detected particle:

```
<class> <x_center> <y_center> <width> <height>
```

All values are normalised to `[0, 1]` relative to the image dimensions. `class` is
always `0` (single particle class).

---

## Tips

- Inspect detections with `--plot` before committing to full labeling runs.
- Set `--nms-distance` to roughly your expected particle diameter to suppress duplicate detections.
- If detections are too many / too few, adjust `--cutoff` in `label_images.py` without retraining.
- Use `--detect-mode ratio --cutoff 0.3` as a good starting point for crowded frames.
- On large microscopy frames (2048px+), inference is GPU-accelerated but peak-finding runs on CPU — this is normal. If it seems slow, lower `--detect-batch-size` to `1`.
- GPU OOM: reduce `--detect-batch-size`. The script falls back to CPU automatically if needed.
- Early stopping fires around epoch 30–50 for typical crop sets; `--epochs 100` is a safe cap.

