# Molecular Dynamics Simulation & Particle Tracking

End-to-end pipeline for running molecular dynamics simulations, auto-labeling microscopy data with [LodeSTAR](https://github.com/softmatterlab/DeepTrack2), training a particle detection model, and tracking particles across frames.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Components](#components)
  - [1. Simulation](#1-simulation-lammps-scripts)
  - [2. Auto-Labeling with LodeSTAR](#2-auto-labeling-with-lodestar-data-setup)
  - [3. Particle Detection — RF-DETR](#3-particle-detection--rf-detr-rf-detr)
  - [4. Particle Tracking](#4-particle-tracking-particle-tracking)
- [Full Pipeline Overview](#full-pipeline-overview)
- [Contributing](#contributing)
- [Resources](#resources)

## Repository Structure

```
molecular-dynamics-simulation/
├── lammps-scripts/          # LAMMPS simulation scripts and analysis tools
├── data-setup/              # LodeSTAR auto-labeling pipeline (generates YOLO labels)
├── rf-detr/                 # RF-DETR particle detection model (training & evaluation)
├── yolov12/                 # YOLOv12 particle detection model (alternative)
└── particle-tracking/       # Particle tracking using a trained detection model
    ├── track.py             # Unified tracker (RF-DETR, YOLOv12, or LodeSTAR)
    ├── config.yaml          # Tracking configuration (model, input, output, tracker)
    ├── models/
    │   ├── rf_detr/         # RF-DETR weights and local venv
    │   └── yolov12/         # YOLOv12 weights
    ├── data/raw/            # Raw input TIFF files
    └── evaluation/results/  # Tracking outputs (tracks.csv, annotated video)
```

---

## Components

### 1. Simulation (`lammps-scripts/`)

Runs LAMMPS molecular dynamics simulations and analyzes results.

**Setup:** [Install and build LAMMPS](https://docs.lammps.org/Install.html), then add the executable to `PATH`:

```bash
export PATH=/path/to/lammps/bin:$PATH
```

**Run a simulation:**

```bash
cd lammps-scripts
python3 run.py --input continuous_force.in --output results --molecules 1000
# or via config file:
python3 run.py --config config/continuous_force_test.json
```

**Analyze results:**

```bash
python3 velocity_graph.py results/    # velocity distribution plots
python3 temp_graph.py results/        # temperature vs. time
python3 phase_diagram.py results/     # phase diagram
python3 hexatic_order_analysis.py     # hexatic order parameter
```

**Install dependencies:**

```bash
pip install -r lammps-scripts/requirements.txt
```

---

### 2. Auto-Labeling with LodeSTAR (`data-setup/`)

Unsupervised particle detection pipeline. Trains a LodeSTAR model on particle crops and uses it to produce YOLO-format `.txt` label files for downstream detection model training.

**Install dependencies:**

```bash
cd data-setup
pip install -r requirements.txt
```

**Workflow: Extract → Crop → Train → Label**

**Step 1 — Extract frames from TIFF:**

```bash
python extract_frames.py video.tif frames/ --nth 5
```

**Step 2 — Create crops for training (GUI tool):**

```bash
python crop_tool.py frames/
```

**Step 3 — Train LodeSTAR model:**

```bash
python train_lodestar.py \
  --input-dir frames/ \
  --model-path models/lodestar_model_15/
```

Training is logged to MLflow automatically. View runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

**Step 4 — Batch-label images:**

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_15/ \
  --input data/raw_tiffs/ \
  --use-radius \
  --alpha 0.9 --cutoff 0.001 \
  --nms-distance 35 \
  --plot
```

Outputs YOLO `.txt` label files alongside images in a RoboFlow-compatible directory structure.

Pre-tuned configs are available in `data-setup/configs/` for different particle sizes. Pass `--config configs/autolabel_2um_lodestar_model_15.json` to use one.

See [`data-setup/README.md`](data-setup/README.md) for full argument reference and configuration details.

---

### 3. Particle Detection — RF-DETR (`rf-detr/`)

Trains and evaluates an [RF-DETR](https://github.com/roboflow/rf-detr) transformer-based object detector on the labeled data produced by the auto-labeling step. Experiment tracking via MLflow.

**Requirements:** Python 3.11, [uv](https://docs.astral.sh/uv/), CUDA GPU recommended.

**Setup:**

```bash
cd rf-detr
uv sync
```

**Dataset format:** A directory with `images/` and a single `annotations.json` in COCO format. Edit `config.yaml` to set the dataset path and assign experiment names to train/val/test splits.

**Train:**

```bash
uv run python train.py --config config.yaml
```

**Evaluate:**

```bash
# Most recent checkpoint
uv run python evaluate.py --config config.yaml --batch-size 16

# Specific MLflow run
uv run python evaluate.py --config config.yaml --run-id <run-id> --batch-size 16
```

**View MLflow results:**

```bash
uv run mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow.db
# Open http://127.0.0.1:5000
```

**Run tests:**

```bash
uv run pytest tests/ -v
```

See [`rf-detr/README.md`](rf-detr/README.md) for full configuration options.

---

### 4. Particle Tracking (`particle-tracking/`)

Runs a trained detection model (RF-DETR, YOLOv12, or LodeSTAR) on video or image sequences and links detections into particle tracks.

**Input:** video file, folder of PNG/TIFF frames, or multi-page TIFF stack.

**Output:** `tracks.csv` with per-frame `(track_id, x, y, w, h, conf)` and optionally an annotated `.mp4`.

**Setup:**

The RF-DETR backend uses its own virtualenv. Install it once:

```bash
cd particle-tracking/models/rf_detr
uv sync
```

For all other dependencies (YOLO, LodeSTAR, supervision, trackpy):

```bash
pip install pyyaml opencv-python numpy pandas tifffile pillow tqdm supervision trackpy ultralytics deeplay deeptrack
```

**Configuration:**

Edit `particle-tracking/config.yaml` to set your model, input, and tracking parameters. All paths in the config are relative to the `particle-tracking/` directory.

```yaml
input: data/raw/your_video.tif

model:
  type: rf-detr          # rf-detr | yolo | lodestar
  checkpoint: models/rf_detr/checkpoints/checkpoint_best_ema.pth
  variant: large         # rf-detr only: nano | small | medium | large
  device: "0"

detection:
  threshold: 0.25

tracking:
  tracker: trackpy       # trackpy (default) | bytetrack
  search_range: 10.0
  memory: 3
  stub_filter: 5

output:
  dir: evaluation/results/tracking_output
  save_video: false
  fps: 30
```

**Usage:**

Run with defaults from the config file:

```bash
cd particle-tracking
python track.py
```

Override any config value on the command line:

```bash
# RF-DETR with a specific checkpoint
python track.py \
  --model-type rf-detr \
  --checkpoint models/rf_detr/checkpoints/checkpoint_best_ema.pth \
  --input data/raw/trial1.tif \
  --save-video

# YOLOv12
python track.py \
  --model-type yolo \
  --checkpoint models/yolov12/runs/detect/train/weights/best.pt \
  --input data/raw/trial1.tif

# LodeSTAR (uses the auto-labeling model directly — no separate training step)
python track.py \
  --model-type lodestar \
  --checkpoint ../data-setup/models/lodestar_model_15/model.pt \
  --input data/raw/trial1.tif
```

**CLI reference:**

| Argument | Config key | Default | Description |
|---|---|---|---|
| `--config` | — | `config.yaml` | Path to YAML config file |
| `--model-type` | `model.type` | `rf-detr` | `rf-detr`, `yolo`, or `lodestar` |
| `--checkpoint` | `model.checkpoint` | best EMA checkpoint | Path to model weights |
| `--variant` | `model.variant` | `large` | RF-DETR size: `nano`, `small`, `medium`, `large` |
| `--device` | `model.device` | `0` | Inference device (`0` for GPU, `cpu`) |
| `--threshold` | `detection.threshold` | `0.25` | Detection confidence threshold |
| `--input` | `input` | — | Video, image folder, or TIFF stack |
| `--output-dir` | `output.dir` | `evaluation/results/tracking_output` | Where to write results |
| `--tracker` | `tracking.tracker` | `trackpy` | `trackpy` (offline) or `bytetrack` (online) |
| `--search-range` | `tracking.search_range` | `10.0` | Trackpy: max pixel distance between frames |
| `--memory` | `tracking.memory` | `3` | Trackpy: frames a particle may be missing |
| `--stub-filter` | `tracking.stub_filter` | `5` | Trackpy: min track length to keep |
| `--save-video` | `output.save_video` | off | Save annotated `.mp4` |
| `--fps` | `output.fps` | `30` | FPS for output video |

---

## Full Pipeline Overview

```
Raw microscopy TIFFs
       │
       ▼
[data-setup] Extract frames, crop particles, train LodeSTAR
       │ YOLO-format labels
       ▼
[rf-detr] Train RF-DETR detection model on labeled data
       │ Trained checkpoint (.pth)
       ▼
[particle-tracking] Detect + track particles across frames
       │
       ▼
tracks.csv  +  annotated video
```

---

## Contributing

### Workflow

1. Create a branch off `main` for your changes:
   ```bash
   git checkout main
   git pull
   git checkout -b feat/your-feature-name
   ```
2. Make your changes and commit them (the pre-commit hook will run automatically).
3. Push the branch and open a pull request against `main`:
   ```bash
   git push -u origin feat/your-feature-name
   ```
4. Address any review feedback, then merge once approved.

### Pre-commit hook

This repo uses [pre-commit](https://pre-commit.com/) to auto-format Python files with [Black](https://black.readthedocs.io/) (line length 100) before every commit.

Install the hook once after cloning:

```bash
pip install pre-commit
pre-commit install
```

After that, Black runs automatically on staged files. If it reformats anything, stage the changes and commit again.

### Linting

`lint.sh` mirrors the CI lint check locally. Run it before opening a PR:

```bash
# Lint only files changed relative to origin/main (default)
./lint.sh

# Lint the entire repository
./lint.sh --full
```

Reports are written to `lint-reports/` (pylint text, JSON, and a summary). Fix any Black formatting issues with the command printed by the script, then re-run to confirm.

---

## Resources

- [LAMMPS Manual](https://docs.lammps.org/Manual.html)
- [LodeSTAR / DeepTrack2](https://github.com/softmatterlab/DeepTrack2)
- [RF-DETR (Roboflow)](https://github.com/roboflow/rf-detr)
- [OVITO (simulation visualization)](https://www.ovito.org/)
- [Light-Responsive Assembly](https://pubs.acs.org/doi/10.1021/acs.jpcb.4c02301)
- [Molecular Dynamics Simulation of Active Particles Video](https://www.youtube.com/watch?v=wsM2kUB6XU4&ab_channel=SoftMatterLab)
- [Molecular Dynamics Simulation of Active Particles (Brownian Motion)](https://arxiv.org/abs/2102.10399)
