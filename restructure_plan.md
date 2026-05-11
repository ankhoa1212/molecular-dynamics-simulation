# Repository Restructure Plan

This document outlines the restructuring of the `molecular-dynamics-simulation` repository to improve maintainability, scalability, and logical separation of concerns.

> **Status: Complete.** The restructure has been fully implemented.

## 1. Top-Level Structure

The repository is divided into two main functional pillars:
- `simulation/`: Data generation and physics-based analysis.
- `particle_tracking/`: Particle detection, tracking, and evaluation.

## 2. Naming Conventions

- **Directories**: All directory names use `snake_case` (e.g., `lammps_inputs`, `rf_detr`).
- **Files**: All Python scripts, shell scripts, and configuration files use `snake_case`.
- **Classes/Variables**: Follow PEP 8 (PascalCase for classes, snake_case for variables/functions).

## 3. Dependency Management

- **Primary Manager**: Use [`uv`](https://github.com/astral-sh/uv) for managing virtual environments and installing packages.
- **Configuration**:
  - For `uv` users: Use `pyproject.toml` and `uv.lock`.
  - For compatibility: Maintain a `requirements.txt` (generated via `uv export --format requirements-txt > requirements.txt`).
- **Environment**: Always use a local virtual environment (`.venv/`).

## 4. Detailed Mapping

### 1. Simulation Pillar (`simulation/`)

| Original Path | New Path | Description |
| :--- | :--- | :--- |
| `lammps-scripts/*.in` | `simulation/lammps/inputs/` | LAMMPS simulation configurations. |
| `lammps-scripts/run.py`, `run.sh` | `simulation/lammps/scripts/` | Execution wrappers. |
| `lammps-scripts/hexatic_order*.py` | `simulation/analysis/` | Physics analysis scripts. |
| `lammps-scripts/graph.py`, `velocity_graph.py`, `temp_graph.py`, `phase_diagram.py` | `simulation/analysis/` | Visualization of physics data. |
| `lammps-scripts/lammps_parser.py` | `simulation/analysis/` | LAMMPS output parser. |
| `lammps-scripts/lammpstrj_to_video.py` | `simulation/processing/` | Trajectory post-processing. |
| `lammps-scripts/tif_to_frames.py` | `simulation/processing/` | Image format conversion. |
| *(new)* | `simulation/outputs/` | Simulation results and logs (gitignored). |

### 2. Particle Tracking Pillar (`particle_tracking/`)

#### Data and Labeling (`particle_tracking/data/`)
- **`processing/`**: `extract_frames.py`, `preview_augmentations.py`, `check_blur.py`.
- **`labeling/`**: `lodestar_autolabeler.py`, `trackpy_auto_labeler.py`, `crop_tool.py`, `label_images.py`, `yolo_to_coco.py`.

#### Detection Models (`particle_tracking/models/`)
- **`rf_detr/`**: RF-DETR model directory (renamed from `rf-detr` to enforce `snake_case`).
- **`yolov12/`**: YOLOv12 project directory (includes `yolov12/yolov12` submodule).
- **`train_lodestar.py`**, **`train_yolo.py`**: Training entry points.
- **`weights/`**: Pretrained model weights.
- **`legacy/`**: Archived older model code.

#### Tracking Engine (`particle_tracking/tracking/`)
- **`track.py`**: Main tracking entry point.
- **`configs/`**: Tracking configuration YAML files.

#### Evaluation (`particle_tracking/evaluation/`)
- **`results/`**: Per-run evaluation outputs.
- **`runs/`**: MLFlow tracking data.
- **`lightning_logs/`**: PyTorch Lightning training logs.
- **`mlflow.db`**: MLFlow experiment database.

## 5. Implementation Steps (Completed)

1. ~~**Preparation**: Ensure all current work is committed.~~
2. ~~**Move Simulation Files**: Create `simulation/` structure and migrate files.~~
3. ~~**Move Vision Pipeline Files**: Create `particle_tracking/` structure and migrate files.~~
4. ~~**Enforce Naming**: Rename `rf-detr` → `rf_detr` during migration.~~
5. ~~**Update Configuration Paths**: Update hardcoded paths in `.md` and `.yaml` files.~~
6. ~~**Cleanup**: Remove empty old directories (`data-setup/`, `lammps-scripts/`).~~
7. ~~**Update `.gitmodules`**: Update submodule path for `yolov12/yolov12`.~~

## 6. Rationale
- **Separation of Concerns**: Clearly separates "generating simulation data" from "detecting and tracking particles."
- **Scalability**: New models or simulation types can be added without cluttering the root.
- **Professionalism**: A clean root directory makes the project easier to navigate for new contributors.
