# molecular-dynamics-simulation

A two-pillar project combining LAMMPS-based molecular dynamics simulation with computer-vision particle tracking for microscopy data.

## Repository Structure

```
.
├── simulation/                  # Physics simulation and analysis
│   ├── lammps/
│   │   ├── inputs/              # LAMMPS .in configuration files
│   │   └── scripts/             # run.py / run.sh execution wrappers
│   ├── analysis/                # Hexatic order, phase diagrams, velocity/temp graphs
│   ├── processing/              # Trajectory and image post-processing
│   └── outputs/                 # Simulation results and logs (gitignored)
│
└── particle_tracking/           # Computer-vision detection and tracking
    ├── data/
    │   ├── raw/                 # Source microscopy frames
    │   ├── processing/          # Frame extraction and augmentation preview
    │   └── labeling/            # Auto-labelers (LodeSTAR, trackpy, YOLO)
    ├── models/
    │   ├── rf_detr/             # RF-DETR detection model
    │   ├── yolov12/             # YOLOv12 detection model
    │   ├── train_lodestar.py    # LodeSTAR training entry point
    │   ├── train_yolo.py        # YOLO training entry point
    │   └── weights/             # Pretrained model weights
    ├── tracking/
    │   ├── track.py             # Main tracking entry point
    │   └── configs/             # Tracking configuration YAML files
    └── evaluation/              # Results, MLflow DB, Lightning logs
```

## Setup

### Python Dependencies

```bash
pip install -r simulation/requirements.txt
# or, with uv:
uv pip install -r simulation/requirements.txt
```

### LAMMPS

[Install](https://docs.lammps.org/Install.html) and [build](https://docs.lammps.org/Build.html) LAMMPS, then add the executable to your `PATH`.

**Temporary (current session):**
```bash
EXECUTABLE_DIR=$PWD   # run from the directory containing the lmp executable
export PATH=$EXECUTABLE_DIR:$PATH
```

**Permanent (add to `~/.bashrc`):**
```bash
export PATH=/path/to/lammps/bin:$PATH
```

> **Note (Linux < 22.04):** OVITO may require `libxcb-cursor0` if `qt.qpa.plugin 6.5.0` is unavailable.

### Pre-commit Hooks

```bash
pre-commit install
```

## Usage

### Running a Simulation

```bash
python simulation/lammps/scripts/run.py simulation/lammps/inputs/<config>.in
```

### Particle Tracking

```bash
python particle_tracking/tracking/track.py \
    --source <path/to/frames> \
    --weights <path/to/weights>
```

### Training Detection Models

```bash
# RF-DETR
cd particle_tracking/models/rf_detr && python train.py

# YOLOv12
python particle_tracking/models/train_yolo.py

# LodeSTAR
python particle_tracking/models/train_lodestar.py
```

### Analysis

```bash
# Hexatic order parameter
python simulation/analysis/hexatic_order.py

# Velocity / temperature graphs
python simulation/analysis/velocity_graph.py
python simulation/analysis/temp_graph.py
```

## Resources

- [LAMMPS Manual](https://docs.lammps.org/Manual.html) — [Install](https://docs.lammps.org/Install.html) · [Examples](https://docs.lammps.org/Examples.html) · [Tools](https://docs.lammps.org/Tools.html)
- [OVITO](https://www.ovito.org/) — visualization of simulation trajectories
- [Light-Responsive Assembly](https://pubs.acs.org/doi/10.1021/acs.jpcb.4c02301)
- [MD Simulation of Active Particles (video)](https://www.youtube.com/watch?v=wsM2kUB6XU4&ab_channel=SoftMatterLab)
- [MD Simulation of Active Particles — Brownian Motion (arXiv)](https://arxiv.org/abs/2102.10399)
