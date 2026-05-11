# CLAUDE.md - Development Guide

## Git
- DO NOT MAKE ANY GIT COMMITS, I will do that manually

## Build and Setup Commands

### Environment Setup
- **Install Python Dependencies**: `pip install -r simulation/requirements.txt` (or `uv pip install -r simulation/requirements.txt` if using `uv`)
- **LAMMPS Setup**: Ensure [LAMMPS](https://docs.lammps.org/Install.html) is installed and the `lmp` executable is in your `PATH`.
- **Pre-commit Hooks**: `pre-commit install` (uses Black for formatting)

## Development Commands

### Linting & Formatting
- **Full Lint Check**: `./lint.sh --full` (Runs Black and Pylint)
- **Format Code**: `black --line-length=100 <path_to_file_or_dir>`
- **Lint Individual File**: `pylint --rcfile=.pylintrc <path_to_file>`

### Execution
- **Particle Tracking**: `python particle_tracking/tracking/track.py --source <input_path> --weights <model_path>`
- **LAMMPS Simulations**: Scripts located in `simulation/lammps/`.

## Coding Style Guidelines

### Naming Conventions
- **Directories & Files**: Always use `snake_case` (e.g., `particle_tracking/`, `track_particles.py`).
- **Classes**: Use `PascalCase` (e.g., `ParticleTracker`).
- **Functions & Variables**: Use `snake_case` (e.g., `detect_particles()`, `frame_count`).

### Formatting
- **Python**: Follow PEP 8 standards.
- **Line Length**: 100 characters (configured for Black and Pylint).
- **Tooling**: Use **Black** for formatting and **Pylint** for linting.

### Imports
- Prefer absolute imports or consistent relative imports within the defined pillars (`simulation/` and `particle_tracking/`).
- Avoid adding unnecessary paths to `sys.path` where possible.
