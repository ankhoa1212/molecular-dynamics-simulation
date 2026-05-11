# Simulation

LAMMPS-based molecular dynamics simulations and post-simulation analysis.

## Setup

### 1. Install Python dependencies

```bash
pip install -r simulation/requirements.txt
# or, with uv:
uv pip install -r simulation/requirements.txt
```

### 2. Install LAMMPS

[Install](https://docs.lammps.org/Install.html) and [build](https://docs.lammps.org/Build.html) LAMMPS, then make the `lmp` executable available on your `PATH`.

**Temporary (current session):**
```bash
EXECUTABLE_DIR=$PWD   # run from the directory containing lmp
export PATH=$EXECUTABLE_DIR:$PATH
```

**Permanent (add to `~/.bashrc`):**
```bash
export PATH=/path/to/lammps/bin:$PATH
```

---

## Running Simulations

Input scripts live in `lammps/inputs/`. Two launchers are available — both support sweeping over a grid of molecule counts and epsilon (LJ interaction strength) values, running each combination in parallel.

### `run.py` (recommended)

```bash
python simulation/lammps/scripts/run.py <input_file> [options]
```

Runs LAMMPS in parallel using Python's `ThreadPoolExecutor`, automatically sets `OMP_NUM_THREADS` to the number of available cores, and saves trajectories + logs to the output directory.

**Key options:**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `input_file` | *(required)* | Path to the `.in` LAMMPS script |
| `output_dir` | `results` | Directory for `.lammpstrj` files and logs |
| `molecules` | `100` | Start of molecule count sweep |
| `molecules_end` | same as start | End of molecule count sweep |
| `molecules_step` | `1` | Step size for molecule sweep |
| `var_epsilon` | `5.0` | Start of epsilon sweep |
| `var_epsilon_end` | same as start | End of epsilon sweep |
| `var_epsilon_step` | `5.0` | Step size for epsilon sweep |
| `--var_tstart` | `1.0` | Starting temperature |
| `--var_tstop` | `1.0` | Stopping temperature |
| `--vel_force_scale` | auto | Scaling factor for initial velocity or continuous force |
| `--steps` | `10000` | Number of simulation timesteps |

**Single run:**
```bash
python simulation/lammps/scripts/run.py simulation/lammps/inputs/central_pair_interaction.in
```

**Parameter sweep:**
```bash
python simulation/lammps/scripts/run.py simulation/lammps/inputs/central_pair_interaction.in \
    results 100 1500 100 0.5 5.0 0.5
```
This sweeps molecules from 100 → 1500 (step 100) and epsilon from 0.5 → 5.0 (step 0.5), producing 300 parallel runs.

### Using a JSON Config

Pass `--config <file>.json` instead of positional arguments. The JSON keys map directly to the argument names above:

```json
{
    "input_file": "simulation/lammps/inputs/continuous_force.in",
    "output_dir": "simulation/outputs/continuous_force_sweep",
    "molecules": "100",
    "molecules_end": "1500",
    "molecules_step": "100",
    "var_epsilon": "0.5",
    "var_epsilon_end": "5.0",
    "var_epsilon_step": "0.5",
    "t_start": "1.0",
    "t_stop": "0.05",
    "steps": "15000",
    "vel_force_scale": "1"
}
```

```bash
python simulation/lammps/scripts/run.py --config my_config.json
```

Any extra keys not in the positional list are passed through as `--key value` flags.

### `run.sh`

A Bash equivalent for quick one-off runs. Accepts the same positional arguments as `run.py` but without JSON config support and without drift-corrected temperature analysis.

```bash
bash simulation/lammps/scripts/run.sh simulation/lammps/inputs/central_pair_interaction.in \
    results 100 1500 100 5.0 10.0 5.0
```

---

## Analysis Scripts

All scripts are in `simulation/analysis/`. They operate on `.lammpstrj` trajectory files (or `.log` files where noted) written to `simulation/outputs/` after a run.

### `graph.py` — Batch analysis runner

Runs `hexatic_order_graph.py`, `velocity_graph.py`, and `temp_graph.py` together on one file or an entire directory of trajectory files.

```bash
# Single file
python simulation/analysis/graph.py simulation/outputs/central_pair_interaction_100_5.0.lammpstrj

# All .lammpstrj files in a directory
python simulation/analysis/graph.py simulation/outputs/ --output_dir simulation/outputs/plots

# Suppress interactive plot windows (useful for batch/headless runs)
python simulation/analysis/graph.py simulation/outputs/ --no-show
```

---

### `hexatic_order_graph.py` — Hexatic order over time

Plots the global hexatic order parameter |ψ₆| as a function of simulation frame for a single trajectory. |ψ₆| ranges from 0 (disordered) to 1 (perfect hexagonal crystal). Saves a `.png` alongside the trajectory file.

```bash
python simulation/analysis/hexatic_order_graph.py \
    simulation/outputs/central_pair_interaction_100_5.0.lammpstrj \
    [--output_dir simulation/outputs/plots] [--no-show]
```

*Depends on `hexatic_order_analysis.py` (freud) and `phase_diagram.py` for filename parsing.*

---

### `hexatic_order_analysis.py` — Hexatic order computation (freud)

The calculation backend used by `hexatic_order_graph.py` and `phase_diagram.py`. Parses a `.lammpstrj` file frame-by-frame and computes the per-atom hexatic order parameter using the [`freud`](https://freud.readthedocs.io/) library.

Can also be run standalone to print per-frame |ψ₆| values and display a summary plot:

```bash
python simulation/analysis/hexatic_order_analysis.py \
    simulation/outputs/central_pair_interaction_100_5.0.lammpstrj
```

---

### `hexatic_order.py` — Hexatic order and Voronoi (legacy)

An older, self-contained implementation using `scipy` and `sklearn` instead of `freud`. Reads particle positions from text files, computes |ψ₆| via nearest-neighbour angle sums, and optionally draws colour-coded Voronoi diagrams (4–9+ sided polygons each get a distinct colour).

Useful for quick visual inspection of a single configuration snapshot.

```bash
python simulation/analysis/hexatic_order.py
```

*(Processes all subdirectories in the current working directory by default.)*

---

### `velocity_graph.py` — Velocity magnitude over time

Plots the mean ± standard deviation of particle velocity magnitudes across all atoms at each timestep. Reads `vx` / `vy` columns directly from the trajectory.

```bash
python simulation/analysis/velocity_graph.py \
    --filename simulation/outputs/central_pair_interaction_100_5.0.lammpstrj \
    [--output_dir simulation/outputs/plots] [--no-show]
```

---

### `temp_graph.py` — Temperature over time

Two modes depending on input file type:

- **`.log` file** — reads `Step` / `Temp` columns from LAMMPS thermodynamic output. Most accurate.
- **`.lammpstrj` file** — computes temperature from kinetic energy. Also plots a *drift-corrected* temperature that subtracts the mean radial (bulk-implosion) velocity component, isolating true thermal motion from force-driven drift.

```bash
# From log file (recommended)
python simulation/analysis/temp_graph.py \
    --filename simulation/outputs/logs/central_pair_interaction_100_5.0.log

# From trajectory (includes drift-corrected curve)
python simulation/analysis/temp_graph.py \
    --filename simulation/outputs/central_pair_interaction_100_5.0.lammpstrj
```

---

### `phase_diagram.py` — N vs ε phase diagram

Scans a directory of `.lammpstrj` files, parses `N` (molecule count) and `ε` (epsilon) from their filenames, computes the final-frame mean |ψ₆| for each, and plots a scatter diagram of N vs ε colored by hexatic order (red = disordered, green = crystalline).

Filenames must follow the pattern produced by `run.py`: `<script>_<N>_<ε>.lammpstrj`.

```bash
python simulation/analysis/phase_diagram.py simulation/outputs/

# Test a single file's filename parsing and hexatic output
python simulation/analysis/phase_diagram.py --test \
    simulation/outputs/central_pair_interaction_100_5.0.lammpstrj
```

---

## Processing Scripts

`simulation/processing/` contains utilities for converting simulation output into other formats.

| Script | Input | Output | Description |
| :--- | :--- | :--- | :--- |
| `lammpstrj_to_video.py` | `.lammpstrj` | `.mp4` | Renders particle positions from a trajectory as a video |
| `tif_to_frames.py` | `.tif` stack | PNG frames | Extracts individual frames from a multi-page TIFF |
| `frames_to_video.py` | PNG frames dir | `.mp4` | Assembles a directory of PNG frames into a video |
