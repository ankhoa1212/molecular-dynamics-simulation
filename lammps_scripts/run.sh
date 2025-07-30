#!/bin/bash

# ==============================================================================
# A bash script to run a LAMMPS simulation using a filename provided as a
# command-line argument.
#
# Usage: ./run_lammps.sh <input_file> [output_directory]

# Set the name of the LAMMPS executable.
LAMMPS_EXECUTABLE="lmp"

# --- Set OpenMP Threads ---

# Automatically determine the number of available processor cores and set
# OMP_NUM_THREADS for multi-threaded execution with OpenMP packages.
if [[ -x "$(command -v nproc)" ]]; then
  # For Linux systems
  export OMP_NUM_THREADS=$(nproc)
else
  # Fallback if no command is found
  echo "Could not determine number of CPUs. Defaulting OMP_NUM_THREADS to 1."
  export OMP_NUM_THREADS=1
fi
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"

# Check if the input script exists as first argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <lammps_input_file> [output_directory]"
    echo "Example: $0 in.test"
    exit 1
fi

# Check if the output directory exists as second argument
if [ "$#" -eq 2 ]; then
    OUTPUT_DIR="$2"
else
    OUTPUT_DIR="results"
fi

# The input script is the first argument passed to the script.
INPUT_SCRIPT="$1"

# Create log file name based on the input script
LOG_FILE="${INPUT_SCRIPT}.log"

# --- Run Simulation ---
echo "=========================================="
echo "Starting LAMMPS simulation..."
echo "  Executable: $LAMMPS_EXECUTABLE"
echo "  Input file: $INPUT_SCRIPT"
echo "  Log file:   $LOG_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "=========================================="

# Run the LAMMPS simulation.
# The '-in' flag specifies the input script.
# The '-log' flag specifies the output log file.
# The '-var' flag allows passing a variable into the input script
"$LAMMPS_EXECUTABLE" -in "$INPUT_SCRIPT" -log "${OUTPUT_DIR}/${INPUT_SCRIPT}.log" -var filename "$INPUT_SCRIPT"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# If there is a trajectory file, move it to the output directory.
TRAJ_FILE="${INPUT_SCRIPT}.lammpstrj"
if [ -f "$TRAJ_FILE" ]; then
  mv "$TRAJ_FILE" "$OUTPUT_DIR/"
  echo "Moved trajectory file '$TRAJ_FILE' to '$OUTPUT_DIR/'"
fi

# --- Post-simulation ---
echo ""
echo "=========================================="
echo "LAMMPS simulation finished."
echo "Check the log file '$LOG_FILE' for details."
echo "Run the trajectory file '${INPUT_SCRIPT}.lammpstrj' with ovito for visualization"
echo "=========================================="

# --- Post-processing ---
