#!/bin/bash

# ==============================================================================
# A bash script to run a LAMMPS simulation using a filename provided as a
# command-line argument.
#
# Usage: ./run_lammps.sh <input_file>

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

# Check if the input script exists
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <lammps_input_file>"
    echo "Example: $0 in.test"
    exit 1
fi

# The input script is the first argument passed to the script.
INPUT_SCRIPT="$1"

# Automatically generate a log file name based on the input script name.
# For example, if the input is "in.test", the log will be "in.test.log".
LOG_FILE="${INPUT_SCRIPT}.log"

# --- Run Simulation ---
echo "=========================================="
echo "Starting LAMMPS simulation..."
echo "  Executable: $LAMMPS_EXECUTABLE"
echo "  Input file: $INPUT_SCRIPT"
echo "  Log file:   $LOG_FILE"
echo "=========================================="

# Run the LAMMPS simulation.
# The '-in' flag specifies the input script.
# The '-log' flag specifies the output log file.
# The '-var' flag allows passing a variable into the input script
"$LAMMPS_EXECUTABLE" -in "$INPUT_SCRIPT" -log "$LOG_FILE" -var filename "$INPUT_SCRIPT"

# --- Post-simulation ---
echo ""
echo "=========================================="
echo "LAMMPS simulation finished."
echo "Check the log file '$LOG_FILE' for details."
echo "Run the trajectory file '${INPUT_SCRIPT}.lammpstrj' with ovito for visualization"
echo "=========================================="

# --- Post-processing ---
