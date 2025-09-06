#!/bin/bash

# ==============================================================================
# A bash script to run a LAMMPS simulation using a filename provided as a
# command-line argument.

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

# Set default values for optional arguments
OUTPUT_DIR="results"
MOLECULES="1000"
MOLECULES_END="$MOLECULES"
MOLECULES_STEP="1"
VAR_EPSILON="5.0"
VAR_EPSILON_END="$VAR_EPSILON"
VAR_EPSILON_STEP="1.0"

# Check if the input script exists as first argument
if [ "$#" -lt 1 ] || [ "$#" -gt 8 ]; then
  echo "Usage: $0 <lammps_input_file> [output_directory] [molecules [molecules_end molecules_step]] [var_epsilon [var_epsilon_end var_epsilon_step]]"
  echo "Examples:"
  echo "  $0 in.test"
  echo "  $0 in.test results"
  echo "  $0 in.test results 1000"
  echo "  $0 in.test results 1000 2000 100"
  echo "  $0 in.test results 1000 2000 100 5.0"
  echo "  $0 in.test results 1000 2000 100 5.0 10.0 1.0"
  exit 1
fi

# Check if the output directory exists as second argument
if [ "$#" -gt 1 ]; then
    OUTPUT_DIR="$2"
fi

# Check if the var for molecules exists as third argument
if [ "$#" -gt 2 ]; then
    MOLECULES="$3"
    MOLECULES_END="$4"
    MOLECULES_STEP="$5"
fi

# Check if the var for epsilon exists as fourth argument
if [ "$#" -gt 5 ]; then
    VAR_EPSILON="$6"
    VAR_EPSILON_END="$7"
    VAR_EPSILON_STEP="$8"
fi

# The input script is the first argument passed to the script.
INPUT_SCRIPT="$1"

# Create log file name based on the input script
LOG_FILE="${INPUT_SCRIPT}.log"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create log file directory within output directory
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${INPUT_SCRIPT}.log"

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
# The '-var' flag allows passing a variable into the LAMMPS input script
for (( m=$MOLECULES; m<=${MOLECULES_END:-$MOLECULES}; m+=${MOLECULES_STEP:-1} )); do
  for (( e=$(printf "%.0f" "${VAR_EPSILON}"); e<=$(printf "%.0f" "${VAR_EPSILON_END:-$VAR_EPSILON}"); e+=$(printf "%.0f" "${VAR_EPSILON_STEP:-1}") )); do
    EPSILON_VAL=$(printf "%.1f" "$e")
    FILENAME="${INPUT_SCRIPT}_${m}_${EPSILON_VAL}"
    "$LAMMPS_EXECUTABLE" -in "$INPUT_SCRIPT" -log "$LOG_FILE" -var filename "$FILENAME" -var molecules "$m" -var var_epsilon "$EPSILON_VAL"
  done
done

# If there is a trajectory file, move it to the output directory.
# Find any file ending with .lammpstrj in the current directory
for TRAJ_FILE in ./*.lammpstrj; do
  if [ -f "$TRAJ_FILE" ]; then
    mv "$TRAJ_FILE" "$OUTPUT_DIR/"
    echo "Moved trajectory file '$TRAJ_FILE' to '$OUTPUT_DIR/'"
  fi
done

# --- Post-simulation ---
echo ""
echo "=========================================="
echo "LAMMPS simulation finished."
echo "Check the log file '$LOG_FILE' for details."
echo "Run the trajectory file '$OUTPUT_DIR/${INPUT_SCRIPT}_${MOLECULES}_${VAR_EPSILON}.lammpstrj' with ovito for visualization"
echo "=========================================="

# --- Post-processing ---
