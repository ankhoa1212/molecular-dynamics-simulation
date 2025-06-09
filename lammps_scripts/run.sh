# Set the name of the LAMMPS executable.
LAMMPS_EXECUTABLE="lmp"

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
"$LAMMPS_EXECUTABLE" -in "$INPUT_SCRIPT" -log "$LOG_FILE"

# --- Post-simulation ---
echo ""
echo "=========================================="
echo "LAMMPS simulation finished."
echo "Check the log file '$LOG_FILE' for details."
echo "=========================================="
