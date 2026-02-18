#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run_simulation(cmd, filename, output_dir):
    """Runs a single simulation command and moves the output file."""
    try:
        # Run the command
        # Using shell=True to match bash behavior, though shell=False with list is usually safer.
        # Given the command string construction, shell=True is easier here.
        subprocess.run(cmd, shell=True, check=True)

        # Move the trajectory file
        traj_file = f"{filename}.lammpstrj"
        if os.path.exists(traj_file):
            shutil.move(traj_file, os.path.join(output_dir, traj_file))
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}\n{e}")


def main():
    """Main function to parse arguments and run LAMMPS simulations in parallel."""
    # Set LAMMPS executable
    lammps_executable = "lmp"

    # Determine number of CPUs
    try:
        # Try to get the number of CPUs available to the process
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for systems where sched_getaffinity is not available
        num_cpus = os.cpu_count() or 1

    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    print(f"Setting OMP_NUM_THREADS to {num_cpus}")

    # Pre-process sys.argv to handle --config file
    while "--config" in sys.argv:
        try:
            config_index = sys.argv.index("--config")
            if config_index + 1 < len(sys.argv):
                config_file = sys.argv[config_index + 1]
                config_args = []

                if config_file.endswith(".json"):
                    with open(config_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            config_args = [str(x) for x in data]
                        elif isinstance(data, dict):
                            # Handle positional arguments in order defined in parser
                            positional_keys = [
                                "input_file",
                                "output_dir",
                                "molecules",
                                "molecules_end",
                                "molecules_step",
                                "var_epsilon",
                                "var_epsilon_end",
                                "var_epsilon_step",
                            ]
                            for key in positional_keys:
                                if key in data:
                                    config_args.append(str(data[key]))

                            # Handle optional arguments (flags)
                            for key, value in data.items():
                                if key in positional_keys:
                                    continue
                                # Determine flag prefix
                                prefix = "-" if len(key) == 1 else "--"
                                # If key doesn't start with -, add prefix
                                arg_name = (
                                    key if key.startswith("-") else f"{prefix}{key}"
                                )
                                config_args.append(arg_name)
                                config_args.append(str(value))
                else:
                    with open(config_file, "r", encoding="utf-8") as f:
                        # Read content, ignore comments, split by whitespace
                        for line in f:
                            line = line.split("#", 1)[0].strip()
                            if line:
                                config_args.extend(line.split())

                # Replace --config and its value with the file arguments
                sys.argv = (
                    sys.argv[:config_index] + config_args + sys.argv[config_index + 2 :]
                )
            else:
                print("Error: --config requires a file path")
                sys.exit(1)
        except Exception as e:
            print(f"Error processing config file: {e}")
            sys.exit(1)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run LAMMPS simulation.")
    parser.add_argument(
        "--config", help="Path to file containing arguments", required=False
    )
    parser.add_argument("input_file", help="LAMMPS input file")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("molecules", nargs="?", help="Start molecules")
    parser.add_argument("molecules_end", nargs="?", help="End molecules")
    parser.add_argument("molecules_step", nargs="?", help="Step molecules")
    parser.add_argument("var_epsilon", nargs="?", help="Start epsilon")
    parser.add_argument("var_epsilon_end", nargs="?", help="End epsilon")
    parser.add_argument("var_epsilon_step", nargs="?", help="Step epsilon")

    # New options
    parser.add_argument(
        "--var_tstart",
        "--t_start",
        dest="var_tstart",
        default="1.0",
        help="Variable tstart (default: 1.0)",
    )
    parser.add_argument(
        "--var_tstop",
        "--t_stop",
        dest="var_tstop",
        default="1.0",
        help="Variable tstop (default: 1.0)",
    )

    parser.add_argument(
        "--vel_force_scale",
        dest="vel_force_scale",
        default=None,
        help="Scaling factor for initial velocity or continuous force (default: 9 for velocity_initialization.in, 1.0 for continuous_force.in)",
    )
    parser.add_argument(
        "--steps", default="10000", help="Simulation steps (default: 10000)"
    )

    args = parser.parse_args()

    # Default values
    output_dir = "results"
    molecules = "100"
    molecules_end = None
    molecules_step = "100"
    var_epsilon = "5.0"
    var_epsilon_end = None
    var_epsilon_step = "5.0"
    vel_force_scale = args.vel_force_scale

    # Apply logic similar to bash script
    if args.output_dir is not None:
        output_dir = args.output_dir

    if args.molecules is not None:
        molecules = args.molecules
        molecules_end = args.molecules_end
        molecules_step = args.molecules_step

        # If step is not provided when molecules is provided, default to 1 (matching bash script behavior)
        if molecules_step is None:
            molecules_step = "1"

    if args.var_epsilon is not None:
        var_epsilon = args.var_epsilon
        var_epsilon_end = args.var_epsilon_end
        var_epsilon_step = args.var_epsilon_step

        # If step is not provided when epsilon is provided, default to 1 (matching bash script behavior)
        if var_epsilon_step is None:
            var_epsilon_step = "1"

    # Convert to types
    try:
        m_start = int(molecules)
        m_end = int(molecules_end) if molecules_end is not None else m_start
        m_step = int(molecules_step)

        e_start = float(var_epsilon)
        e_end = float(var_epsilon_end) if var_epsilon_end is not None else e_start
        e_step = float(var_epsilon_step)
        if e_step == 0:
            e_step = 1.0

        # Set default scale if not provided
        if vel_force_scale is None:
            # Choose default based on input script
            if "velocity_initialization" in input_script:
                vel_force_scale = "9"
            else:
                vel_force_scale = "1.0"
    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    input_script = args.input_file
    
    # Strip .in extension for output filename generation
    base_script_name = os.path.basename(input_script)
    if base_script_name.endswith(".in"):
        base_script_name = base_script_name[:-3]
        
    log_dir = os.path.join(output_dir, "logs")

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 100)
    print("Starting LAMMPS simulation...")
    print(f"  Executable: {lammps_executable}")
    print(f"  Input file: {input_script}")
    print(f"  Log dir:   {log_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 100)

    commands = []

    # Clear commands.txt
    with open("commands.txt", "w", encoding="utf-8") as f:
        pass

    m = m_start

    # Helper to check loop condition
    def check_molecules_loop(current_m, end_m, step_m):
        if step_m > 0:
            return current_m <= end_m
        else:
            return current_m >= end_m

    while check_molecules_loop(m, m_end, m_step):
        e = e_start

        def check_epsilon_loop(current_e, end_e, step_e):
            # Float comparison with tolerance
            if step_e > 0:
                return current_e <= end_e + 1e-9
            else:
                return current_e >= end_e - 1e-9

        while check_epsilon_loop(e, e_end, e_step):
            epsilon_val = f"{e:.1f}"
            filename = f"{base_script_name}_{m}_{epsilon_val}"
            log_file = os.path.join(log_dir, f"{filename}.log")

            # Set current tstart and tstop to either epsilon or fixed value
            current_tstart = (
                epsilon_val if args.var_tstart == "epsilon" else args.var_tstart
            )
            current_tstop = (
                epsilon_val if args.var_tstop == "epsilon" else args.var_tstop
            )

            cmd = (
                f'"{lammps_executable}" -in "{input_script}" -log "{log_file}" '
                f'-var filename "{filename}" -var molecules "{m}" -var var_epsilon "{epsilon_val}" '
                f'-var var_tstart "{current_tstart}" -var var_tstop "{current_tstop}" -var steps "{args.steps}" '
                f'-var vel_force_scale "{vel_force_scale}"'
            )
            commands.append((cmd, filename))

            # Append to commands.txt
            with open("commands.txt", "a", encoding="utf-8") as f:
                f.write(cmd + "\n")

            e += e_step

        m += m_step

    # Run commands in parallel
    max_jobs = num_cpus
    print(f"Running simulations in parallel with up to {max_jobs} jobs...")

    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        futures = [
            executor.submit(run_simulation, cmd, filename, output_dir)
            for cmd, filename in commands
        ]
        for future in futures:
            future.result()  # Wait for all to complete

    print("All parallel jobs finished.")

    # Cleanup any remaining .lammpstrj files
    for file in os.listdir("."):
        if file.endswith(".lammpstrj"):
            shutil.move(file, os.path.join(output_dir, file))
            print(f"Moved trajectory file '{file}' to '{output_dir}/'")

    print("=" * 100)
    print("LAMMPS simulation finished.")
    print(f"Check the log directory '{log_dir}' for details.")

    example_epsilon = f"{e_start:.1f}"
    print(
        f"Visualize the output: 'ovito {output_dir}/{base_script_name}_{m_start}_{example_epsilon}.lammpstrj'"
    )
    print(
        f"Graph the temperature: 'python temp_graph.py --filename {output_dir}/{base_script_name}_{m_start}_{example_epsilon}.lammpstrj'"
    )
    print(
        f"Graph the velocity: 'python velocity_graph.py --filename {output_dir}/{base_script_name}_{m_start}_{example_epsilon}.lammpstrj'"
    )
    print(
        f"Graph the hexatic order: 'python hexatic_order_graph.py {output_dir}/{base_script_name}_{m_start}_{example_epsilon}.lammpstrj'"
    )
    print(
        f"Create a phase diagram of the outputs: 'python phase_diagram.py {output_dir}'"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
