import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from phase_diagram import extract_epsilon_and_molecules


def generate_temp_graph_filename(filename, ending, output_dir=None):
    # Get base filename without extension
    base_name = os.path.basename(filename)
    if "." in base_name:
        base_name = base_name[:base_name.rfind(".")]
    
    out_name = f"{base_name}_temp_graph_{ending}.png"
    
    if output_dir:
        out_name = os.path.join(output_dir, out_name)
    
    return out_name


def plot_log_temperature(filename, output_dir=None, no_show=False):
    steps = []
    temps = []

    print(f"Reading log file: {filename}...")

    reading_data = False

    try:
        with open(filename, "r") as f:
            for line in f:
                # Detect the start of the data table in log.lammps
                if "Step" in line and "Temp" in line:
                    reading_data = True
                    continue

                # Stop reading if we hit the end loop info
                if "Loop time" in line:
                    reading_data = False
                    continue

                if reading_data:
                    parts = line.split()
                    # Ensure line is data (numbers) and not empty
                    if len(parts) > 1 and parts[0].isdigit():
                        try:
                            # Typically Step is Col 0, Temp is Col 1
                            # Based on: thermo_style custom step temp ...
                            s = int(parts[0])
                            t = float(parts[1])
                            steps.append(s)
                            temps.append(t)
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        sys.exit(1)

    if not steps:
        print("No temperature data found. Did you point to the correct .log file?")
        sys.exit(1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps, temps, color="blue", linewidth=2.0, label="LAMMPS Temp")

    plt.title("Temperature over Time (From LAMMPS Log)")
    plt.xlabel("Timestep")
    plt.ylabel("Temperature (LJ Units)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    output_img = generate_temp_graph_filename(filename, "log", output_dir)
    plt.savefig(output_img)
    print(f"Graph saved to: {output_img}")
    if not no_show:
        plt.show()


def plot_temperatures(filename, output_dir=None, no_show=False):
    timesteps = []
    total_temps = []
    corrected_temps = []  # Drift-Corrected (True Thermal)

    # Define the center of the box
    center_x = 100.0
    center_y = 100.0

    print(f"Reading file: {filename}...")

    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                step = int(f.readline().strip())
                timesteps.append(step)

            if "ITEM: NUMBER OF ATOMS" in line:
                n_atoms = int(f.readline().strip())

            if "ITEM: ATOMS" in line:
                # Store data for this frame to process in bulk
                v_x_list = []
                v_y_list = []
                x_list = []
                y_list = []

                for _ in range(n_atoms):
                    atom_data = f.readline().split()
                    x_list.append(float(atom_data[2]))
                    y_list.append(float(atom_data[3]))
                    v_x_list.append(float(atom_data[4]))
                    v_y_list.append(float(atom_data[5]))

                # Convert to numpy arrays for vector math
                x = np.array(x_list)
                y = np.array(y_list)
                vx = np.array(v_x_list)
                vy = np.array(v_y_list)

                # --- 1. Calculate Radial and Tangential Components ---
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + dy**2)

                # Avoid division by zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    rx = dx / dist
                    ry = dy / dist
                    # Handle exact center case
                    rx[dist == 0] = 0
                    ry[dist == 0] = 0

                # Radial Velocity (Scalar) for each atom
                v_rad = vx * rx + vy * ry

                # Tangential Velocity Vectors
                v_tan_x = vx - (v_rad * rx)
                v_tan_y = vy - (v_rad * ry)

                # --- 2. Calculate "Standard" Total Temperature ---
                # Raw KE = 0.5 * m * (vx^2 + vy^2)
                # T = Sum(v^2) / (2 * N)  [2 Degrees of Freedom]
                total_sq_sum = np.sum(vx**2 + vy**2)
                t_tot = total_sq_sum / (2.0 * n_atoms)
                total_temps.append(t_tot)

                # --- 3. Calculate Drift-Corrected Temperature ---
                # Calculate the Mean Radial Velocity (The "Bulk Implosion Speed")
                # This is the velocity added by the force that we want to remove.
                mean_v_rad = np.mean(v_rad)

                # Subtract this coherent drift from every atom's radial velocity
                # We essentially shift the reference frame to move with the implosion.
                v_rad_fluctuation = v_rad - mean_v_rad

                # Re-calculate Total Kinetic Energy using the FLUCTUATIONS only
                # KE_corrected = KE_tangential + KE_radial_fluctuation

                v_tan_sq = v_tan_x**2 + v_tan_y**2
                v_rad_fluc_sq = v_rad_fluctuation**2

                corrected_sq_sum = np.sum(v_tan_sq + v_rad_fluc_sq)
                t_corrected = corrected_sq_sum / (2.0 * n_atoms)
                corrected_temps.append(t_corrected)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot Standard Total Temp
    plt.plot(
        timesteps,
        total_temps,
        color="red",
        linewidth=1.0,
        alpha=0.6,
        label="Raw Total Temp (Includes Force Work)",
    )

    # Plot Corrected Temp
    plt.plot(
        timesteps,
        corrected_temps,
        color="blue",
        linewidth=2.0,
        label="Corrected Temp (Force Drift Removed)",
    )

    plt.title("Temperature over Time (From LAMMPS dump)")
    plt.xlabel("Timestep")
    plt.ylabel("Temperature (LJ Units)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    output_img = generate_temp_graph_filename(filename, "lammpstrj", output_dir)
    print(f"Graph saved to: {output_img}")
    plt.savefig(output_img)
    if not no_show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot drift-corrected temperature from LAMMPS trajectory"
    )
    ex = "/home/austin/git/molecular-dynamics-simulation/lammps-scripts/test_same/test.in_100_5.0.lammpstrj"
    # ex = '/home/austin/git/molecular-dynamics-simulation/lammps-scripts/test_same/logs/test.in_100_5.0.log'
    parser.add_argument("--filename", "-f", default=ex, help="Path to file")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--no-show", action="store_true", help="Do not display the graph")
    args = parser.parse_args()
    if args.filename.endswith(".log"):
        plot_log_temperature(args.filename, args.output_dir, args.no_show)
    elif args.filename.endswith(".lammpstrj"):
        print("Note: graphing the .log file will be more accurate")
        plot_temperatures(args.filename, args.output_dir, args.no_show)
    else:
        print("Error: File should be either .log or .lammpstrj")
        sys.exit(1)
