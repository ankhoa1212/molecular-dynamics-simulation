"""
This module processes LAMMPS log and dump files to plot temperature evolution over time.
It supports both standard thermodynamic output from logs and drift-corrected temperature
calculations from trajectory dump files.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from lammps_parser import parse_lammps_dump


def generate_temp_graph_filename(filename, ending, output_dir=None):
    """
    Generates a consistent output filename for the temperature graph.
    """
    # Get base filename without extension
    base_name = os.path.basename(filename)
    if "." in base_name:
        base_name = base_name[: base_name.rfind(".")]

    out_name = f"{base_name}_temp_graph_{ending}.png"

    if output_dir:
        out_name = os.path.join(output_dir, out_name)

    return out_name


def plot_log_temperature(filename, output_dir=None, no_show=False):
    """
    Plots temperature from a LAMMPS log file.
    """
    steps = []
    temps = []

    print(f"Reading log file: {filename}...")

    reading_data = False

    try:
        with open(filename, "r", encoding="utf-8") as f:
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


def _process_atom_lines(atom_lines):
    """
    Parses atom lines and returns position and velocity arrays.
    """
    data_list = []
    for line in atom_lines:
        parts = line.split()
        # x, y, vx, vy are indices 2, 3, 4, 5
        data_list.append([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])

    data_arr = np.array(data_list)
    return data_arr[:, 0], data_arr[:, 1], data_arr[:, 2], data_arr[:, 3]


def _compute_radial_projection(x, y, vx, vy):
    """Computes radial velocity and tangential squared velocity."""
    center_x = 100.0
    center_y = 100.0
    dx = x - center_x
    dy = y - center_y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        rx = np.divide(dx, dist, out=np.zeros_like(dx), where=dist != 0)
        ry = np.divide(dy, dist, out=np.zeros_like(dy), where=dist != 0)

    v_rad = vx * rx + vy * ry
    v_sq = vx ** 2 + vy ** 2
    v_tan_sq = v_sq - v_rad ** 2
    return v_rad, v_sq, v_tan_sq


def _compute_frame_temperature(x, y, vx, vy, n_atoms):
    """
    Computes standard and drift-corrected temperatures for a single frame.
    """
    v_rad, v_sq, v_tan_sq = _compute_radial_projection(x, y, vx, vy)

    # --- 2. Calculate "Standard" Total Temperature ---
    # Raw KE = 0.5 * m * (vx^2 + vy^2)
    # T = Sum(v^2) / (2 * N)  [2 Degrees of Freedom]
    t_tot = np.sum(v_sq) / (2.0 * n_atoms)

    # --- 3. Calculate Drift-Corrected Temperature ---
    # Calculate the Mean Radial Velocity (The "Bulk Implosion Speed")
    mean_v_rad = np.mean(v_rad)

    # Subtract this coherent drift from every atom's radial velocity
    v_rad_fluctuation = v_rad - mean_v_rad

    # Re-calculate Total Kinetic Energy using the FLUCTUATIONS only
    corrected_sq_sum = np.sum(v_tan_sq + v_rad_fluctuation ** 2)
    t_corrected = corrected_sq_sum / (2.0 * n_atoms)

    return t_tot, t_corrected


def _calculate_temps_for_frame(frame):
    """calculates temperature for a single frame"""
    x, y, vx, vy = _process_atom_lines(frame["atoms"])
    return _compute_frame_temperature(x, y, vx, vy, frame["n_atoms"])


def plot_temperatures(filename, output_dir=None, no_show=False):
    """
    Plots drift-corrected temperature from a LAMMPS dump file.
    """
    timesteps = []
    total_temps = []
    corrected_temps = []  # Drift-Corrected (True Thermal)

    print(f"Reading file: {filename}...")

    try:
        for frame in parse_lammps_dump(filename):
            if frame["n_atoms"] == 0:
                continue

            t_tot, t_corr = _calculate_temps_for_frame(frame)

            timesteps.append(frame["timestep"])
            total_temps.append(t_tot)
            corrected_temps.append(t_corr)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    if not timesteps:
        print("No valid temperature data found.")
        return

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
    ex = (
        "/home/austin/git/molecular-dynamics-simulation/"
        "lammps-scripts/test_same/test.in_100_5.0.lammpstrj"
    )

    # ex = '.../test_same/logs/test.in_100_5.0.log'
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
