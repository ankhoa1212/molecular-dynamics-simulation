import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from phase_diagram import extract_epsilon_and_molecules


def plot_velocity_over_time(filename, output_dir, no_show=False):
    timesteps = []
    avg_velocities = []
    std_velocities = []

    current_velocities = []
    reading_atoms = False

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                # If we just finished a frame, calculate the average
                if current_velocities:
                    avg_velocities.append(np.mean(current_velocities))
                    std_velocities.append(np.std(current_velocities))
                    current_velocities = []
                reading_atoms = False
                timesteps.append(int(next(f)))

            elif line.startswith("ITEM: ATOMS"):
                reading_atoms = True
                # Identify column indices for vx and vy
                header = line.split()[2:]
                vx_idx = header.index("vx")
                vy_idx = header.index("vy")
                continue

            elif reading_atoms:
                parts = line.split()
                if not parts:
                    continue
                vx = float(parts[vx_idx])
                vy = float(parts[vy_idx])
                # Calculate magnitude: sqrt(vx^2 + vy^2)
                v_mag = np.sqrt(vx**2 + vy**2)
                current_velocities.append(v_mag)

        # Append the final frame
        if current_velocities:
            avg_velocities.append(np.mean(current_velocities))
            std_velocities.append(np.std(current_velocities))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        timesteps,
        avg_velocities,
        yerr=std_velocities,
        marker="o",
        linestyle="-",
        color="b",
        capsize=5,
    )
    plt.xlabel("Timestep")
    plt.ylabel("Mean Velocity Magnitude")
    plt.title("Average Velocity Magnitude Over Time")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Generate output filename
    base_name = os.path.basename(filename)
    if "." in base_name:
        base_name = base_name[:base_name.rfind(".")]
    
    output_filename = f"{base_name}_velocity_graph.png"
    
    if output_dir:
        output = os.path.join(output_dir, output_filename)
    else:
        output = output_filename

    plt.savefig(output)
    print(f"Graph saved to {output}")
    if not no_show:
        plt.show()


# Replace 'your_file.lammpstrj' with your actual filename
# plot_velocity_over_time('your_file.lammpstrj')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot velocity over time from LAMMPS trajectory file"
    )
    parser.add_argument(
        "--filename",
        "-f",
        default="/home/austin/git/molecular-dynamics-simulation/lammps-scripts/test_same/test.in_100_5.0.lammpstrj",
        help="Path to the LAMMPS trajectory file",
    )
    parser.add_argument(
        "--output_dir", default=None, help="output directory to save graph file to"
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the graph")
    args = parser.parse_args()

    plot_velocity_over_time(args.filename, args.output_dir, args.no_show)
