import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from hexatic_order_analysis import parse_and_calc_hexatic


def extract_epsilon_and_molecules(filename):
    """
    Extracts the epsilon and molecules values from the filename.
    Expected format: *.in_{molecules}_{epsilon}.lammpstrj
    """
    match = re.search(r"\.in_(\d+)_([+-]?\d*\.\d+|\d+)\.(lammpstrj|log)", filename)
    if match:
        molecules = int(match.group(1))
        epsilon = float(match.group(2))
        return molecules, epsilon
    return None, None


def test_single_file(filename):
    """
    Test and display hexatic analysis results for a single lammpstrj file.
    """
    n_mols, eps = extract_epsilon_and_molecules(filename)

    if eps is None or n_mols is None:
        print(f"Could not extract parameters from filename: {filename}")
        return

    print(f"Analyzing: {filename}")
    print(f"  Molecules: {n_mols}")
    print(f"  Epsilon: {eps}")

    _, values = parse_and_calc_hexatic(filename)

    final_frame_psi6 = np.mean(values[-1])
    all_frames_mean = np.mean(values)

    print(f"  Final frame avg |psi6|: {final_frame_psi6:.4f}")
    print(f"  All frames avg |psi6|: {all_frames_mean:.4f}")
    print(f"  Total frames: {len(values)}")


def generate_stability_plot(data_dir, pattern, verbose):
    """Generate a phase diagram plot based on hexatic order analysis."""
    file_pattern = os.path.join(data_dir, pattern)
    filenames = glob.glob(file_pattern)

    epsilons = []
    num_molecules = []
    avg_psi6 = []
    if verbose:
        print(filenames)
    for fname in filenames:
        n_mols, eps = extract_epsilon_and_molecules(fname)
        if eps is None or n_mols is None:
            if verbose:
                print(fname)
            continue

        # Use your existing function to get timesteps and psi6 values
        # Assuming values is a 2D array: [timestep, particle_index]
        _, values = parse_and_calc_hexatic(fname, verbose)

        # We take the mean of the last frame to represent the "stable" state
        final_frame_psi6 = np.mean(values[-1])

        epsilons.append(eps)
        num_molecules.append(n_mols)
        avg_psi6.append(final_frame_psi6)

    # Get max values
    max_epsilon = max(epsilons) * 1.05  # add 5% buffer to graph
    max_molecules = max(num_molecules) * 1.05  # add 5% buffer to graph

    # Plotting
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        epsilons,
        num_molecules,
        c=avg_psi6,
        cmap="RdYlGn",
        s=100,
        edgecolor="black",
        alpha=0.8,
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(
        r"Average Hexatic Order Parameter $\langle |\psi_6| \rangle$", fontsize=12
    )

    plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=12)
    plt.title(
        r"Phase Behavior: $N$ vs $\epsilon$ colored by Hexatic Order", fontsize=14
    )
    plt.ylabel("Number of Molecules ($N$)", fontsize=12)
    plt.xlim(0, max_epsilon)
    plt.ylim(0, max_molecules)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"{os.path.basename(data_dir)}_hexatic_order_phase_diagram.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hexatic order phase diagram from LAMMPS trajectory files."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="/home/austin/git/molecular-dynamics-simulation/lammps-scripts/1.0_temp",
        help="Path to folder containing .lammpstrj files (default: ./data)",
    )
    parser.add_argument(
        "--pattern",
        default="*.in_*_*.lammpstrj",
        help="Filename pattern inside the folder (default: %(default)s)",
    )
    parser.add_argument(
        "--test", help="Test a single lammpstrj file instead of generating plot"
    )
    parser.add_argument("--verbose", default=0, help="Set to 1 to print out results")

    args = parser.parse_args()

    if args.test:
        test_single_file(args.test)
    else:
        if args.verbose:
            print(args.data_dir)
        generate_stability_plot(args.data_dir, args.pattern, verbose=args.verbose)
