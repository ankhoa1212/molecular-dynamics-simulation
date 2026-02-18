"""
This module contains functions to parse LAMMPS trajectory files and calculate the
hexatic order parameter using the freud library.
"""

import argparse

import freud
import matplotlib.pyplot as plt
import numpy as np


def _read_box_bounds(f):
    """
    Reads box bounds from the file stream and returns a freud Box.
    """
    x_range = [float(x) for x in f.readline().split()]
    y_range = [float(y) for y in f.readline().split()]
    _ = f.readline()  # Skip z_range

    box_lx = x_range[1] - x_range[0]
    box_ly = y_range[1] - y_range[0]
    return freud.Box(Lx=box_lx, Ly=box_ly, is2D=True)


def _read_atoms(f, n_atoms):
    """
    Reads atom data from the file stream.
    Returns positions array (Nx3).
    """
    data = []
    for _ in range(n_atoms):
        data.append(f.readline().split())

    data = np.array(data, dtype=float)
    positions = data[:, 2:4]  # Grab x and y columns (index 2 and 3)
    # Add a zero z-column for freud compatibility
    return np.column_stack((positions, np.zeros(n_atoms)))


def parse_and_calc_hexatic(filename, verbose=1):
    """
    Parses a LAMMPS trajectory file and calculates the hexatic order parameter.

    Args:
        filename (str): Path to the LAMMPS trajectory file.
        verbose (int): Verbosity level (default: 1).

    Returns:
        tuple: A tuple containing two lists: steps and mean psi6 values.
    """
    steps = []
    psi6_means = []

    # Initialize variables
    n_atoms = 0
    current_box = None
    step = 0

    with open(filename, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                step = int(f.readline())
                steps.append(step)

            elif "ITEM: NUMBER OF ATOMS" in line:
                n_atoms = int(f.readline())

            elif "ITEM: BOX BOUNDS" in line:
                current_box = _read_box_bounds(f)

            elif "ITEM: ATOMS" in line:
                if current_box is None:
                    # Skip if box not defined (should not happen in valid dump)
                    _read_atoms(f, n_atoms)
                    continue

                positions = _read_atoms(f, n_atoms)

                # Calculate Hexatic Order
                hex_comp = freud.order.Hexatic(k=6)
                hex_comp.compute(
                    system=(current_box, positions), neighbors={"num_neighbors": 6}
                )

                # Magnitude of psi6 for each atom
                mag_psi6 = np.abs(hex_comp.particle_order)
                mean_psi6 = np.mean(mag_psi6)
                psi6_means.append(mean_psi6)

                if verbose:
                    print(f"Step {step}: Avg |psi6| = {mean_psi6:.4f}")

    return steps, psi6_means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute hexatic order from a LAMMPS dump file."
    )
    parser.add_argument(
        "filename", nargs="?", default="dump.lammps", help="Path to LAMMPS dump file"
    )
    args = parser.parse_args()

    timesteps, values = parse_and_calc_hexatic(args.filename)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, values, "o-", color="#2c3e50")
    plt.xlabel("Timestep")
    plt.ylabel(r"Average Hexatic Order Parameter $\langle|\psi_6|\rangle$")
    plt.title("Hexatic Order over Time")
    plt.grid(True)
    plt.show()
