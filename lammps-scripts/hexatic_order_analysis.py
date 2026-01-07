import argparse

import freud
import matplotlib.pyplot as plt
import numpy as np


def parse_and_calc_hexatic(filename, verbose=1):
    steps = []
    psi6_means = []

    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                step = int(f.readline())
                steps.append(step)

            if "ITEM: BOX BOUNDS" in line:
                # Read x, y, z bounds
                x_range = [float(x) for x in f.readline().split()]
                y_range = [float(y) for y in f.readline().split()]
                z_range = [float(z) for z in f.readline().split()]
                Lx = x_range[1] - x_range[0]
                Ly = y_range[1] - y_range[0]
                # Create a 2D box for freud
                current_box = freud.Box(Lx=Lx, Ly=Ly, is2D=True)
            if "ITEM: NUMBER OF ATOMS" in line:
                n_atoms = int(f.readline())

            if "ITEM: ATOMS" in line:
                # Identify column indices (id type x y vx vy fx fy v_dist)
                # x is index 2, y is index 3
                data = []
                for _ in range(n_atoms):
                    data.append(f.readline().split())

                data = np.array(data, dtype=float)
                positions = data[:, 2:4]  # Grab x and y columns
                # Add a zero z-column for freud compatibility
                positions = np.column_stack((positions, np.zeros(n_atoms)))

                # Calculate Hexatic Order
                hex_comp = freud.order.Hexatic(k=6)
                hex_comp.compute(
                    system=(current_box, positions), neighbors={"num_neighbors": 6}
                )

                # Magnitude of psi6 for each atom
                mag_psi6 = np.abs(hex_comp.particle_order)
                psi6_means.append(np.mean(mag_psi6))
                if verbose:
                    print(f"Step {step}: Avg |psi6| = {psi6_means[-1]:.4f}")

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
