import argparse
import os

import matplotlib.pyplot as plt
from hexatic_order_analysis import parse_and_calc_hexatic
from phase_diagram import extract_epsilon_and_molecules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Hexatic Order Parameter from LAMMPS trajectory."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="Specific LAMMPS trajectory file to process. If not provided, processes all in results/.",
    )
    args = parser.parse_args()

    if args.filename:
        filepath = args.filename

        plt.figure(figsize=(10, 6))
        filename = os.path.basename(filepath)

        n_molecules, eps = extract_epsilon_and_molecules(filename)

        frames, psi6 = parse_and_calc_hexatic(filepath)

        label = f"N={n_molecules}, ε={eps}"
        plt.plot(frames, psi6, label=label, alpha=0.7)

        plt.xlabel("Frame")
        plt.ylabel(r"Global Hexatic Order $|\Psi_6|$")
        plt.title(f"Hexatic Order Parameter: N={n_molecules}, ε={eps}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output = (
            f"{filename[:filename.find('.')]}_{n_molecules}_{eps}_hexatic_order.png"
        )
        plt.savefig(output, dpi=300)
        print(f"Graph saved to {output}")
        plt.show()
