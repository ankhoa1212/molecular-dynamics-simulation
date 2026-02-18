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
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--no-show", action="store_true", help="Do not display the graph")
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

        if "." in filename:
             base = filename[:filename.rfind(".")]
        else:
             base = filename
        
        output_filename = f"{base}_hexatic_order.png"

        if args.output_dir:
            # Ensure output directory exists if provided
            if not os.path.exists(args.output_dir):
                 os.makedirs(args.output_dir)
            output = os.path.join(args.output_dir, output_filename)
        else:
            output = output_filename

        plt.savefig(output, dpi=300)
        print(f"Graph saved to {output}")
        if not args.no_show:
            plt.show()
