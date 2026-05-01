"""
Shared utilities for parsing LAMMPS dump files.
"""


def parse_lammps_dump(filename):
    """
    Generator that yields simulation frames from a LAMMPS dump file.
    
    Yields:
        dict: containing:
            - timestep (int)
            - n_atoms (int)
            - box_bounds (list of strings)
            - atoms (list of strings or generator)
    """
    with open(filename, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                # Read timestep
                try:
                    timestep = int(f.readline().strip())
                except ValueError:
                    continue

                # Read number of atoms
                line = f.readline()
                while line and "ITEM: NUMBER OF ATOMS" not in line:
                    line = f.readline()
                if not line:
                    break
                n_atoms = int(f.readline().strip())

                # Read box bounds
                line = f.readline()
                while line and "ITEM: BOX BOUNDS" not in line:
                    line = f.readline()
                if not line:
                    break
                box_lines = [f.readline() for _ in range(3)]

                # Read atoms
                line = f.readline()
                while line and "ITEM: ATOMS" not in line:
                    line = f.readline()
                if not line:
                    break

                atom_header = line.strip()
                atom_lines = [f.readline() for _ in range(n_atoms)]

                yield {
                    "timestep": timestep,
                    "n_atoms": n_atoms,
                    "box_bounds": box_lines,
                    "atom_header": atom_header,
                    "atoms": atom_lines,
                }
