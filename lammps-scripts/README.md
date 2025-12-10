## Usage

### Running Simulations with `run.sh`
The `run.sh` script automates running multiple LAMMPS simulations by sweeping over a range of molecule counts and epsilon values. The usage is:

```bash
./run.sh <input_file> <output_dir> <min_molecules> <max_molecules> <molecules_step> <min_epsilon> <max_epsilon> <epsilon_step>
```

- `<input_file>`: LAMMPS input script (e.g., `central_pair_interaction.in`)
- `<output_dir>`: Directory to store simulation outputs (e.g., `tests`)
- `<min_molecules>`: Minimum number of molecules to simulate
- `<max_molecules>`: Maximum number of molecules to simulate
- `<molecules_step>`: Step size for molecule count
- `<min_epsilon>`: Minimum epsilon value (interaction strength)
- `<max_epsilon>`: Maximum epsilon value
- `<epsilon_step>`: Step size for epsilon

The script will run simulations for each combination of molecule count and epsilon value in the specified ranges.
To run a LAMMPS simulation, use the following command format:

```bash
./run.sh <input_file> <output_dir> <min_molecules> <max_molecules> <molecules_step> <min_epsilon> <max_epsilon> <epsilon_step>
```

For example:

```bash
./run.sh central_pair_interaction.in tests 1000 2000 1000 5.0 10.0 5.0
```

---

### Plotting Results with `graph.py`

Install requirements

```bash
pip install -r requirements.txt
```

To generate plots from simulation output data, use:

```bash
python3 graph.py <filename>
```

For example:

```bash
python3 graph.py test.lammpstrj
```

- `<filename>`: Trajectory result file to plot data from

This will produce a plot of the specified property over time using the data in the given output directory.
