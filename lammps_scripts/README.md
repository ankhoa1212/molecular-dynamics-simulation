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

**Arguments:**
- `<input_file>`: LAMMPS input script (e.g., `central_pair_interaction.in`)
- `<output_dir>`: Directory to store simulation outputs (e.g., `tests`)
- `<nsteps>`: Number of simulation steps (e.g., `1000`)
- `<dump_freq>`: Frequency of dumping trajectory data (e.g., `2000`)
- `<thermo_freq>`: Frequency of thermo output (e.g., `500`)
- `<temp>`: Simulation temperature (e.g., `5.0`)
- `<density>`: Particle density (e.g., `10.0`)
- `<cutoff>`: Interaction cutoff distance (e.g., `2.0`)

---

### Plotting Results with `graph.py`

To generate plots from simulation output data, use:

```bash
python3 graph.py <output_dir> <property>
```

For example:

```bash
python3 graph.py tests energy
```

- `<output_dir>`: Directory containing simulation outputs (e.g., `tests`)
- `<property>`: Property to plot (e.g., `energy`, `temperature`, `pressure`)

This will produce a plot of the specified property over time using the data in the given output directory.
