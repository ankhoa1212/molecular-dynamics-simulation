# molecular-dynamics-simulation
This project uses LAMMPS to simulate molecular dynamics.

## Resources
- [LAMMPS Manual](https://docs.lammps.org/Manual.html)
- [Light-Responsive Assembly](https://pubs.acs.org/doi/10.1021/acs.jpcb.4c02301)
- [Molecular Dynamics Simulation of Active Particles Video](https://pubs.acs.org/doi/10.1021/acs.jpcb.4c02301)
- [Molecular Dynamics Simulation of Active Particles](https://arxiv.org/abs/2102.10399)
- [OVITO (for Visualization)](https://www.ovito.org/)

## Linux Setup
[Install](https://docs.lammps.org/Install.html) and [Build](https://docs.lammps.org/Build.html) LAMMPS.

### Temporary Environment Variable Setup
Use the following to set up env variables to run the ```lmp``` command (first navigate to the directory of the executable):

```EXECUTABLE_DIR=$PWD```

```export PATH=$EXECUTABLE_DIR:$PATH```

### Permanent Environment Variable Setup
Add the following to the ```~/.bashrc``` file (replace ```/path/to/dir``` with the path to the directory of the executable):

```export PATH=/path/to/dir:$PATH```

After modifying the file, run ```source ~/.bashrc``` to refresh bash shell and register the changes.
