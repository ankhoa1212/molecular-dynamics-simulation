# molecular-dynamics-simulation
This project uses LAMMPS to simulate molecular dynamics.

## Linux Setup
[Install](https://docs.lammps.org/Install.html) and [Build](https://docs.lammps.org/Build.html) LAMMPS.

### Temporary Environment Variable Setup
Use the following to set up env variables in the commandline to run the ```lmp``` command (first navigate to the directory of the executable):

```
EXECUTABLE_DIR=$PWD
```

```
export PATH=$EXECUTABLE_DIR:$PATH
```

### Bash Shell Environment Variable Setup
Add the following to the ```~/.bashrc``` file (replace ```/path/to/dir``` with the path to the directory of the executable):

```
export PATH=/path/to/dir:$PATH
```

#### Setup Ovito

If on an old Linux Ubuntu version less than 22.04 that does not have ```qt.qpa.plugin 6.5.0``` may need to install ```libxcb-cursor0```

## Docker Setup (Work in progress)
[Install](https://www.docker.com/get-started/) Docker.

Pull docker image:
```
docker pull ankhoa1212/lammps-simulations:latest
```

## Resources
- [LAMMPS Manual](https://docs.lammps.org/Manual.html)
  - [Installation](https://docs.lammps.org/Install.html)
  - [Examples](https://docs.lammps.org/Examples.html)
  - [Processing Tools](https://docs.lammps.org/Tools.html)
  - [Library Interfaces](https://docs.lammps.org/Library.html)
  - [Modifying and extending LAMMPS](https://docs.lammps.org/Modify.html)
- [Light-Responsive Assembly](https://pubs.acs.org/doi/10.1021/acs.jpcb.4c02301)
- [Molecular Dynamics Simulation of Active Particles Video](https://www.youtube.com/watch?v=wsM2kUB6XU4&ab_channel=SoftMatterLab)
- [Molecular Dynamics Simulation of Active Particles (Brownian Motion)](https://arxiv.org/abs/2102.10399)
- [OVITO (for Visualization)](https://www.ovito.org/)
