# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    curl \
    libfftw3-dev \
    libjpeg-dev \
    libpng-dev \
    libgmp-dev \
    libblas-dev \
    liblapack-dev \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a shallow copy of LAMMPS repository by cloning only a subset of the git history
RUN git clone -b release https://github.com/lammps/lammps.git --depth=1000

# Set the working directory
WORKDIR /lammps/build

# Compile LAMMPS using CMake
RUN cmake ../cmake \
    -D BUILD_MPI=no \
    -D BUILD_OMP=yes \
    -D CMAKE_INSTALL_PREFIX="/usr/local/lammps"

# Compile in parallel and copy compiled files into installation location
RUN make -j$(nproc) && \
    make install

# Add LAMMPS to PATH
ENV PATH="/usr/local/lammps/bin:$PATH"

# Set variable to turn on OpenMP support at runtime
ENV OMP_NUM_THREADS=$(nproc)

# Set default command to run LAMMPS
CMD ["lmp", "-h"]
