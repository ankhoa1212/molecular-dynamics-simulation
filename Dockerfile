# Use the official Ubuntu base image
FROM ubuntu:20.04

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

# Set working directory
WORKDIR /lammps

# Clone the LAMMPS repository
RUN git clone -b stable https://github.com/lammps/lammps.git .

# Create a build directory and compile LAMMPS
RUN mkdir build && cd build && \
    cmake ../cmake -D BUILD_MPI=no -D BUILD_OMP=yes -D CMAKE_INSTALL_PREFIX=/usr/local/lammps && \
    make -j$(nproc) && make install

# Add LAMMPS to PATH
ENV PATH="/usr/local/lammps/bin:$PATH"

# Set default command to run LAMMPS
CMD ["lmp", "-h"]