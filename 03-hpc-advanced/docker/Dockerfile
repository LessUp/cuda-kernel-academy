# HPC-AI-Optimization-Lab Development Environment
# Based on CUDA 13.1 with Ubuntu 22.04
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-venv \
    vim \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch \
    numpy \
    pytest \
    matplotlib \
    pandas

# Set up environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create workspace directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
