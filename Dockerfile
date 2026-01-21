# Use the base CUDA image with CUDA 11.8
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Install CUDA compatibility libraries
RUN apt-get update && apt-get install -y cuda-compat-11-8

# Set the environment variables for CUDA paths
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH


# Prevent interactive prompts for tzdata and other packages
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for Matplotlib and OpenGL
ENV MPLBACKEND=TkAgg

# Install Python, pip, and other required dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    x11-apps \
    python3-tk \
    tzdata \
    xvfb \
    tigervnc-standalone-server \
    openmpi-bin \
    libopenmpi-dev \
    cmake \
    zlib1g-dev \
    && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Downgrade pip to a version <24.1 to avoid metadata issues with gym==0.21.0
RUN pip3 install pip==23.0.1

# Install PyTorch compatible with CUDA 11.8
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the working directory
WORKDIR /tf/rl_project

# Copy the requirements.txt first for better caching of the pip install step
COPY requirements.txt /tf/rl_project/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Copy the rest of the project files to the container
COPY . /tf/rl_project

# Copy the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["entrypoint.sh"]

# Default command (can be overridden)
CMD ["python3", "scripts/run_simulation.py"]


