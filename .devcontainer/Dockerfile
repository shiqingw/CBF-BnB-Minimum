# Make sure you are using the 'devel' image as we discussed
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

# Use bash shell
SHELL ["/bin/bash", "--login", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=non-interactive \
    CUDA_HOME=/usr/local/cuda-11.8 \
    PATH="/usr/local/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system packages, sorted and grouped for readability
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    \
    # --- Build Tools & Libraries ---
    build-essential \
    cm-super \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libglib2.0-dev \
    liblapack-dev \
    libopenblas-dev \
    libpoco-dev \
    \
    # --- System & CLI Utilities ---
    curl \
    gnupg-agent \
    htop \
    net-tools \
    software-properties-common \
    vim \
    \
    # --- Python ---
    python3-pip \
    python3-tk \
    \
    # --- Graphics/Display (X11/GL) ---
    dvipng \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libx11-6 \
    libxext6 \
    \
    # --- TeX Live ---
    texlive-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-latex-recommended \
    \
    # --- Cleanup ---
    && texhash \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dreal4
# This script will now work because gnupg-agent and software-properties-common are installed
RUN curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | bash

# Install Python packages
# Use --no-cache-dir to avoid caching and save a layer
RUN python -m pip install --no-cache-dir \
    dreal \
    ipykernel \
    matplotlib \
    ninja \
    numpy==1.26.4 \
    scipy==1.12.0 \
    torchinfo
    
# Install auto_LiRPA and alpha-beta-CROWN
RUN git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git \
    && cd alpha-beta-CROWN/auto_LiRPA \
    && pip3 install -e . \
    && cd .. \
    && cd complete_verifier \
    && pip3 install -r requirements.txt \
    && cd .. \
    && export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"

# Clean up
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache/pip

# Spin the container to keep it running
CMD ["tail", "-f", "/dev/null"]