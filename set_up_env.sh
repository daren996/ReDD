#!/bin/bash

# Function to check if Conda is already installed
function check_conda_installed() {
    which conda >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Conda is already installed."
        return 1
    else
        echo "Conda is not installed, proceeding with installation."
        return 0
    fi
}

check_conda_installed
if [ $? -eq 0 ]; then
    sudo apt-get update
    sudo apt-get install curl

    cd /tmp
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh

    conda create -name trustudms python=3.9
    conda activate trustudms
else
    echo "Installation skipped."
fi

# Check if Conda environment exists
env_exists=$(conda info --envs | grep 'trustudms' || true)

if [ -z "$env_exists" ]; then
    echo "Environment 'trustudms' does not exist. Creating..."
    conda create -n trustudms python=3.9 -y
else
    echo "Environment 'trustudms' already exists. No action needed."
fi

source activate base
conda activate trustudms


# Install PyTorch according to the CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION=12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
else
    conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
fi


# Install screen
if ! command -v screen &> /dev/null; then
    echo "screen not installed. Installing ..."
    if command -v apt &> /dev/null; then
        sudo apt update
        sudo apt install -y screen
    elif command -v yum &> /dev/null; then
        sudo yum install -y screen
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y screen
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm screen
    else
        echo "Package manager not found. Please install screen manually."
        exit 1
    fi
    echo "screen installed."
else
    echo "screen already installed."
fi

