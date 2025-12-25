#!/bin/bash

# 1. Clean up existing CUDA/NVIDIA installations
echo "--- Cleaning up existing installations ---"
sudo apt-get purge -y "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
                      "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*"
sudo apt-get autoremove -y
sudo rm -rf /usr/local/cuda*

# 2. Install C++ 12 (GCC/G++ 12)
echo "--- Installing GCC/G++ 12 ---"
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install -y gcc-12 g++-12

# Set GCC 12 as the default choice
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12
sudo update-alternatives --set gcc /usr/bin/gcc-12

# 3. Install CUDA 12.4
echo "--- Installing CUDA 12.4 ---"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

# 4. Adding to Paths
echo "--- Updating .bashrc paths ---"
# Remove old CUDA paths if they exist to prevent duplicates
sed -i '/\/usr\/local\/cuda/d' ~/.bashrc

# Append new paths
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

echo "--- DONE! Please run 'source ~/.bashrc' and 'nvcc --version' to verify ---"