#!/bin/bash

# 1) Remove old CUDA 11 toolkit (not drivers)
sudo apt remove --purge -y "cuda-11*" "cuda-toolkit-11*"
sudo apt autoremove -y
sudo rm -rf /usr/local/cuda-11*
sudo rm -rf /usr/local/cuda

# 2) Add CUDA  repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt update

# 3) Install CUDA toolkit
sudo apt install -y cuda-toolkit
sudo apt install -y nvidia-gds
# 4) Install gcc-12 and g++-12
sudo apt install -y gcc g++ build-essential

# 5) Clean old PATH lines in bashrc (basic delete)
sed -i '/cuda/d' ~/.bashrc
sed -i '/CUDA/d' ~/.bashrc
sed -i '/CC=/d' ~/.bashrc
sed -i '/CXX=/d' ~/.bashrc

# 6) Add CUDA  environment
echo 'export PATH=${PATH}:/usr/local/cuda-13.1/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CC=/usr/bin/gcc-12' >> ~/.bashrc
echo 'export CXX=/usr/bin/g++-12' >> ~/.bashrc

# Register gcc-12 and g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120

# Force-select gcc-12 and g++-12 without prompt
sudo update-alternatives --set gcc /usr/bin/gcc-12
sudo update-alternatives --set g++ /usr/bin/g++-12


# 7) Apply new environment to this session
source ~/.bashrc

echo ""
echo "Done. Close and reopen terminal or run: source ~/.bashrc"
echo "Check with: nvcc --version  |  gcc --version"