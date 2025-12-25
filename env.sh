#!/bin/bash

echo "[1] Removing old CUDA installations (only if present)..."
sudo apt remove --purge -y "cuda*" "nvidia-cuda*" "libcudnn*" >/dev/null 2>&1
sudo apt autoremove -y
sudo rm -rf /usr/local/cuda /usr/local/cuda-*


echo "[2] Removing old or conflicting PATH entries from ~/.bashrc..."
# remove any previous cuda/gcc exports
sed -i '/cuda/d' ~/.bashrc
sed -i '/CUDA/d' ~/.bashrc
sed -i '/gcc-1/d' ~/.bashrc
sed -i '/g++-1/d' ~/.bashrc
sed -i '/^export CC/d' ~/.bashrc
sed -i '/^export CXX/d' ~/.bashrc
sed -i '/LD_LIBRARY_PATH.*cuda/d' ~/.bashrc


echo "[3] Adding CUDA 12.2 repository..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb
sudo dpkg -i cuda-keyring.deb
sudo apt update -y


echo "[4] Installing CUDA 12.2 Toolkit..."
sudo apt install -y cuda-12-2 cuda-toolkit-12-2


echo "[5] Installing correct host compilers (gcc-12 + g++-12)..."
sudo apt install -y gcc-12 g++-12


echo "[6] Setting environment variables in ~/.bashrc..."
cat << 'EOF' >> ~/.bashrc

# ---- CUDA 12.2 Stable Environment ----
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
EOF


echo "[7] Applying environment to current session..."
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12


echo ""
echo "=============================================="
echo "Cleanup + CUDA 12.2 setup complete."
echo "Run these to verify:"
echo "  nvcc --version     # should show CUDA 12.2"
echo "  gcc --version      # should show gcc 12.x"
echo "  which nvcc         # should point to /usr/local/cuda-12.2/bin/nvcc"
echo "=============================================="
echo ""
