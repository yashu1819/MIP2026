#!/bin/bash


sudo apt update
sudo apt-get -y install zlib1g-dev
sudo apt install -y coinor-libcbc-dev 
sudo apt install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev


export PATH=${PATH}:/usr/local/cuda-13.0/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-13.0/lib64
source ~/.bashrc  

# 4) Install gcc-12 and g++-12
sudo apt install -y gcc-12 g++-12 build-essential
echo 'export CC=/usr/bin/gcc-12' >> ~/.bashrc
echo 'export CXX=/usr/bin/g++-12' >> ~/.bashrc

# Register gcc-12 and g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120

# Force-select gcc-12 and g++-12 without prompt
sudo update-alternatives --set gcc /usr/bin/gcc-12
sudo update-alternatives --set g++ /usr/bin/g++-12
sudo  apt update
# 7) Apply new environment to this session
source ~/.bashrc


pip install --extra-index-url=https://pypi.nvidia.com \
  'nvidia-cuda-runtime==13.0.*' \
  'cuopt-server-cu13==26.2.*' \
  'cuopt-sh-client==26.2.*'

