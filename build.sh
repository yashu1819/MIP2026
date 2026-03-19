#!/bin/bash


sudo apt update
sudo apt-get -y install zlib1g-dev
sudo apt install -y coinor-libcbc-dev 
sudo apt install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev


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

SRC_MAIN="src/main.cpp"
SRC_MIP="src/mip_problem.cpp"
SRC_LP="src/lp_relaxation.cpp"
BIN="mip_lp_solver"

# -----------------------------------------
# 1. Find cuOpt header
# -----------------------------------------
echo "[1/4] Searching for cuOpt header (cuopt_c.h)..."

HEADER_FILE=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" 2>/dev/null | head -n 1)

if [ -z "$HEADER_FILE" ]; then
    echo "ERROR: cuopt_c.h not found"
    exit 1
fi

# From: .../include/cuopt/linear_programming/cuopt_c.h
# To:   .../include
INCLUDE_PATH=$(echo "$HEADER_FILE" | sed 's/\/cuopt\/linear_programming\/cuopt_c.h//')

echo "Found cuOpt include path: $INCLUDE_PATH"

# -----------------------------------------
# 2. Find cuOpt library
# -----------------------------------------
echo "[2/4] Searching for libcuopt.so..."

LIB_FILE=$(find / -name "libcuopt.so" 2>/dev/null | head -n 1)

if [ -z "$LIB_FILE" ]; then
    echo "ERROR: libcuopt.so not found"
    exit 1
fi

LIBCUOPT_LIB_DIR=$(dirname "$LIB_FILE")

echo "Found cuOpt library path: $LIBCUOPT_LIB_DIR"

# -----------------------------------------
# 3. Export runtime library path (optional)
# -----------------------------------------
export LD_LIBRARY_PATH=$LIBCUOPT_LIB_DIR:$LD_LIBRARY_PATH

# -----------------------------------------
# 4. Compile
# -----------------------------------------
echo "[3/4] Compiling."

nvcc \
  -std=c++17 \
  -O3 -g -G \
-I"$INCLUDE_PATH" \
    -L"$LIBCUOPT_LIB_DIR" \
  src/main.cpp  src/mip_problem.cpp src/feasibility_jump.cu src/lp_relaxation.cpp \
  -lCoinUtils -lClp -lOsiClp -lOsi \
 -lcuopt \
    -Xlinker -rpath,"$LIBCUOPT_LIB_DIR" \
  -o fj_solver
