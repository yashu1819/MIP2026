#!/bin/bash

set -e

# -------------------------------------------------
# Source Files
# -------------------------------------------------
SRC_MAIN="main.cpp"
SRC_ACTIVITY="activity.cu"
SRC_SCORE="score.cpp"
SRC_EPSILON_SEARCH="epsilon_search.cpp"
SRC_REPAIR="repair.cpp"
SRC_RESULTS_FILE="results_file.cpp"


SRC_MIP="../mip_problem.cpp"
SRC_LP="../lp_relaxation.cpp"

BIN="general_rounding_solver"

INSTANCE_DIR="../../test_set/instances"

# -------------------------------------------------
# 1. Find cuOpt header
# -------------------------------------------------
echo "[1/5] Searching for cuOpt header (cuopt_c.h)..."

HEADER_FILE=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" 2>/dev/null | head -n 1)

if [ -z "$HEADER_FILE" ]; then
    echo "ERROR: cuopt_c.h not found"
    exit 1
fi

INCLUDE_PATH=$(echo "$HEADER_FILE" | sed 's/\/cuopt\/linear_programming\/cuopt_c.h//')

echo "Found cuOpt include path:"
echo "$INCLUDE_PATH"

# -------------------------------------------------
# 2. Find cuOpt library
# -------------------------------------------------
echo "[2/5] Searching for libcuopt.so..."

LIB_FILE=$(find / -name "libcuopt.so" 2>/dev/null | head -n 1)

if [ -z "$LIB_FILE" ]; then
    echo "ERROR: libcuopt.so not found"
    exit 1
fi

LIBCUOPT_LIB_DIR=$(dirname "$LIB_FILE")

echo "Found cuOpt library path:"
echo "$LIBCUOPT_LIB_DIR"

# -------------------------------------------------
# 3. Export runtime library path
# -------------------------------------------------
export LD_LIBRARY_PATH=$LIBCUOPT_LIB_DIR:$LD_LIBRARY_PATH

# -------------------------------------------------
# 4. Compile
# -------------------------------------------------
echo "[3/5] Compiling General Rounding Heuristic..."

nvcc \
  -std=c++17 \
  -O3 -g \
  -I"$INCLUDE_PATH" \
  -L"$LIBCUOPT_LIB_DIR" \
  $SRC_MAIN \
  $SRC_ACTIVITY \
  $SRC_SCORE \
  $SRC_EPSILON_SEARCH \
  $SRC_REPAIR \
  $SRC_MIP \
  $SRC_LP \
  -lCoinUtils -lClp -lOsiClp -lOsi \
  -lcuopt \
  -Xlinker -rpath,"$LIBCUOPT_LIB_DIR" \
  -o $BIN

echo "Compilation finished."
echo "[4/5] Running Instances..."

SOLVER="./$BIN"

for i in $(seq -f "%02g" 2 2)
do

  echo "Running Instance: instance_$i.mps"

  $SOLVER $INSTANCE_DIR/instance_$i.mps

done

echo "[5/5] Build successful"
echo "Binary: ./$BIN"