SRC_MAIN="main.cpp"
SRC_MIP="mip_problem.cpp"
SRC_LP="lp_relaxation.cpp"
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
echo "[3/4] Compiling..."

#g++ -std=c++17 -O0 \
#    -I"$INCLUDE_PATH" \
#    -L"$LIBCUOPT_LIB_DIR" \
#    main.cpp mip_problem.cpp lp_relaxation.cpp feasibility_pump.cpp \
#    -lClp -lOsiClp -lCoinUtils \
#    -lcuopt \
#    -Wl,-rpath,"$LIBCUOPT_LIB_DIR" \
#    -o mip_lp_solver

nvcc \
  -std=c++17 \
  -O3 -g -G \
-I"$INCLUDE_PATH" \
    -L"$LIBCUOPT_LIB_DIR" \
  main.cpp  mip_problem.cpp feasibility_jump.cu lp_relaxation.cpp \
  -lCoinUtils -lClp -lOsiClp -lOsi \
 -lcuopt \
    -Xlinker -rpath,"$LIBCUOPT_LIB_DIR" \
  -o fj_solver
SOLVER="./fj_solver"
INSTANCE_DIR="../test_set/instances"

# Loop from 1 to 50
for i in $(seq -f "%02g" 1 50)
do
   # echo "--------------------------------------------------"
#    echo "Running Instance: instance_$i.mps"
    
    # Execute the command
 #   $SOLVER $INSTANCE_DIR/instance_$i.mps
    
    # Optional: Check if the solver exited successfully
   # if [ $? -ne 0 ]; then
    #    echo "Error: Solver failed on instance_$i.mps"
   # fi
done

echo "[4/4] Build successful"
#echo "Binary: ./mip_lp_solver"

