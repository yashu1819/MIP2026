# 1. Find the exact path where cuopt_c.h lives
export CUOPT_INCLUDE=$(find /opt /usr -name "cuopt_c.h" | sed 's|/cuopt/linear_programming/cuopt_c.h||')

# 2. Find the exact path where libcuopt.so lives
export CUOPT_LIB=$(dirname $(find /opt /usr -name "libcuopt.so"))

# 3. Compile with the corrected names
g++ -O3 cuoptHeuristic.cpp -o cuoptHeuristic \
    -I$CUOPT_INCLUDE \
    -L$CUOPT_LIB \
    -lcuopt -lcudart \
    -Wl,-rpath,$CUOPT_LIB
