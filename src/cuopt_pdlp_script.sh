#!/bin/bash
# Build and run the example



# Find the cuopt header file and assign to INCLUDE_PATH
INCLUDE_PATH=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" -printf "%h\n" | sed 's/\/linear_programming//' 2>/dev/null)
# Find the libcuopt library and assign to LIBCUOPT_LIBRARY_PATH
LIBCUOPT_LIBRARY_PATH=$(find / -name "libcuopt.so" 2>/dev/null)
gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o src/cuopt_pdlp src/cuopt_pdlp.c -lcuopt
