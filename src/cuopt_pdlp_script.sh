#!/bin/bash

# 1. Find the header file
HEADER_FILE=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" 2>/dev/null | head -n 1)

if [ -z "$HEADER_FILE" ]; then
    echo "Error: cuopt_c.h not found."
    exit 1
else
    # FIX: Move up TWO levels from the file to reach the base 'include' directory
    # From: .../include/cuopt/linear_programming/cuopt_c.h
    # To:   .../include
    INCLUDE_PATH=$(echo "$HEADER_FILE" | sed 's/\/cuopt\/linear_programming\/cuopt_c.h//')
fi

# 2. Find the library directory
LIB_FILE=$(find / -name "libcuopt.so" 2>/dev/null | head -n 1)

if [ -z "$LIB_FILE" ]; then
    echo "Error: libcuopt.so not found."
    exit 1
else
    LIBCUOPT_LIB_DIR=$(dirname "$LIB_FILE")
fi

echo "Include Path: $INCLUDE_PATH"
echo "Library Dir:  $LIBCUOPT_LIB_DIR"
export LD_LIBRARY_PATH=$LIBCUOPT_LIB_DIR:$LD_LIBRARY_PATH
# Note: Using quotes around paths is safer in case there are spaces
gcc -I"$INCLUDE_PATH" -L"$LIBCUOPT_LIB_DIR" -o cuopt_pdlp cuopt_pdlp.c -lcuopt
for i in $(seq -w 1 50); do
    MPS="../test_set/relaxedInstances/relaxed_${i}.mps"
    LOG="pdlp_logs_1e-6/relaxed_${i}.log"
    OUT="pdlp_sols_1e-6/relaxed_${i}.sol"
    echo "Running $MPS"
    ./cuopt_pdlp "$MPS" "$OUT" > "$LOG" 2>&1
done

