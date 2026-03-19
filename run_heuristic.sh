#!/bin/bash

# Check if correct number of arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <instance_name> <output_dir>"
    exit 1
fi

INSTANCE_NAME=$1
OUTPUT_DIR=$2
mkdir -p "$OUTPUT_DIR"
# Run your solver
 ./fj_solver "$INSTANCE_NAME" "$OUTPUT_DIR"
# ./fj_solver "$INSTANCE_NAME"
