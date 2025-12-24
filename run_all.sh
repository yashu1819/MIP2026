#!/bin/bash

# 1. Define the directories
RESULTS_DIR="results_pdlp_1e-06"
LOGS_DIR="logs_pdlp_1e-06"

# 2. Set the library paths
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 3. Create the directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

echo "Batch solve started."
echo "Solutions: $RESULTS_DIR"
echo "Logs:      $LOGS_DIR"
echo "----------------------------------------------------"

# 4. Loop through the MPS files
for mps_file in test_set/relaxedInstances/*.mps; do
    base_name=$(basename "$mps_file" .mps)
    output_path="$RESULTS_DIR/${base_name}_sol.txt"
    log_path="$LOGS_DIR/${base_name}.log"
    
    echo "Processing: $base_name"
    
    # Run the solver:
    # '>' redirects standard output to the log file
    # '2>&1' redirects errors (stderr) to the same log file
    ./mps_solver "$mps_file" "$output_path" > "$log_path" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Done: $base_name"
    else
        echo "Failed: $base_name (Check $log_path for details)"
    fi
done

echo "----------------------------------------------------"
echo "Batch processing complete."
