#!/bin/bash

# run_all.sh
# Runs the solver on all instance_XX.mps files, shows normal output on screen,
# and collects results into a CSV file.

INSTANCE_DIR="../test_set/instances"
BINARY="./fp_s"
OUTPUT_CSV="results.csv"

# Header of the CSV
echo "instance_name,feasible,objective_value,time_taken_sec,status_message" > "$OUTPUT_CSV"

for i in $(seq -f "%02g" 1 50); do
    INSTANCE="instance_${i}.mps"
    FILEPATH="$INSTANCE_DIR/$INSTANCE"

    if [ ! -f "$FILEPATH" ]; then
        echo "Skipping missing file: $INSTANCE"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "Running instance: $INSTANCE"
    echo "================================================================"

    # Run the solver and capture output while also showing it live
    # tee sends output to both screen and a temp file
    TEMP_OUTPUT=$(mktemp)
    "$BINARY" "$FILEPATH" | tee "$TEMP_OUTPUT"

    # Parse the captured output
    FEASIBLE=0
    OBJECTIVE="N/A"
    TIME_SEC="N/A"
    STATUS="N/A"

    # Detect success
    if grep -q "SUCCESS!" "$TEMP_OUTPUT"; then
        FEASIBLE=1
    fi

    # Extract objective
    OBJ_LINE=$(grep "Objective :" "$TEMP_OUTPUT" | tail -n 1)
    if [ -n "$OBJ_LINE" ]; then
        OBJECTIVE=$(echo "$OBJ_LINE" | awk '{print $3}')
    fi

    # Extract time
    TIME_LINE=$(grep "Time taken:" "$TEMP_OUTPUT" | tail -n 1)
    if [ -n "$TIME_LINE" ]; then
        TIME_SEC=$(echo "$TIME_LINE" | awk '{print $3}')
    fi

    # Extract status message (last non-empty line or failure message)
    STATUS=$(grep -E "SUCCESS|Max iterations|Time limit reached|failed" "$TEMP_OUTPUT" | tail -n 1 | sed 's/^[[:space:]]*//')

    # Write to CSV
    echo "$INSTANCE,$FEASIBLE,$OBJECTIVE,$TIME_SEC,\"$STATUS\"" >> "$OUTPUT_CSV"

    # Clean up temp file
    rm -f "$TEMP_OUTPUT"

    echo ""
    echo "Finished $INSTANCE → feasible=$FEASIBLE, obj=$OBJECTIVE, time=$TIME_SEC s"
    echo ""
done

echo ""
echo "All done."
echo "Results saved to: $OUTPUT_CSV"
echo "You can open it with: cat $OUTPUT_CSV  or  xdg-open $OUTPUT_CSV"
