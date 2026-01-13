import os
import re
import pandas as pd

# Paths
log_dir = os.path.join( "l40s_pdlp_logs_1e-6")
output_file = os.path.join( "l40s1e-6PDLPresults.csv")

# Regex patterns
time_pattern = re.compile(r"Time:\s*([0-9.]+)s")
obj_pattern = re.compile(r"Objective:\s*([-+0-9.eE]+)")

results = []

for i in range(1, 51):
    log_name = f"relaxed_{i:02d}.log"
    log_path = os.path.join(log_dir, log_name)

    with open(log_path, "r") as f:
        content = f.read()

    time_match = time_pattern.search(content)
    obj_match = obj_pattern.search(content)

    if time_match and obj_match:
        time_val = float(time_match.group(1))
        obj_val = float(obj_match.group(1))
    else:
        time_val = None
        obj_val = None

    results.append({
        "instance": f"relaxed_{i:02d}",
        "time_sec": time_val,
        "objective": obj_val
    })

# Create table
df = pd.DataFrame(results)

# Save
df.to_csv(output_file, index=False)

print(f"Results written to {output_file}")
