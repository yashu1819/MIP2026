import os
import pandas as pd

sol_dir = "test_set/relaxSol"
results = []

for fname in sorted(os.listdir(sol_dir)):
    if not fname.startswith("sol") or not fname.endswith(".txt"):
        continue

    path = os.path.join(sol_dir, fname)

    with open(path, "r") as f:
        lines = f.readlines()
                                     
    # defaults
    objective = None
    runtime = None
    optimal = False

    for line in lines:
        line = line.strip()

        if line.startswith("Objective value:"):
            val = line.split("Objective value:")[1].strip()
            objective = None if val == "None" else float(val)

        elif line.startswith("Solve time (seconds):"):
            runtime = float(line.split("Solve time (seconds):")[1].strip())

        elif line.startswith("Solved to optimality:"):
            optimal = line.split("Solved to optimality:")[1].strip() == "True"

    results.append({
        "solution_file": fname,
        "objective": objective,
        "runtime_sec": runtime,
        "optimal": optimal
    })

# build dataframe
df = pd.DataFrame(results)

# sort by instance number (important)
df["instance_id"] = df["solution_file"].str.extract(r"(\d+)").astype(int)
df = df.sort_values("instance_id").drop(columns="instance_id")

# save
df.to_csv("test_set/relaxSol/summary.csv", index=False)

print(df)
