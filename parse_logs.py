import glob
import csv
import re

# 1) Find all relaxed_*.log files
files = sorted(glob.glob("logs_pdlp_1e-06/relaxed_*.log"))

# 2) Prepare CSV output
with open("summary.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "optimal_objective", "time"])  # header

    for file in files:
        with open(file, "r") as f:
            content = f.read()

            # 3) Regex to extract Optimal objective
            obj_match = re.search(r"Objective = ([\-\d\.]+)", content)

            # 4) Regex to extract Time (last Time: value in seconds)
            time_match = re.search(r"Time: ([\d\.]+)s", content)

            objective = obj_match.group(1) if obj_match else ""
            time = time_match.group(1) if time_match else ""

            # 5) Write row
            writer.writerow([file, objective, time])

print("CSV created successfully: summary.csv")
