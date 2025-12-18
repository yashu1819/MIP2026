import gurobipy as gp
from gurobipy import GRB
import pandas as pd

results = []

i = 0
while i <= 50:
    i += 1

    if i >= 10:
        fname = f"test_set/relaxedInstances/relaxed_{i}.mps"
        solname = f"test_set/relaxSol/sol{i}.txt"
    else:
        fname = f"test_set/relaxedInstances/relaxed_0{i}.mps"
        solname = f"test_set/relaxSol/sol0{i}.txt"

    model = gp.read(fname)

    # ---- time limit: 3 minutes ----
    model.setParam("TimeLimit", 180)
    # model.setParam("Method", 2)   # barrier
    model.setParam("Crossover", 0)

    model.optimize()

    runtime = model.Runtime
    status = model.Status

    # incumbent (best primal)
    if model.SolCount > 0:
        obj = model.ObjVal
        solved_optimal = (status == GRB.OPTIMAL)
    else:
        obj = None
        solved_optimal = False

    # ---- write solution file ----
    with open(solname, "w") as f:
        f.write(f"Status: {status}\n")
        f.write(f"Solved to optimality: {solved_optimal}\n")
        f.write(f"Objective value: {obj}\n")
        f.write(f"Solve time (seconds): {runtime}\n\n")
        f.write("Variable values:\n")

        if model.SolCount > 0:
            for v in model.getVars():
                f.write(f"{v.VarName} {v.X}\n")

    # ---- dataframe entry ----
    results.append({
        "instance": fname,
        "objective": obj,
        "runtime_sec": runtime,
        "optimal": solved_optimal
    })

# ---- dataframe ----
df = pd.DataFrame(results)
df.to_csv("test_set/relaxSol/summary.csv", index=False)

print(df)
