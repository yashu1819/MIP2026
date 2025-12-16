import gurobipy as gp
from gurobipy import GRB

# Load MILP
i = 1
while i<=50:
    model=1
    if i>=10:
        model = gp.read(f"test_set/instances/instance_{i}.mps")
    else:
        model = gp.read(f"test_set/instances/instance_0{i}.mps")
        
    # Relax integrality (MILP → LP)
    relaxed = model.relax()

    # IMPORTANT: Do NOT optimize
    # Just write relaxed model to disk
    if i>=10:
        relaxed.write(f"test_set/relaxedInstances/relaxed_{i}.mps")
    else:
        relaxed.write(f"test_set/relaxedInstances/relaxed_0{i}.mps")
    i+=1
    # print("✅ Relaxed LP written to model_relaxed.mps")