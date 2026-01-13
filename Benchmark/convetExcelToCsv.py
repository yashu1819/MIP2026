import pandas as pd
df = pd.read_excel("Relaxation Benchmark Solution.xlsx", sheet_name="Sheet1")
df.to_csv("relaxationResults.csv", index=False)
