import pandas as pd
df= pd.read_csv("lp_comparison.csv")
for i in range(50):
    if (df['rel_diff'].iloc[i]>0.01):
        print(i+1," abs_dff= ",f"{df['abs_diff'].iloc[i]:.2e}"," current_obj= ",f"{df['lp_obj'].iloc[i]:.2e}"," benchmark_obj= ",f"{df['bench_obj'].iloc[i]:.2e}"  )