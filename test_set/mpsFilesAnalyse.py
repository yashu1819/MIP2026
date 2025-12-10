import pandas as pd 
import numpy as np
df= pd.read_csv("test_set/milp_analysis_results.csv")
numeric_df = df.select_dtypes(include=np.number)
numeric_df['only integer']= numeric_df['Integer Variables']-  numeric_df['Binary Variables']

# Calculate the mean of each numeric column
mean_values = numeric_df.mean().rename('Mean')

min_values = numeric_df.min().rename('Min')
max_values = numeric_df.max().rename('Max')


# Calculate the median of each numeric column
median_values = numeric_df.median().rename('Median')

# Combine the results into a single DataFrame
analysis_df = pd.concat([mean_values, median_values, min_values, max_values], axis=1)

# Sort the index alphabetically for cleaner presentation
analysis_df = analysis_df.sort_index()

print("Analysis of Mean and Median for each numeric column:")
print(analysis_df)