import gurobipy as gp
from gurobipy import GRB
import os
import pandas as pd
import glob

# --- Configuration (Keep these the same) ---
INSTANCES_DIR = os.path.join(os.getcwd(), 'instances')
OUTPUT_FILE = 'milp_analysis_results.csv'

# --- Analysis Function (CORRECTED) ---

def analyze_milp_instances(directory):
    """
    Analyzes all .mps files in the specified directory using Gurobi 
    and returns a pandas DataFrame with key statistics.
    """
    
    # 1. Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Instances directory not found at {directory}")
        return pd.DataFrame() 

    # 2. Find all .mps files using glob
    search_path = os.path.join(directory, '*.mps')
    mps_files = glob.glob(search_path)
    
    if not mps_files:
        print(f"No '.mps' files found in {directory}.")
        return pd.DataFrame()

    print(f"Found {len(mps_files)} .mps files to analyze.")
    print("-" * 60)
    
    results = []

    # 3. Use a Gurobi environment (suppress output)
    try:
        with gp.Env(empty=True) as env:
            # Setting OutputFlag=0 suppresses Gurobi console output
            env.setParam('OutputFlag', 0) 
            env.start()
            
            # 4. Loop through each .mps file
            for mps_path in mps_files:
                file_name = os.path.basename(mps_path)
                print(f"Analyzing: {file_name}")

                try:
                    # Read the MILP instance into a Gurobi model
                    model = gp.read(mps_path, env=env)
                    
                    # Get Variable Counts
                    num_vars = model.NumVars
                    num_int = model.getAttr('NumIntVars')
                    num_bin = model.getAttr('NumBinVars')
                    num_cont = num_vars - num_int
                    
                    # --- CORRECTED CONSTRAINT COUNTING ---
                    num_constrs = model.NumConstrs
                    num_eq = 0
                    num_leq = 0
                    num_geq = 0
                    
                    # Iterate over all linear constraints to check their sense
                    for constr in model.getConstrs():
                        sense = constr.getAttr('Sense')
                        if sense == '=':
                            num_eq += 1
                        elif sense == '<':
                            num_leq += 1
                        elif sense == '>':
                            num_geq += 1
                            
                    # Get Objective Sense
                    obj_sense = 'Minimize' if model.ModelSense == GRB.MINIMIZE else 'Maximize'

                    # Store results for the current file
                    results.append({
                        'File Name': file_name,
                        'Total Variables': num_vars,
                        'Integer Variables': num_int,
                        'Binary Variables': num_bin,
                        'Continuous Variables': num_cont,
                        'Total Constraints': num_constrs,
                        'Equality Constraints': num_eq,
                        'Less-Equal Constraints': num_leq,
                        'Greater-Equal Constraints': num_geq,
                        'Objective Sense': obj_sense
                    })
                    
                except gp.GurobiError as e:
                    print(f"Gurobi Error while reading {file_name}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred with {file_name}: {e}")

    except gp.GurobiError as e:
        print(f"Gurobi Environment Error: {e}")
        return pd.DataFrame()

    # 5. Convert the list of results into a DataFrame
    df = pd.DataFrame(results)
    return df

# --- Execution (Keep this the same) ---
# ...
# ... (the rest of your original script)

# --- Execution ---

if __name__ == '__main__':
    # Step 1: Run the analysis
    analysis_df = analyze_milp_instances(INSTANCES_DIR)
    
    if not analysis_df.empty:
        # Step 2: Save the DataFrame to a CSV file
        output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
        analysis_df.to_csv(output_path, index=False)
        
        print("-" * 50)
        print(f"✅ Analysis complete! Results saved to: {output_path}")
        print("\nHead of the resulting DataFrame:")
        print(analysis_df.head())
    else:
        print("\n❌ Analysis failed or no data was processed.")