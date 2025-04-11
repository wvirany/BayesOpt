import numpy as np
import pandas as pd

import os
import pickle
from pathlib import Path
import argparse

def main(save_csv=False):
    # Define parameter ranges to iterate over
    TARGETS = ['PARP1', 'F2', 'ESR2', 'PGR']
    RADII = [2, 4]
    N_TRAINS = [100, 1000, 10000]

    # Initialize dictionaries to store results
    r2_results = {
        'Sparse': {},
        'Compressed': {}
    }

    mse_results = {
        'Sparse': {},
        'Compressed': {}
    }

    mae_results = {
        'Sparse': {},
        'Compressed': {}
    }
    
    # Set up results table
    results = []
    
    for target in TARGETS:
        for radius in RADII:
            for n_train in N_TRAINS:
                
                config = (target, radius, n_train)
                
                # Define paths for sparse and compressed results
                sparse_results_path = f'results/dockstring-regression/{target}/{n_train}/sparse-r{radius}-count.pkl'
                compressed_results_path = f'results/dockstring-regression/{target}/{n_train}/compressed-r{radius}-count.pkl'
                
                # Initialize arrays to store metrics
                sparse_r2, sparse_mse, sparse_mae = [], [], []
                compressed_r2, compressed_mse, compressed_mae = [], [], []
                
                # Load sparse results if they exist
                if os.path.exists(sparse_results_path):
                    with open(sparse_results_path, 'rb') as f:
                        sparse_data = pickle.load(f)
                        
                    # Extract metrics from each trial
                    for _, result in sparse_data.items():
                        sparse_r2.append(result['R2'])
                        sparse_mse.append(result['MSE'])
                        sparse_mae.append(result['MAE'])
                else:
                    print(f"Warning: {sparse_results_path} not found")
                    continue
                
                # Load compressed results if they exist
                if os.path.exists(compressed_results_path):
                    with open(compressed_results_path, 'rb') as f:
                        compressed_data = pickle.load(f)
                        
                    # Extract metrics from each trial
                    for _, result in compressed_data.items():
                        compressed_r2.append(result['R2'])
                        compressed_mse.append(result['MSE'])
                        compressed_mae.append(result['MAE'])
                else:
                    print(f"Warning: {compressed_results_path} not found")
                    continue
                
                # Store mean and std in the dictionaries
                r2_results['Sparse'][config] = (np.mean(sparse_r2), np.std(sparse_r2))
                r2_results['Compressed'][config] = (np.mean(compressed_r2), np.std(compressed_r2))
                
                mse_results['Sparse'][config] = (np.mean(sparse_mse), np.std(sparse_mse))
                mse_results['Compressed'][config] = (np.mean(compressed_mse), np.std(compressed_mse))
                
                mae_results['Sparse'][config] = (np.mean(sparse_mae), np.std(sparse_mae))
                mae_results['Compressed'][config] = (np.mean(compressed_mae), np.std(compressed_mae))
    
    # Create MultiIndex columns
    configs = sorted(list(r2_results['Sparse'].keys()))
    columns = pd.MultiIndex.from_tuples(configs, names=['Target', 'Radius', 'N_train'])

    # Create DataFrames for each metric
    r2_df = pd.DataFrame(index=['Sparse', 'Compressed'], columns=columns)
    mse_df = pd.DataFrame(index=['Sparse', 'Compressed'], columns=columns)
    mae_df = pd.DataFrame(index=['Sparse', 'Compressed'], columns=columns)

    # Fill DataFrames with formatted results
    for config in configs:
        r2_df.loc['Sparse', config] = f"{r2_results['Sparse'][config][0]:.3f} ± {r2_results['Sparse'][config][1]:.2f}"
        r2_df.loc['Compressed', config] = f"{r2_results['Compressed'][config][0]:.3f} ± {r2_results['Compressed'][config][1]:.2f}"
        
        mse_df.loc['Sparse', config] = f"{mse_results['Sparse'][config][0]:.3f} ± {mse_results['Sparse'][config][1]:.2f}"
        mse_df.loc['Compressed', config] = f"{mse_results['Compressed'][config][0]:.3f} ± {mse_results['Compressed'][config][1]:.2f}"
        
        mae_df.loc['Sparse', config] = f"{mae_results['Sparse'][config][0]:.3f} ± {mae_results['Sparse'][config][1]:.2f}"
        mae_df.loc['Compressed', config] = f"{mae_results['Compressed'][config][0]:.3f} ± {mae_results['Compressed'][config][1]:.2f}"
    
    # Print tables
    print("\nR2 Results:")
    print(r2_df.to_string())
    
    print("\nMSE Results:")
    print(mse_df.to_string())
    
    print("\nMAE Results:")
    print(mae_df.to_string())
    
    # Save results to CSVs
    if save_csv:
        os.makedirs("results/dockstring-regression/summary", exist_ok=True)
        r2_df.to_csv("results/dockstring-regression/summary/r2_summary.csv")
        mse_df.to_csv("results/dockstring-regression/summary/mse_summary.csv")
        mae_df.to_csv("results/dockstring-regression/summary/mae_summary.csv")

    # Also save a nicely formatted Excel file with all metrics
    with pd.ExcelWriter("results/dockstring-regression/summary/all_metrics.xlsx") as writer:
        r2_df.to_excel(writer, sheet_name='R²')
        mse_df.to_excel(writer, sheet_name='MSE')
        mae_df.to_excel(writer, sheet_name='MAE')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_csv", action="store_true")

    args = parser.parse_args()

    main(save_csv=args.save_csv)