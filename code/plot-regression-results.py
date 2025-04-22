import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle
from pathlib import Path

# Set up Seaborn plotting style
sns.set_style("darkgrid", {
    "axes.facecolor": ".95",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
})
sns.set_palette('muted')


def load_results(target, n_train, radius, fp_size=None, sparse=False):
    """Load results for a specific configuration"""
    if sparse:
        path = f'results/dockstring-regression/{target}/{n_train}/sparse-r{radius}.pkl'
    else:
        path = f'results/dockstring-regression/{target}/{n_train}/compressed-r{radius}-{fp_size}.pkl'
    
    if not os.path.exists(path):
        return None
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract metrics from all trials
    metrics = {
        'R2': [],
        'MSE': [],
        'MAE': []
    }
    
    for _, result in data.items():
        metrics['R2'].append(result['R2'])
        metrics['MSE'].append(result['MSE'])
        metrics['MAE'].append(result['MAE'])

    return {k: (np.mean(v), .1*np.std(v)) for k, v in metrics.items()}



def plot_results(target, n_trains, radius, metric='R2', fp_sizes=[2048, 1024, 512, 256]):
    """Create plot comparing sparse vs compressed fingerprints for a specific metric"""
    # plt.figure(figsize=(10, 6))

    fig, axes = plt.subplots(1, len(n_trains), figsize=(6*len(n_trains), 5), squeeze=False)
    fig.suptitle(f'{metric} vs Fingerprint Size\n(Target: {target}, radius={radius})', y=.95)

    for idx, n_train in enumerate(n_trains):
        ax = axes[0][idx]
        
        # Load sparse FP results
        sparse_results = load_results(target, n_train, radius, sparse=True)
        sparse_mean, sparse_std = sparse_results[metric]

        # Plot horizontal line for sparse fingerprint
        ax.axhline(y=sparse_mean, color='green', linestyle='-', label='Sparse FP')
        ax.fill_between(fp_sizes,
                        [sparse_mean - sparse_std] * len(fp_sizes),
                        [sparse_mean + sparse_std] * len(fp_sizes),
                        color='green', alpha=0.2)

        # Load and plot compressed FP results
        means, stds = [], []
        for size in fp_sizes:
            results = load_results(target, n_train, radius, size)
            if results is not None:
                mean, std = results[metric]
                means.append(mean)
                stds.append(std)

        ax.plot(fp_sizes, means, 'o-', color='darkorange', label='Compressed FP')
        ax.fill_between(fp_sizes, 
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), 
                        color='orange', alpha=0.2)
    
        ax.set_xlabel('Fingerprint Size')
        ax.set_ylabel(metric)
        ax.set_title(f'N = {n_train}')
        ax.legend()
    
    plt.tight_layout()

    # Create directory if it doesn't exist
    save_dir = f'../figures/dockstring-regression/{target}/r{radius}/'
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, f'{metric.lower()}.png'), bbox_inches='tight')
    plt.close()



def main():
    TARGETS = ['PARP1']
    RADII = [2, 4]
    N_TRAINS = [100, 1000]
    FP_SIZES = [2048, 1024, 512, 256]
    METRICS = ['R2', 'MSE', 'MAE']

    for target in TARGETS:
        for radius in RADII:
            print(f"Generating plots for {target}, radius={radius}")
            for metric in METRICS:
                plot_results(target, N_TRAINS, radius, metric, FP_SIZES)



if __name__ == "__main__":
    main()