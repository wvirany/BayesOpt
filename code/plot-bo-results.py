import numpy as np

import tanimoto_gp
from utils import bo, acq_funcs, get_data

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import argparse


# Set up Seaborn plotting style
sns.set_style("darkgrid",
              {"axes.facecolor": ".95",
               "axes.edgecolor": "#000000",
               "grid.color": "#EBEBE7",
               "font.family": "serif",
               "axes.labelcolor": "#000000",
               "xtick.color": "#000000",
               "ytick.color": "#000000",
               "grid.alpha": 0.4 })
sns.set_palette('muted')



def plot_hist(y, p99, p999, target):
    """Save histogram of dataset"""
    plt.figure()
    plt.hist(-y, bins=50, alpha=0.5)
    plt.axvline(p99, color='red', ls='dashed', lw=.75, alpha=.5, label="$99^\\text{th}$ percentile")
    plt.axvline(p999, color='red', ls='dashed', lw=.75, label="$99.9^\\text{th}$ percentile")
    plt.xlabel(f"Docking score ({target})")
    plt.legend()
    plt.savefig(f"../figures/dockstring-bo/{target}-dataset")



def main(target, n_init, budget, sparse, radius, make_hist):

    # Load dataset
    smiles_train, smiles_test, y_train, y_test = get_data.get_dockstring_dataset(target=target)
    y_train, y_test = -y_train, -y_test

    sampled_indices = np.random.choice(np.arange(10000), size=n_init)
    complement_indices = np.setdiff1d(np.arange(10000), sampled_indices)

    X_init, y_init = smiles_train[sampled_indices], y_train[sampled_indices]
    X, y = np.concatenate([smiles_train[complement_indices], smiles_test]), np.concatenate([y_train[complement_indices], y_test])

    # Compute percentiles, best scores for top 1 and top 10 molecules
    percentile999 = - np.quantile(np.concatenate([y, y_init]), .999)
    percentile99 = - np.quantile(np.concatenate([y, y_init]), .99)
    best_score = - np.max(np.concatenate([y, y_init]))
    best_top10 = -bo.find_top10_avg(np.concatenate([y, y_init]))


    if make_hist:
        plot_hist(y, percentile99, percentile999, target)


    # Load BO results
    with open(f'results/dockstring-bo/{target}/{n_init}-{budget}/sparse-r{radius}.pkl', 'rb') as f:
        data_sparse = pickle.load(f)

    with open(f'results/dockstring-bo/{target}/{n_init}-{budget}/compressed-r{radius}.pkl', 'rb') as f:
        data_compressed = pickle.load(f)

    best_all_iters_sparse = np.zeros(budget + 1)
    best_all_iters_compressed = np.zeros(budget + 1)

    top10_all_iters_sparse = np.zeros(budget + 1)
    top10_all_iters_compressed = np.zeros(budget + 1)

    n = len(data_sparse)
    for i in range(n):
        best_sparse = data_sparse[i]['best']
        best_compressed = data_compressed[i]['best']

        top10_sparse = data_sparse[i]['top10']
        top10_compressed = data_compressed[i]['top10']

        best_all_iters_sparse = np.vstack([best_all_iters_sparse, best_sparse])
        best_all_iters_compressed = np.vstack([best_all_iters_compressed, best_compressed])
        top10_all_iters_sparse = np.vstack([top10_all_iters_sparse, top10_sparse])
        top10_all_iters_compressed = np.vstack([top10_all_iters_compressed, top10_compressed])

    best_all_iters_sparse = np.delete(best_all_iters_sparse, 0, axis=0)
    best_all_iters_compressed = np.delete(best_all_iters_compressed, 0, axis=0)
    top10_all_iters_sparse = np.delete(top10_all_iters_sparse, 0, axis=0)
    top10_all_iters_compressed = np.delete(top10_all_iters_compressed, 0, axis=0)
    
    best_median_sparse = - np.median(best_all_iters_sparse, axis=0)
    best_75_sparse = - np.quantile(best_all_iters_sparse, .75, axis=0)
    best_25_sparse = - np.quantile(best_all_iters_sparse, .25, axis=0)
    
    best_median_compressed = - np.median(best_all_iters_compressed, axis=0)
    best_75_compressed = - np.quantile(best_all_iters_compressed, .75, axis=0)
    best_25_compressed = - np.quantile(best_all_iters_compressed, .25, axis=0)

    top10_median_sparse = - np.median(top10_all_iters_sparse, axis=0)
    top10_75_sparse = - np.quantile(top10_all_iters_sparse, .75, axis=0)
    top10_25_sparse = - np.quantile(top10_all_iters_sparse, .25, axis=0)

    top10_median_compressed = - np.median(top10_all_iters_compressed, axis=0)
    top10_75_compressed = - np.quantile(top10_all_iters_compressed, .75, axis=0)
    top10_25_compressed = - np.quantile(top10_all_iters_compressed, .25, axis=0)


    FIGPATH = f"../figures/dockstring-bo/{target}/{n_init}-{budget}/"
    os.makedirs(os.path.dirname(FIGPATH), exist_ok=True)

    # FIGURE 1: Best molecule
    plt.figure()

    xs = np.arange(len(best_median_sparse))

    plt.plot(xs, best_median_sparse, color='green', label='Uncompressed FP')
    plt.fill_between(xs, best_25_sparse, best_75_sparse, color='lightgreen', alpha=.5)

    plt.plot(xs, best_median_compressed, color='darkorange', label='Compressed FP')
    plt.fill_between(xs, best_25_compressed, best_75_compressed, color='orange', alpha=.25)

    plt.axhline(percentile999, color='red', ls='dashed', lw=.75, label="$99.9^\\text{th}$ percentile")
    plt.axhline(best_score, color='purple', ls='dashed', lw=.75, label='Best Molecule')

    plt.ylim(bottom=best_score - .1)

    plt.xlabel("Observation")
    plt.ylabel("Objective")
    plt.title(f"Top 1 Molecule (Target: {target}, radius: {radius})")
    plt.legend()

    filename = f"r{radius}-best.png"
    filepath = os.path.join(FIGPATH, filename)
    plt.savefig(filepath)



    # FIGURE 2: Top 10
    plt.figure()

    xs = np.arange(len(top10_median_sparse))

    plt.plot(xs, top10_median_sparse, color='green', label='Uncompressed FP')
    plt.fill_between(xs, top10_25_sparse, top10_75_sparse, color='lightgreen', alpha=.5)

    plt.plot(xs, top10_median_compressed, color='darkorange', label='Compressed FP')
    plt.fill_between(xs, top10_25_compressed, top10_75_compressed, color='orange', alpha=.25)

    plt.axhline(best_top10, color='purple', ls='dashed', lw=.75, label='Best Top 10')

    plt.xlabel("Observation")
    plt.ylabel("Top 10 Observed")
    plt.title(f"Top 10 Average (Target: {target}, radius: {radius})")
    plt.legend()


    filename = f"r{radius}-top10.png"
    filepath = os.path.join(FIGPATH, filename)
    plt.savefig(filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--n_init", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--make_hist", action="store_true")
 
    args = parser.parse_args()

    main(n_init=args.n_init,
         budget=args.budget,
         target=args.target,
         sparse=args.sparse,
         radius=args.radius,
         make_hist=args.make_hist)