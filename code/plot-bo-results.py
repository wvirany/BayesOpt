import numpy as np

import tanimoto_gp
from utils import bo, acq_funcs
from utils.get_data import get_dockstring_dataset

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



def get_data(pool=10000, target="PARP1", include_test=True):
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(n_train=pool, target=target)
    X, y = np.concatenate([smiles_train, smiles_test]), np.concatenate([y_train, y_test])
    return X, y



def parse_results(data, budget):

    # Initialize results lists
    best_all_iters = np.zeros(budget + 1)
    top10_all_iters = np.zeros(budget + 1)

    keys = data.keys()
    for i in keys:
        # Best and top10 scores for each trial
        best = data[i]['best']
        top10 = data[i]['top10']

        best_all_iters = np.vstack([best_all_iters, best])
        top10_all_iters = np.vstack([top10_all_iters, top10])

    # Get rid of 0 in first row
    best_all_iters = np.delete(best_all_iters, 0, axis=0)
    top10_all_iters = np.delete(top10_all_iters, 0, axis=0)

    # Store statistics in results dict
    results = {}

    results['best_median'] = - np.median(best_all_iters, axis=0)
    results['best_75'] = - np.quantile(best_all_iters, .75, axis=0)
    results['best_25'] = - np.quantile(best_all_iters, .25, axis=0)

    results['top10_median'] = - np.median(top10_all_iters, axis=0)
    results['top10_75'] = - np.quantile(top10_all_iters, .75, axis=0)
    results['top10_25'] = - np.quantile(top10_all_iters, .25, axis=0)

    return results

def main(pool, target, n_init, budget, radius, make_hist):

    # Load dataset
    X, y = get_data(pool=pool, target=target)
    y = -y

    # Compute percentiles, best scores for top 1 and top 10 molecules
    percentile999 = - np.quantile(y, .999)
    percentile99 = - np.quantile(y, .99)
    best_score = - np.max(y)
    best_top10 = -bo.find_top10_avg(y)


    if make_hist:
        plot_hist(y, percentile99, percentile999, target)


    # Load BO results
    with open(f'results/dockstring-bo/{target}/{pool}-{n_init}-{budget}/sparse-r{radius}.pkl', 'rb') as f:
        data_sparse = pickle.load(f)
    with open(f'results/dockstring-bo/{target}/{pool}-{n_init}-{budget}/compressed-r{radius}-2048.pkl', 'rb') as f:
        data_compressed_2048 = pickle.load(f)
    with open(f'results/dockstring-bo/{target}/{pool}-{n_init}-{budget}/compressed-r{radius}-1024.pkl', 'rb') as f:
        data_compressed_1024 = pickle.load(f)
    with open(f'results/dockstring-bo/{target}/{pool}-{n_init}-{budget}/compressed-r{radius}-512.pkl', 'rb') as f:
        data_compressed_512 = pickle.load(f)
    with open(f'results/dockstring-bo/{target}/{pool}-{n_init}-{budget}/compressed-r{radius}-256.pkl', 'rb') as f:
        data_compressed_256 = pickle.load(f)
    # with open(f'results/dockstring-bo/{target}/{pool}-{n_init}-{budget}/random.pkl', 'rb') as f:
    #     data_random = pickle.load(f)

    results_sparse = parse_results(data=data_sparse, budget=budget)
    results_compressed_2048 = parse_results(data=data_compressed_2048, budget=budget)
    results_compressed_1024 = parse_results(data=data_compressed_1024, budget=budget)
    results_compressed_512 = parse_results(data=data_compressed_512, budget=budget)
    results_compressed_256 = parse_results(data=data_compressed_256, budget=budget)
    # results_random = parse_results(data=data_random, budget=budget)

    results = [results_sparse,
               results_compressed_2048,
               results_compressed_1024,
               results_compressed_512,
               results_compressed_256,]
            #    results_random]


    # === TEMP ===
    # results_compressed = parse_results(data=data_compressed, budget=budget)
    # results = [results_sparse, results_compressed, results_random]
    # ============

    FIGPATH = f"../figures/dockstring-bo/{target}/{pool}-{n_init}-{budget}/"
    os.makedirs(os.path.dirname(FIGPATH), exist_ok=True)

    # FIGURE 1: Best molecule
    plt.figure()
    xs = np.arange(budget + 1)

    plt.plot()

    # Sparse
    plt.plot(xs, results_sparse['best_median'], color='green', label='Uncompressed FP')
    plt.fill_between(xs, results_sparse['best_25'], results_sparse['best_75'], color='lightgreen', alpha=.25)
    # Compressed
    plt.plot(xs, results_compressed_2048['best_median'], color='darkorange', label='Compressed (2048)')
    plt.fill_between(xs, results_compressed_2048['best_25'], results_compressed_2048['best_75'], color='orange', alpha=.25)
    plt.plot(xs, results_compressed_1024['best_median'], color='darkorange', alpha=.5, label='Compressed (1024)')
    plt.fill_between(xs, results_compressed_1024['best_25'], results_compressed_1024['best_75'], color='orange', alpha=.2)
    plt.plot(xs, results_compressed_512['best_median'], color='darkorange', alpha=.3, label='Compressed (512)')
    plt.fill_between(xs, results_compressed_512['best_25'], results_compressed_512['best_75'], color='orange', alpha=.15)
    plt.plot(xs, results_compressed_256['best_median'], color='darkorange', alpha=.15, label='Compressed (256)')
    plt.fill_between(xs, results_compressed_256['best_25'], results_compressed_256['best_75'], color='orange', alpha=.1)
    # Random
    # plt.plot(xs, results_random['best_median'], color='gray', label='Random')
    # plt.fill_between(xs, results_random['best_25'], results_random['best_75'], color='lightgray', alpha=.5)

    plt.axhline(percentile999, color='red', ls='dashed', lw=.75, label="$99.9^\\text{th}$ percentile")
    plt.axhline(best_score, color='purple', ls='dashed', lw=.75, label='Best Molecule')

    plt.ylim(bottom=best_score - .1)

    plt.xlabel("Observation")
    plt.ylabel("Objective")
    plt.title(f"Top 1 Molecule (Target: {target}, radius: {radius}, pool: {len(X)-n_init},  n_init: {n_init})")
    plt.legend()

    filename = f"r{radius}-best.png"
    filepath = os.path.join(FIGPATH, filename)
    plt.savefig(filepath)



    # FIGURE 2: Top 10
    plt.figure()

    # Sparse
    plt.plot(xs, results_sparse['top10_median'], color='green', label='Uncompressed FP')
    plt.fill_between(xs, results_sparse['top10_25'], results_sparse['top10_75'], color='lightgreen', alpha=.25)
    # Compressed
    plt.plot(xs, results_compressed_2048['top10_median'], color='darkorange', label='Compressed (2048)')
    plt.fill_between(xs, results_compressed_2048['top10_25'], results_compressed_2048['top10_75'], color='orange', alpha=.25)
    plt.plot(xs, results_compressed_1024['top10_median'], color='darkorange', alpha=.5, label='Compressed (1024)')
    plt.fill_between(xs, results_compressed_1024['top10_25'], results_compressed_1024['top10_75'], color='orange', alpha=.2)
    plt.plot(xs, results_compressed_512['top10_median'], color='darkorange', alpha=.3, label='Compressed (512)')
    plt.fill_between(xs, results_compressed_512['top10_25'], results_compressed_512['top10_75'], color='orange', alpha=.15)
    plt.plot(xs, results_compressed_256['top10_median'], color='darkorange', alpha=.15, label='Compressed (256)')
    plt.fill_between(xs, results_compressed_256['top10_25'], results_compressed_256['top10_75'], color='orange', alpha=.1)
    # Random
    # plt.plot(xs, results_random['top10_median'], color='gray', label='Random')
    # plt.fill_between(xs, results_random['top10_25'], results_random['top10_75'], color='lightgray', alpha=.5)

    plt.axhline(best_top10, color='purple', ls='dashed', lw=.75, label='Best Top 10')
    plt.ylim(bottom=best_top10 - .1)

    plt.xlabel("Observation")
    plt.ylabel("Top 10 Observed")
    plt.title(f"Top 10 Average (Target: {target}, radius: {radius}, pool: {len(X) - n_init}, n_init: {n_init})")
    plt.legend()

    filename = f"r{radius}-top10.png"
    filepath = os.path.join(FIGPATH, filename)
    plt.savefig(filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=1000000)
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--n_init", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--make_hist", action="store_true")
 
    args = parser.parse_args()

    main(pool=args.pool,
         n_init=args.n_init,
         budget=args.budget,
         target=args.target,
         radius=args.radius,
         make_hist=args.make_hist)