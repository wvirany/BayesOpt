import numpy as np

import tanimoto_gp
from utils import bo, acq_funcs, get_data, GPCheckpoint
from utils.misc import init_gp

import matplotlib.pyplot as plt
import seaborn as sns

import pickle


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


n_init=100


for target in ["PARP1", "F2"]:

    smiles_train, smiles_test, y_train, y_test = get_data.get_dockstring_dataset(target=target)

    n = len(smiles_train)

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

    # Save histogram of dataset
    plt.figure()
    plt.hist(-y, bins=50, alpha=0.5)
    plt.axvline(percentile99, color='red', ls='dashed', lw=.75, alpha=.5, label="$99^\\text{th}$ percentile")
    plt.axvline(percentile999, color='red', ls='dashed', lw=.75, label="$99.9^\\text{th}$ percentile")
    plt.xlabel(f"Docking score ({target})")
    plt.legend()
    plt.savefig(f"../figures/dockstring/{target}-dataset")


    for radius in [2, 4]:

        with open(f'results/dockstring-bo/{n_init}/results-{target}-sparse-r{radius}.pkl', 'rb') as f:
            data_sparse = pickle.load(f)

        with open(f'results/dockstring-bo/{n_init}/results-{target}-compressed-r{radius}.pkl', 'rb') as f:
            data_compressed = pickle.load(f)

        best_iter1, best_iter2, best_iter3 = data_sparse[0][0], data_sparse[1][0], data_sparse[2][0]
        best_iter1_compressed, best_iter2_compressed, best_iter3_compressed = data_compressed[0][0], data_compressed[1][0], data_compressed[2][0]

        best_all_iters = np.vstack([best_iter1, best_iter2, best_iter3])
        best_all_iters_compressed = np.vstack([best_iter1_compressed, best_iter2_compressed, best_iter3_compressed])

        best_median = - np.median(best_all_iters, axis=0)
        best_max = - np.min(best_all_iters, axis=0)
        best_min = - np.max(best_all_iters, axis=0)

        best_median_compressed = - np.median(best_all_iters_compressed, axis=0)
        best_max_compressed = - np.min(best_all_iters_compressed, axis=0)
        best_min_compressed = - np.max(best_all_iters_compressed, axis=0)


        # Plotting
        plt.figure()

        xs = np.arange(len(best_median))

        plt.plot(xs, best_median, color='green', label='Uncompressed FP')
        plt.fill_between(xs, best_min, best_max, color='lightgreen', alpha=.5)

        plt.plot(xs, best_median_compressed, color='darkorange', label='Compressed FP')
        plt.fill_between(xs, best_min_compressed, best_max_compressed, color='orange', alpha=.25)

        plt.axhline(percentile999, color='red', ls='dashed', lw=.75, label="$99.9^\\text{th}$ percentile")
        plt.axhline(best_score, color='purple', ls='dashed', lw=.75, label='Best Molecule')

        plt.ylim(bottom=best_score - .1)

        plt.xlabel("Observation")
        plt.ylabel("Objective")
        plt.title(f"Top 1 Molecule (Target: {target}, radius: {radius})")
        plt.legend()

        FIGPATH = f"../figures/dockstring/bo-{n_init}/{target}-r{radius}-top1.png"
        plt.savefig(FIGPATH)


        plt.figure()

        top10_iter1, top10_iter2, top10_iter3 = data_sparse[0][1], data_sparse[1][1], data_sparse[2][1]
        top10_iter1_compressed, top10_iter2_compressed, top10_iter3_compressed = data_compressed[0][1], data_compressed[1][1], data_compressed[2][1]

        top10_all_iters = np.vstack([top10_iter1, top10_iter2, top10_iter3])
        top10_all_iters_compressed = np.vstack([top10_iter1_compressed, top10_iter2_compressed, top10_iter3_compressed])

        top10_median = - np.median(top10_all_iters, axis=0)
        top10_max = - np.min(top10_all_iters, axis=0)
        top10_min = - np.max(top10_all_iters, axis=0)

        top10_median_compressed = - np.median(top10_all_iters_compressed, axis=0)
        top10_max_compressed = - np.min(top10_all_iters_compressed, axis=0)
        top10_min_compressed = - np.max(top10_all_iters_compressed, axis=0)


        # Plotting
        xs = np.arange(len(top10_median))

        plt.plot(xs, top10_median, color='green', label='Uncompressed FP')
        plt.fill_between(xs, top10_min, top10_max, color='lightgreen', alpha=.5)

        plt.plot(xs, top10_median_compressed, color='darkorange', label='Compressed FP')
        plt.fill_between(xs, top10_min_compressed, top10_max_compressed, color='orange', alpha=.25)

        plt.axhline(best_top10, color='purple', ls='dashed', lw=.75, label='Best Top 10')

        plt.xlabel("Observation")
        plt.ylabel("Top 10 Observed")
        plt.title(f"Top 10 Average (Target: {target}, radius: {radius})")
        plt.legend()

        FIGPATH2 = f"../figures/dockstring/bo-{n_init}/{target}-r{radius}-top10.png"
        plt.savefig(FIGPATH2)