import numpy as np
import jax.numpy as jnp

import polaris as po
from polaris.hub.client import PolarisHubClient

import tanimoto_gp
from utils.misc import optimize_params, smiles_to_fp
from utils.get_data import get_data, split_data
from utils import acq_funcs
from utils.bo import optimization_loop

import matplotlib.pyplot as plt
import seaborn as sns

import os
import argparse
import warnings
warnings.filterwarnings("ignore")


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



X, y = get_data()

quantile_90 = np.quantile(y, .9)
quantile_95 = np.quantile(y, .95)
quantile_99 = np.quantile(y, .99)
quantile_999 = np.quantile(y, .999)



def init_gp(X_observed, y_observed, optimize=True, amp=1.0, noise=1e-2):

    gp = tanimoto_gp.TanimotoGP(smiles_to_fp, X_observed, y_observed)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(amp), raw_noise=jnp.asarray(noise))

    if optimize:
        gp_params = optimize_params(gp, gp_params, tol=1e-3, max_iters=10000)

    return gp, gp_params



def make_plots(data, acq, beta, savefig=False):

    xs = np.arange(len(data['means']))

    plt.figure(1, figsize=(10, 6))

    plt.plot(xs, data['means'], color='midnightblue', label=acq.upper())
    plt.scatter(xs, data['means'], color='midnightblue', s=7)
    _, caps, bars = plt.errorbar(xs, data['means'], yerr=data['stds'], lw=.75, capsize=2, color='midnightblue')
    for bar in bars:
        bar.set_linestyle('dotted')

    plt.plot(xs, data['means_uniform'], color='orange', label='Uniform')
    plt.scatter(xs, data['means_uniform'], color='orange', s=7)
    _, caps, bars = plt.errorbar(xs, data['means_uniform'], yerr=data['stds_uniform'], lw=.75, capsize=2, color='orange')
    for bar in bars:
        bar.set_linestyle('dotted')

    plt.axhline(quantile_999, color='red', ls='dashed', alpha=1, lw=.75, label='99.9% quantile')
    plt.axhline(quantile_99, color='red', ls='dashed', alpha=.5, lw=.75, label='99% quantile')
    plt.axhline(quantile_95, color='red', ls='dashed', alpha=.25, lw=.75, label='95% quantile')

    plt.xlabel("Iteration")
    plt.ylabel("Log Solubility")
    plt.title(f"Log Solubility of Best Molecule Acquired ($\\beta = {beta}$)")

    plt.legend()

    if savefig:
        PATH = f'../figures/bayes_opt//{acq}/bo-beta{beta}.png'
        # If directory doesn't exist, make it
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        plt.savefig(PATH, bbox_inches='tight')

    plt.figure(2, figsize=(10, 6))

    plt.plot(xs, data['means_top10'], color='midnightblue', label=acq.upper())
    plt.scatter(xs, data['means_top10'], color='midnightblue', s=7)
    _, caps, bars = plt.errorbar(xs, data['means_top10'], yerr=data['stds_top10'], lw=.75, capsize=2, color='midnightblue')
    for bar in bars:
        bar.set_linestyle('dotted')

    plt.plot(xs, data['means_top10_uniform'], color='orange', label='Uniform')
    plt.scatter(xs, data['means_top10_uniform'], color='orange', s=7)
    _, caps, bars = plt.errorbar(xs, data['means_top10_uniform'], yerr=data['stds_top10_uniform'], lw=.75, capsize=2, color='orange')
    for bar in bars:
        bar.set_linestyle('dotted')

    plt.xlabel("Iteration")
    plt.ylabel("# Molecules")
    plt.title(f"Number of top 10% molecules acquired ($\\beta = {beta}$)")

    plt.legend()

    if savefig:
        PATH = f'../figures/bayes_opt/{acq}/bo-beta{beta}-top10.png'
        # If directory doesn't exist, make it
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        plt.savefig(PATH, bbox_inches='tight')

    plt.tight_layout()
    plt.show()



def run_exp(split_method, split, acq, beta, num_iters):

    if acq == 'ucb':
        acq = acq_funcs.ucb
    elif acq == 'ei':
        acq = acq_funcs.ei

    data = {}

    best_5_runs = np.array([])
    best_5_runs_uniform = np.array([])

    num_top10_5_runs = np.array([])
    num_top10_5_runs_uniform = np.array([])

    # Train initial GP on half the dataset
    X, y = get_data()
    X_train, X_init, y_train, y_init = split_data(X, y, split_method='random', frac=.5)
    _, gp_params = init_gp(X_train, y_train)

    for i in range(5):
        X, X_observed, y, y_observed = split_data(X_init, y_init, split_method=split_method, frac=split, as_list=True, random_seed=i)
        gp, _ = init_gp(X_observed, y_observed, optimize=False)
        best, _, _, _, num_top10_acq = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq, beta=beta, num_iters=num_iters)

        X, X_observed, y, y_observed = split_data(X_init, y_init, split_method=split_method, frac=split, as_list=True, random_seed=i)
        gp, _ = init_gp(X_observed, y_observed, optimize=False)
        best_uniform, _, _, _, num_top10_uniform = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq_funcs.uniform, beta, num_iters=num_iters)

        if i == 0:
            best_5_runs = np.append(best_5_runs, best)
            best_5_runs_uniform = np.append(best_5_runs_uniform, best_uniform)

            num_top10_5_runs = np.append(num_top10_5_runs, num_top10_acq)
            num_top10_5_runs_uniform = np.append(num_top10_5_runs_uniform, num_top10_uniform)
        else:
            best_5_runs = np.vstack((best_5_runs, best))
            best_5_runs_uniform = np.vstack((best_5_runs_uniform, best_uniform))

            num_top10_5_runs = np.vstack((num_top10_5_runs, num_top10_acq))
            num_top10_5_runs_uniform = np.vstack((num_top10_5_runs_uniform, num_top10_uniform))

    # Compute statistics for 5 runs
    data['means'] = best_5_runs.mean(axis=0)
    data['means_uniform'] = best_5_runs_uniform.mean(axis=0)
    data['stds'] = best_5_runs.std(axis=0)
    data['stds_uniform'] = best_5_runs_uniform.std(axis=0)

    data['means_top10'] = num_top10_5_runs.mean(axis=0)
    data['means_top10_uniform'] = num_top10_5_runs_uniform.mean(axis=0)
    data['stds_top10'] = num_top10_5_runs.std(axis=0)
    data['stds_top10_uniform'] = num_top10_5_runs_uniform.std(axis=0)

    return data



def main(split_method, split, acq, beta, num_iters, savefig):

    data = run_exp(split_method, split, acq, beta, num_iters)

    make_plots(data, acq, beta, savefig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_method", type=str, default='random')
    parser.add_argument("--split", type=float, default=0.1)
    parser.add_argument("--acq", type=str, default='ucb')
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--savefig", action='store_true')

    args = parser.parse_args()

    main(split_method=args.split_method, split=args.split, acq=args.acq, beta=args.beta, num_iters=args.num_iters, savefig=args.savefig)