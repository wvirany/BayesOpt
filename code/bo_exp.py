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



def make_plots(data, acq, epsilon, savefig=False):

    xs = np.arange(len(data['means']))

    plt.figure(1, figsize=(10, 6))

    plt.plot(xs, data['medians'], color='midnightblue', label=acq.upper(), zorder=2)
    plt.scatter(xs, data['medians'], color='midnightblue', s=7, zorder=2)
    plt.plot(xs, data['percentile25'], color='midnightblue', alpha=1, lw=.5, zorder=2)
    plt.plot(xs, data['percentile75'], color='midnightblue', alpha=1, lw=.5, zorder=2)
    plt.fill_between(xs, data['percentile25'], data['percentile75'], alpha=.15, color='midnightblue')

    plt.plot(xs, data['medians_uniform'], color='orange', label='Uniform', zorder=1)
    plt.scatter(xs, data['medians_uniform'], color='orange', s=7, zorder=1)
    plt.plot(xs, data['percentile25_uniform'], color='orange', alpha=1, lw=.5, zorder=1)
    plt.plot(xs, data['percentile75_uniform'], color='orange', alpha=1, lw=.5, zorder=1)
    plt.fill_between(xs, data['percentile25_uniform'], data['percentile75_uniform'], alpha=.15, color='orange')

    plt.axhline(quantile_999, color='red', ls='dashed', alpha=1, lw=.75)
    plt.axhline(quantile_99, color='red', ls='dashed', alpha=.5, lw=.75)
    plt.axhline(quantile_95, color='red', ls='dashed', alpha=.25, lw=.75)

    plt.xlabel("Iteration")
    plt.ylabel("Log Solubility")
    plt.title(f"Log Solubility of Best Molecule Acquired ($\\epsilon = {epsilon}$)")

    plt.legend()

    if savefig:
        PATH = f'../figures/bayes_opt//{acq}/bo-epsilon{epsilon}.png'
        # If directory doesn't exist, make it
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        plt.savefig(PATH, bbox_inches='tight')

    plt.figure(2, figsize=(10, 6))

    plt.plot(xs, data['medians_top10'], color='midnightblue', label=acq.upper(), zorder=2)
    plt.scatter(xs, data['medians_top10'], color='midnightblue', s=7, zorder=2)
    plt.plot(xs, data['percentile25_top10'], color='midnightblue', alpha=1, lw=.5, zorder=2)
    plt.plot(xs, data['percentile75_top10'], color='midnightblue', alpha=1, lw=.5, zorder=2)
    plt.fill_between(xs, data['percentile25_top10'], data['percentile75_top10'], alpha=.15, color='midnightblue')

    plt.plot(xs, data['medians_top10_uniform'], color='orange', label='Uniform', zorder=1)
    plt.scatter(xs, data['medians_top10_uniform'], color='orange', s=7, zorder=1)
    plt.plot(xs, data['percentile25_top10_uniform'], color='orange', alpha=1, lw=.5, zorder=1)
    plt.plot(xs, data['percentile75_top10_uniform'], color='orange', alpha=1, lw=.5, zorder=1)
    plt.fill_between(xs, data['percentile25_top10_uniform'], data['percentile75_top10_uniform'], alpha=.15, color='orange')

    plt.xlabel("Iteration")
    plt.ylabel("# Molecules")
    plt.title(f"Number of top 10% molecules acquired ($\\epsilon = {epsilon}$)")

    plt.legend()

    if savefig:
        PATH = f'../figures/bayes_opt/{acq}/bo-epsilon{epsilon}-top10.png'
        # If directory doesn't exist, make it
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        plt.savefig(PATH, bbox_inches='tight')

    plt.tight_layout()
    plt.show()



def run_exp(split_method, split, acq, epsilon, num_iters):

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
        best, _, _, _, num_top10_acq = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq, epsilon=epsilon, num_iters=num_iters)

        X, X_observed, y, y_observed = split_data(X_init, y_init, split_method=split_method, frac=split, as_list=True, random_seed=i)
        gp, _ = init_gp(X_observed, y_observed, optimize=False)
        best_uniform, _, _, _, num_top10_uniform = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq_funcs.uniform, epsilon, num_iters=num_iters)

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
    data['stds'] = best_5_runs.std(axis=0)

    data['means_uniform'] = best_5_runs_uniform.mean(axis=0)
    data['stds_uniform'] = best_5_runs_uniform.std(axis=0)

    data['medians'] = np.median(best_5_runs, axis=0)
    data['percentile25'] = np.quantile(best_5_runs, .25, axis=0)
    data['percentile75'] = np.quantile(best_5_runs, .75, axis=0)
    
    data['medians_uniform'] = np.median(best_5_runs_uniform, axis=0)
    data['percentile25_uniform'] = np.quantile(best_5_runs_uniform, .25, axis=0)
    data['percentile75_uniform'] = np.quantile(best_5_runs_uniform, .75, axis=0)

    data['means_top10'] = num_top10_5_runs.mean(axis=0)
    data['stds_top10'] = num_top10_5_runs.std(axis=0)

    data['means_top10_uniform'] = num_top10_5_runs_uniform.mean(axis=0)
    data['stds_top10_uniform'] = num_top10_5_runs_uniform.std(axis=0)

    data['medians_top10'] = np.median(num_top10_5_runs, axis=0)
    data['percentile25_top10'] = np.quantile(num_top10_5_runs, .25, axis=0)
    data['percentile75_top10'] = np.quantile(num_top10_5_runs, .75, axis=0)

    data['medians_top10_uniform'] = np.median(num_top10_5_runs_uniform, axis=0)
    data['percentile25_top10_uniform'] = np.quantile(num_top10_5_runs_uniform, .25, axis=0)
    data['percentile75_top10_uniform'] = np.quantile(num_top10_5_runs_uniform, .75, axis=0)

    return data



def main(split_method, split, acq, epsilon, num_iters, savefig):

    data = run_exp(split_method, split, acq, epsilon, num_iters)

    make_plots(data, acq, epsilon, savefig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_method", type=str, default='random')
    parser.add_argument("--split", type=float, default=0.1)
    parser.add_argument("--acq", type=str, default='ucb')
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--savefig", action='store_true')

    args = parser.parse_args()

    main(split_method=args.split_method, split=args.split, acq=args.acq, epsilon=args.epsilon, num_iters=args.num_iters, savefig=args.savefig)