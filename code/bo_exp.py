import numpy as np
import jax.numpy as jnp
import pandas as pd

from sklearn.model_selection import train_test_split

import polaris as po
from polaris.hub.client import PolarisHubClient

import tanimoto_gp
from utils.misc import optimize_params, smiles_to_fp
from utils.get_data import get_data, split_data
from utils.acq_funcs import ucb, uniform
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



def init_gp(X_observed, y_observed, optimize=True, amp=1.0, noise=1e-2):

    gp = tanimoto_gp.TanimotoGP(smiles_to_fp, X_observed, y_observed)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(amp), raw_noise=jnp.asarray(noise))

    if optimize:
        gp_params = optimize_params(gp, gp_params, tol=1e-3, max_iters=10000)

    return gp, gp_params



def make_plot(best, best_uniform, num_top10_ucb, num_top10_uniform, exp_num, split_method, num_iters):

    plt.plot(np.arange(len(best)), best, lw=1, label=f'UCB (Num > 90th percentile: {num_top10_ucb})')
    plt.scatter(np.arange(len(best)), best, s=5)

    plt.plot(np.arange(len(best)), best_uniform, lw=1, label=f'Uniform (Num > 90th percentile: {num_top10_uniform})')
    plt.scatter(np.arange(len(best)), best_uniform, s=5)

    plt.xlabel('Iteration')
    plt.ylabel('LogP')
    plt.ylim(1.6, 2.2)
    plt.title('Max logP value in set at each iteration')
    plt.legend()

    PATH = f"../figures/bayes_opt/exp{exp_num}/{split_method}_{num_iters}iters"

    # If directory doesn't exist, make it
    os.makedirs(os.path.dirname(PATH), exist_ok=True)

    plt.savefig(PATH, bbox_inches='tight')

    plt.show()



def run_exp1(split_method, split, acq, num_iters):

    X, X_observed, y, y_observed = get_data(split=True, split_method=split_method, frac=split)
    gp, gp_params = init_gp(X_observed, y_observed)
    best, _, _, _, num_top10_acq = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq, num_iters=num_iters)

    X, X_observed, y, y_observed = get_data(split=True, split_method=split_method, frac=split)
    gp, gp_params = init_gp(X_observed, y_observed)
    best_uniform, _, _, _, num_top10_uniform = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, uniform, num_iters=num_iters)

    make_plot(best, best_uniform, num_top10_acq, num_top10_uniform, exp_num=1, split_method=split_method, num_iters=num_iters)



def run_exp2(split_method, split, acq, num_iters):
    
    X, y = get_data()
    X_train, X_init, y_train, y_init = split_data(X, y, split_method='random', split=.5)
    _, gp_params = init_gp(X_train, y_train)

    X, X_observed, y, y_observed = split_data(X_init, y_init, split_method=split_method, split=split)

    gp, _ = init_gp(X_observed, y_observed, optimize=False)
    best, _, _, _, num_top10_acq = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq, num_iters=num_iters)
    
    X, X_observed, y, y_observed = split_data(X_init, y_init, split_method=split_method, split=split)
    gp, _ = init_gp(X_observed, y_observed, optimize=False)
    best_uniform, _, _, _, num_top10_uniform = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, uniform, num_iters=num_iters)

    make_plot(best, best_uniform, num_top10_acq, num_top10_uniform, exp_num=2, split_method=split_method, num_iters=num_iters)



def main(exp, split_method, split, acq, num_iters):

    if exp == 1:
        run_exp1(split_method, split, acq, num_iters)
    elif exp == 2:
        run_exp2(split_method, split, acq, num_iters)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=[1, 2], default=1)
    parser.add_argument("--split_method", type=str, default='random')
    parser.add_argument("--split", type=float, default=0.1)
    parser.add_argument("--num_iters", type=int, default=10)

    args = parser.parse_args()

    main(exp=args.experiment, split_method=args.split_method, split=args.split, acq=ucb, num_iters=args.num_iters)