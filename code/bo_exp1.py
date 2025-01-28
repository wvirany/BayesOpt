import numpy as np
import jax.numpy as jnp
import pandas as pd

from sklearn.model_selection import train_test_split

import polaris as po
from polaris.hub.client import PolarisHubClient

import tanimoto_gp
from utils.misc import optimize_params, smiles_to_fp
from utils.acq_funcs import ucb, uniform
from utils.bo import optimization_loop

import matplotlib.pyplot as plt
import seaborn as sns

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


# Login to Polaris
client = PolarisHubClient()
client.login()

dataset = po.load_dataset("biogen/adme-fang-v1")



def init_gp(X_observed, y_observed, optimize=True, amp=1.0, noise=1e-2):

    gp = tanimoto_gp.TanimotoGP(smiles_to_fp, X_observed, y_observed)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(amp), raw_noise=jnp.asarray(noise))

    if optimize:
        gp_params = optimize_params(gp, gp_params)

    return gp, gp_params



def get_data():

    # Get all SMILES strings and logP values from dataset
    X = [dataset.get_data(
        row=dataset.rows[i],
        col='MOL_smiles'
        ) for i in range(dataset.size()[0])]

    y = [dataset.get_data(
        row=dataset.rows[i],
        col='LOG_SOLUBILITY'
        ) for i in range(dataset.size()[0])]
    
    # Filter molecules with NaN logP values
    filter = ~np.isnan(y)

    X = np.array([i for idx, i in enumerate(X) if filter[idx]])
    y = np.array([i for idx, i in enumerate(y) if filter[idx]])

    return X, y



def split_data(X, y, split_method, split):

    if split_method == 'random':
        X, X_observed, y, y_observed = train_test_split(X, y, test_size=split, random_state=42)

    elif split_method == 'n_worst':
        
        n = int(split * len(X))
        sorted_indices = np.argsort(y)

        lowest_indices = sorted_indices[:n] # Lowest n values
        rest_indices = sorted_indices[n:]   # All other indices

        X, X_observed, y, y_observed = list(X[rest_indices]), list(X[lowest_indices]), list(y[rest_indices]), list(y[lowest_indices])

    return X, X_observed, y, y_observed



def make_plot(best, best_uniform):

    plt.plot(np.arange(len(best)), best, lw=1, label='UCB')
    plt.scatter(np.arange(len(best)), best, s=5)

    plt.plot(np.arange(len(best)), best_uniform, lw=1, label='Uniform')
    plt.scatter(np.arange(len(best)), best_uniform, s=5)

    plt.xlabel('Iteration')
    plt.ylabel('LogP')
    plt.ylim(1.6, 2.2)
    plt.title('Max logP value in set at each iteration')
    plt.legend()

    plt.show()



def run_exp1(split_method, split, acq, num_iters):

    X, y = get_data()
    X, X_observed, y, y_observed = split_data(X, y, split_method, split)
    gp, gp_params = init_gp(X_observed, y_observed)
    best, _, _, _ = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq, num_iters=num_iters)

    X, y = get_data()
    X, X_observed, y, y_observed = split_data(X, y, split_method, split)
    gp, gp_params = init_gp(X_observed, y_observed)
    best_uniform, _, _, _ = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, uniform, num_iters=num_iters)

    make_plot(best, best_uniform)



def run_exp2(split_method, split, acq, num_iters):
    
    X, y = get_data()
    X_train, X_init, y_train, y_init = split_data(X, y, split_method='random', split=.5)
    _, gp_params = init_gp(X_train, y_train)

    print(len(X), len(X_observed))
    X, X_observed, y, y_observed = split_data(X_init, y_init, split_method=split_method, split=split)

    gp, _ = init_gp(X_observed, y_observed, optimize=False)
    best, X_observed, y_observed, _ = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq, num_iters=num_iters)

    print(len(X), len(X_observed))
    
    X, X_observed, y, y_observed = split_data(X_init, y_init, split_method=split_method, split=split)
    gp, _ = init_gp(X_observed, y_observed, optimize=False)
    best_uniform, _, _, _ = optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq, num_iters=num_iters)



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