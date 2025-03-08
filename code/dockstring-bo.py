import numpy as np
import jax
import jax.numpy as jnp

import tanimoto_gp
from utils import bo, acq_funcs
from utils.get_data import get_dockstring_dataset
from utils.misc import init_gp, config_fp_func
from utils import GPCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
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



def get_data(target="PARP1", n_init=1000):

    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(target=target)

    n = len(smiles_train)

    y_train, y_test = -y_train, -y_test

    # Sample n_init molecules from bottom 80% of dataset
    cutoff = np.percentile(y_train, 80)
    bottom_80_indices = np.where(y_train <= cutoff)[0]
    sampled_indices = np.random.choice(bottom_80_indices, size=1000, replace=False)
    top_20_indices = np.where(y_train > cutoff)[0]
    bottom_80_complement = np.setdiff1d(bottom_80_indices, sampled_indices)
    full_complement = np.concatenate([bottom_80_complement, top_20_indices])

    X_init, y_init = smiles_train[sampled_indices], y_train[sampled_indices]
    X, y = np.concatenate([smiles_train[full_complement], smiles_test]), np.concatenate([y_train[full_complement], y_test])


    return X, X_init, y, y_init




def main(n_init, budget, target, sparse, radius):

    print(f"Experiment Params: n_init: {n_init} | budget: {budget} | target: {target} | sparse: {sparse} | radius: {radius}")

    data = {}

    # Load GP params from regression experiment
    if sparse:
        MODEL_PATH = f"models/gp-regression-{target}-10k-sparse-r{radius}.pkl"
    else:
        MODEL_PATH = f"models/gp-regression-{target}-10k-compressed-r{radius}.pkl"
    _, gp_params = GPCheckpoint.load_gp_checkpoint(MODEL_PATH)

    for i in range(3):
            
        # Get dataset
        X, X_init, y, y_init = get_data(target, n_init)

        # Initialize GP - currently using same GP params as regression model
        fp_func = config_fp_func(sparse=sparse, radius=radius)
        gp = tanimoto_gp.TanimotoGP(fp_func, X_init, y_init)

        # Optimize parameters instead
        # gp, gp_params = init_gp(X_init, y_init, sparse=sparse, radius=radius)
        
        best, top10, _, _, _ = bo.optimization_loop(X, y, X_init, y_init, gp, gp_params, acq_funcs.ei, epsilon=.01, num_iters=budget)

        data[i] = (best, top10)

    if sparse:
        DATAPATH = f'results/dockstring-bo/{n_init}/results-{target}-sparse-r{radius}.pkl'
    else:
        DATAPATH = f'results/dockstring-bo/{n_init}/results-{target}-compressed-r{radius}.pkl'

    # If directory doesn't exist, make it
    os.makedirs(os.path.dirname(DATAPATH), exist_ok=True)

    with open(DATAPATH, 'wb') as f:
        pickle.dump(data, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_init", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--radius", type=int, default=2)
 
    args = parser.parse_args()

    main(n_init=args.n_init, budget=args.budget, target=args.target, sparse=args.sparse, radius=args.radius)
