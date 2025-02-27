import numpy as np

from utils import bo, acq_funcs
from utils.get_data import get_dockstring_dataset
from utils.misc import init_gp

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

    sampled_indices = np.random.choice(np.arange(n), size=n_init)
    complement_indices = np.setdiff1d(np.arange(n), sampled_indices)

    X_init, y_init = smiles_train[sampled_indices], y_train[sampled_indices]
    X, y = np.concatenate([smiles_train[complement_indices], smiles_test]), np.concatenate([y_train[complement_indices], y_test])

    return X, X_init, y, y_init




def main(n_init, budget, target, sparse, radius):

    print(f"Experiment Params: n_init: {n_init} | budget: {budget} | target: {target} | sparse: {sparse} | radius: {radius}")

    data = {}

    for i in range(3):
            
        X, X_init, y, y_init = get_data(target, n_init)
        
        gp, gp_params = init_gp(X_init, y_init, sparse=sparse, radius=radius)
        
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
