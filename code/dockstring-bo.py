import numpy as np

from utils import bo, acq_funcs, GPCheckpoint
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

    smiles, _, docking_scores, _ = get_dockstring_dataset(target=target)

    neg_docking_scores = -docking_scores

    cutoff = np.percentile(neg_docking_scores, 80)
    bottom_80_indices = np.where(neg_docking_scores <= cutoff)[0]
    sampled_indices = np.random.choice(bottom_80_indices, size=n_init, replace=False)
    bottom_80_complement = np.setdiff1d(bottom_80_indices, sampled_indices)
    top_20_indices = np.where(neg_docking_scores > cutoff)[0]
    complement_indices = np.concatenate([bottom_80_complement, top_20_indices])

    X, X_init, y, y_init = smiles[complement_indices], smiles[sampled_indices], neg_docking_scores[complement_indices], neg_docking_scores[sampled_indices]

    return X, X_init, y, y_init




def main(from_checkpoint, PATH, n_init, target, fp_params, bo_params):

    data = {}
    
    X, X_init, y, y_init = get_data(target, n_init)

    if from_checkpoint:
        gp, gp_params = GPCheckpoint.load_gp_checkpoint(PATH)
    else:
        gp, gp_params = init_gp(X_init, y_init)
        GPCheckpoint.save_gp_checkpoint(gp, gp_params, PATH)

    best, top10, _, _, _ = bo.optimization_loop(X, y, X_init, y_init, gp, gp_params, acq_funcs.ei, epsilon=.01, num_iters=1000)

    data[1] = (best, top10)

    DATAPATH = 'data/dockstring-bo/results.pkl'
    # If directory doesn't exist, make it
    os.makedirs(os.path.dirname(DATAPATH), exist_ok=True)

    with open(DATAPATH, 'wb') as f:
        pickle.dump(data, f)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n_init", type=int, default=1000)
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--fp_params", type=dict, default=None)
    parser.add_argument("--bo_params", type=dict, default=None)

    args = parser.parse_args()

    # If loading checkpoint, model path must be included
    if args.path is None:
        parser.error("--path must be specified")

    main(from_checkpoint=args.from_checkpoint, PATH=args.path, n_init=args.n_init, target=args.target, fp_params=args.fp_params, bo_params=args.bo_params)