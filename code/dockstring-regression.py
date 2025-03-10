import numpy as np
import jax.numpy as jnp
import pandas as pd
import sklearn.metrics

from rdkit.Chem import DataStructs

import tanimoto_gp
from utils.misc import smiles_to_fp, optimize_params, init_gp
from utils import GPCheckpoint
from utils.get_data import get_dockstring_dataset

from pathlib import Path
import argparse
from functools import partial
import warnings
warnings.filterwarnings("ignore")


def main(from_checkpoint=False, n_train=10000, target='PARP1', sparse=True, radius=2, count=True):

    # Get Dockstring data
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(n_train=n_train, target=target)

    print(f"Train size: {len(smiles_train)}\nTest size: {len(smiles_test)}")
    print(f"Experiment Params: n_train: {n_train} | target: {target} | sparse: {sparse} | radius: {radius}")

    # Specify model path
    if sparse:
        MODEL_PATH = f"models/gp-regression-{target}-10k-sparse-r{radius}.pkl"
    else:
        MODEL_PATH = f"models/gp-regression-{target}-10k-compressed-r{radius}.pkl"

    if from_checkpoint:
        gp, gp_params = GPCheckpoint.load_gp_checkpoint(MODEL_PATH)
    else:
        gp, gp_params = init_gp(smiles_train, y_train, sparse=sparse, radius=radius, count=count)
        GPCheckpoint.save_gp_checkpoint(gp, gp_params, f'{MODEL_PATH}')
    
    mean, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)

    r2 = sklearn.metrics.r2_score(y_test, mean)

    print(f"R2 Score: {r2}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--radius", type=int, default=2)

    args = parser.parse_args()

    main(from_checkpoint=args.from_checkpoint, n_train=args.n_train, target=args.target, sparse=args.sparse, radius=args.radius)