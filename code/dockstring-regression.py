import numpy as np
import jax.numpy as jnp
import pandas as pd
import sklearn.metrics

from rdkit.Chem import DataStructs

import tanimoto_gp
from tanimoto_gp import TRANSFORM

from utils.misc import config_fp_func, inverse_softplus
from utils.get_data import get_dockstring_dataset

import os
import pickle
from pathlib import Path
import argparse
from functools import partial
import warnings
warnings.filterwarnings("ignore")


def main(from_checkpoint=False, n_train=10000, target='PARP1', sparse=True, radius=2, count=True):

    # Get Dockstring data
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(n_train=n_train, target=target)

    print(f"Train size: {len(smiles_train)}\nTest size: {len(smiles_test)}")
    print(f"Experiment Params: n_train: {n_train} | target: {target} | sparse: {sparse} | radius: {radius} | count: {count}")

    # Initialize GP parameters
    amp = jnp.var(y_train)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_train)
    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    print(f"=== GP Params ===\nAmplitude: {amp}, Noise: {noise}")

    fp_func = config_fp_func(sparse=sparse, radius=radius, count=count)
    gp = tanimoto_gp.ConstantMeanTanimotoGP(fp_func, smiles_train, y_train)

    mean, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)

    r2 = sklearn.metrics.r2_score(y_test, mean)
    mse = sklearn.metrics.mean_squared_error(y_test, mean)
    mae = sklearn.metrics.mean_absolute_error(y_test, mean)


    # Path to store results
    if sparse:
        if count:
            results_path = f'results/dockstring-regression/{target}/sparse-r{radius}-count.pkl'
        else:
            results_path = f'results/dockstring-regression/{target}/sparse-r{radius}-binary.pkl'
    else:
        if count:
            results_path = f'results/dockstring-regression/{target}/compressed-r{radius}-count.pkl'
        else:
            results_path = f'results/dockstring-regression/{target}/compressed-r{radius}-binary.pkl'
    
    data = {'R2'    : r2,
            'MSE'   : mse,
            'MAE'   : mae}

    # Create directory if needed and save
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"R2 Score: {r2}\nMSE: {mse}\nMAE: {mae}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--count", action="store_true")

    args = parser.parse_args()

    main(from_checkpoint=args.from_checkpoint, n_train=args.n_train, target=args.target, sparse=args.sparse, radius=args.radius, count=args.count)