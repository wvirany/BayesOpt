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


# Use SLURM array ID for random seed
SLURM_ARRAY_ID = os.getenv("SLURM_ARRAY_TASK_ID")
if SLURM_ARRAY_ID is None:
    print("Warning: No SLURM_ARRAY_TASK_ID found, using random seed")
    trial_seed = np.random.randint(0, 10000)
else:
    trial_seed = int(SLURM_ARRAY_ID)
rng = np.random.RandomState(trial_seed)


def main(target='PARP1', n_train=10000, sparse=True, radius=2, fpSize=2048):

    # Get Dockstring data
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(n_train=n_train, target=target, seed=trial_seed)

    print(f"Running trial {trial_seed} | Train size: {len(smiles_train)} | Test size: {len(smiles_test)}")
    print(f"Experiment Params: n_train: {n_train} | target: {target}\n"
          f"sparse: {sparse} | radius: {radius} | fpSize: {fpSize})")

    # Initialize GP parameters
    amp = jnp.var(y_train)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_train)
    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    print(f"=== GP Params ===\nAmplitude: {amp}, Noise: {noise}")

    fp_func = config_fp_func(sparse=sparse, radius=radius, fpSize=fpSize)
    gp = tanimoto_gp.ConstantMeanTanimotoGP(fp_func, smiles_train, y_train)

    mean, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)

    # Compute metrics
    r2 = sklearn.metrics.r2_score(y_test, mean)
    mse = sklearn.metrics.mean_squared_error(y_test, mean)
    mae = sklearn.metrics.mean_absolute_error(y_test, mean)

    # Save results
    results = {'R2'    : r2,
               'MSE'   : mse,
               'MAE'   : mae}


    # Path to store results
    if sparse:
        results_path = f'results/dockstring-regression/{target}/{n_train}/sparse-r{radius}.pkl'
    else:
        results_path = f'results/dockstring-regression/{target}/{n_train}/compressed-r{radius}-s{fpSize}.pkl'

    data = {}
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
    
    data[trial_seed] = results

    # Create directory if needed and save
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"R2 Score: {r2}\nMSE: {mse}\nMAE: {mae}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fpSize", type=int, default=2048)

    args = parser.parse_args()

    main(n_train=args.n_train, target=args.target, sparse=args.sparse, radius=args.radius, fpSize=args.fpSize)