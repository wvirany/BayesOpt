import jax
import jax.numpy as jnp
import numpy as np

import tanimoto_gp
from utils import bo, acq_funcs
from utils.get_data import get_dockstring_dataset
from utils.misc import init_gp, config_fp_func, inverse_softplus

import os
import pickle
import argparse
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



def get_data(target="PARP1", n_init=1000):
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(target=target)
    y_train, y_test = -y_train, -y_test

    # Sample n_init molecules from bottom 80% of dataset
    cutoff = np.percentile(y_train, 80)
    bottom_80_indices = np.where(y_train <= cutoff)[0]
    sampled_indices = rng.choice(bottom_80_indices, size=n_init, replace=False)
    top_20_indices = np.where(y_train > cutoff)[0]
    bottom_80_complement = np.setdiff1d(bottom_80_indices, sampled_indices)
    full_complement = np.concatenate([bottom_80_complement, top_20_indices])

    X_init, y_init = smiles_train[sampled_indices], y_train[sampled_indices]
    X, y = np.concatenate([smiles_train[full_complement], smiles_test]), np.concatenate([y_train[full_complement], y_test])


    return X_init.tolist(), X.tolist(), y_init, y



def main(n_init, budget, target, sparse, radius):
    print(f"Running trial with seed {trial_seed}")
    print(f"Experiment Params: n_init: {n_init} | budget: {budget} | target: {target} | sparse: {sparse} | radius: {radius}")
    
    X_init, X, y_init, y = get_data(target, n_init)

    # Initialize GP parameters
    amp = jnp.var(y_init)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_init)
    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    # Initialize GP
    fp_func = config_fp_func(sparse=sparse, radius=radius)
    gp = tanimoto_gp.FixedTanimotoGP(gp_params, fp_func, X_init, y_init)

    # Run BO procedure
    best, top10, X_observed, y_observed, _ = bo.optimization_loop(
        X, y, X_init, y_init, gp, gp_params,
        acq_funcs.ei, epsilon=.01, num_iters=budget
    )

    # Save results
    results = {
        'best': best,
        'top10': top10,
        'X_observed': X_observed,
        'y_observed': y_observed,
        'gp_params': gp_params
    }
    
    # Path to store results
    if sparse:
        results_path = f'results/dockstring-bo/{target}/{n_init}-{budget}/sparse-r{radius}.pkl'
    else:
        results_path = f'results/dockstring-bo/{target}/{n_init}-{budget}/compressed-r{radius}.pkl'

    data = {}
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            data = pickle.load(f)

    data[trial_seed] = results

    # Create directory if needed and save
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved results to {results_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_init", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--target", type=str, default='PARP1')
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--radius", type=int, default=2)
 
    args = parser.parse_args()

    main(n_init=args.n_init, budget=args.budget, target=args.target, sparse=args.sparse, radius=args.radius)