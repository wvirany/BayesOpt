import time
import jax.numpy as jnp
import numpy as np

from utils import bo, acq_funcs
from utils.get_data import get_dockstring_dataset
from utils.misc import config_fp_func, inverse_softplus
import tanimoto_gp

pool = 250000
n_init = 100
target = 'F2'
seed = 42
rng = np.random.RandomState(seed)

def get_data(pool=10000, n_init=1000, target="PARP1", include_test=True):
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(n_train=pool, target=target, seed=seed)
    y_train, y_test = -y_train, -y_test

    # Sample n_init molecules from bottom 80% of dataset
    cutoff = np.percentile(y_train, 80)
    bottom_80_indices = np.where(y_train <= cutoff)[0]
    sampled_indices = rng.choice(bottom_80_indices, size=n_init, replace=False)
    top_20_indices = np.where(y_train > cutoff)[0]
    bottom_80_complement = np.setdiff1d(bottom_80_indices, sampled_indices)
    full_complement = np.concatenate([bottom_80_complement, top_20_indices])

    X_init, y_init = smiles_train[sampled_indices], y_train[sampled_indices]

    if include_test:
        # Use returned training set and test set for candidate pool:
        X, y = np.concatenate([smiles_train[full_complement], smiles_test]), np.concatenate([y_train[full_complement], y_test])
    else:
        # Only use training set for candidate pool
        X, y = smiles_train[full_complement], y_train[full_complement]

    return X_init.tolist(), X.tolist(), y_init, y


def run_bo_benchmark(target="PARP1", n_iters=30):
    """Run BO experiment and measure average time per iteration"""

    # Get test data
    X_init, X, y_init, y = get_data(pool=pool, n_init=n_init, target=target)  # Start with 100 molecules
    print(f"n_iters: {n_iters}")
    print(f"Pool size: {len(X)}")

    # Initialize GP parameters
    amp = jnp.var(y_init)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_init)
    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    # Initialize GP
    fp_func = config_fp_func(sparse=True, radius=2)
    gp = tanimoto_gp.FixedTanimotoGP(gp_params, fp_func, X_init, y_init)
    
    # Run and time BO loop
    start = time.time()
    bo.optimization_loop(X, y, X_init, y_init, gp, gp_params, acq_funcs.ei, 
                        epsilon=0.01, num_iters=n_iters)
    total_time = time.time() - start
    
    print(f"\nResults:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per iteration: {total_time/n_iters:.2f}s")

if __name__ == "__main__":
    run_bo_benchmark()