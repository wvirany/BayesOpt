import cProfile
import pstats
import io
import numpy as np
import jax.numpy as jnp

import tanimoto_gp
from utils import bo, acq_funcs
from utils.get_data import get_dockstring_dataset
from utils.misc import config_fp_func, inverse_softplus


rng = np.random.RandomState(42)


def get_data(pool=10000, n_init=1000, target="PARP1", include_test=False):
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(n_train=pool, target=target)
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


def run_bo_experiment():
    """Run a larger BO experiment for profiling"""

    pool=100000

    # Get test data
    X_init, X, y_init, y = get_data(pool=pool, n_init=100)  # Start with 100 molecules
    
    # Initialize GP parameters
    amp = jnp.var(y_init)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_init)
    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    # Initialize GP
    fp_func = config_fp_func()
    gp = tanimoto_gp.FixedTanimotoGP(gp_params, fp_func, X_init, y_init)

    # Run BO procedure
    best, top10, X_observed, y_observed, _ = bo.optimization_loop(
        X, y, X_init, y_init, gp, gp_params,
        acq_funcs.ei, epsilon=.01, num_iters=30
    )

def main():
    # Run profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_bo_experiment()
    
    profiler.disable()
    
    # Print sorted stats
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    
    # Print stats sorted by cumulative time
    s.write("Statistics sorted by cumulative time:\n")
    stats.sort_stats('cumulative').print_stats(30)
    s.write("\n\n" + "="*80 + "\n\n")
    
    # Print stats sorted by total time
    s.write("Statistics sorted by total time:\n")
    stats.sort_stats('time').print_stats(30)
    s.write("\n\n" + "="*80 + "\n\n")
    
    # Print call counts for key functions
    s.write("Key function statistics:\n")
    stats.print_stats('predict_f', 'smiles_to_fp', 'predict_y', 'ei')
    
    # Write the stats to a file
    with open('profiles/profile_results_k_test_train.txt', 'w') as f:
        f.write(s.getvalue())
    
    print("\nDetailed profiling results have been written to profile_results_large.txt")

if __name__ == "__main__":
    main()