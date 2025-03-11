import cProfile
import pstats
import io
import numpy as np
import jax.numpy as jnp

import tanimoto_gp
from utils import bo, acq_funcs, GPCheckpoint
from utils.get_data import get_dockstring_dataset
from utils.misc import config_fp_func

def get_test_data(n_init=100, target="PARP1"):
    """Get a larger subset of data for profiling"""
    smiles_train, smiles_test, y_train, y_test = get_dockstring_dataset(target=target)
    
    # Use full training dataset for more realistic profiling
    y_train, y_test = -y_train, -y_test
    
    # Sample initial points from bottom 80%
    cutoff = np.percentile(y_train, 80)
    bottom_80_indices = np.where(y_train <= cutoff)[0]
    sampled_indices = np.random.choice(bottom_80_indices, size=n_init, replace=False)
    top_20_indices = np.where(y_train > cutoff)[0]
    bottom_80_complement = np.setdiff1d(bottom_80_indices, sampled_indices)
    full_complement = np.concatenate([bottom_80_complement, top_20_indices])
    
    X_init, y_init = smiles_train[sampled_indices], y_train[sampled_indices]
    X, y = smiles_train[full_complement], y_train[full_complement]
    
    print(f"Dataset sizes:")
    print(f"Initial training set: {len(X_init)}")
    print(f"Remaining molecules: {len(X)}")
    
    return X, X_init, y, y_init


def run_bo_experiment():
    """Run a larger BO experiment for profiling"""
    # Get test data
    X, X_init, y, y_init = get_test_data(n_init=100)  # Start with 100 molecules
    
    # Load GP params from saved model
    _, gp_params = GPCheckpoint.load_gp_checkpoint("models/gp-regression-PARP1-10k-sparse-r2.pkl")
    
    # Initialize GP
    fp_func = config_fp_func(sparse=True, radius=2)
    gp = tanimoto_gp.TanimotoGP(fp_func, X_init, y_init)
    
    # Run optimization loop with more iterations
    best, top10, X_observed, y_observed, _ = bo.optimization_loop(
        X, y, X_init, y_init, gp, gp_params,
        acq_funcs.ei, epsilon=.01, num_iters=30  # Increased iterations
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
    with open('profiles/profile_results_after_caching.txt', 'w') as f:
        f.write(s.getvalue())
    
    print("\nDetailed profiling results have been written to profile_results_large.txt")

if __name__ == "__main__":
    main()