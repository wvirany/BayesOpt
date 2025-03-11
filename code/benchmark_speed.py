import time
import numpy as np

from utils import bo, acq_funcs, GPCheckpoint
from utils.get_data import get_dockstring_dataset
from utils.misc import config_fp_func
import tanimoto_gp

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


def run_bo_benchmark(target="PARP1", n_iters=30):
    """Run BO experiment and measure average time per iteration"""

    # Get test data
    X, X_init, y, y_init = get_test_data(n_init=100)  # Start with 100 molecules
    print(f"n_iters: {n_iters}")

    # Initialize GP
    _, gp_params = GPCheckpoint.load_gp_checkpoint(f"models/gp-regression-{target}-10k-sparse-r2.pkl")

    # Initialize GP
    fp_func = config_fp_func(sparse=True, radius=2)
    gp = tanimoto_gp.TanimotoGP(fp_func, X_init, y_init)
    
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