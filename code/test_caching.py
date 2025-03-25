import numpy as np
import jax.numpy as jnp

import tanimoto_gp
import kern_gp as kgp
from utils.misc import config_fp_func, inverse_softplus
from utils.get_data import get_dockstring_dataset
from utils import bo, acq_funcs

from rdkit import DataStructs


def get_data(target="PARP1", n_init=1000, rng=None):

    if rng is None:
        rng = np.random.RandomState(42)

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


def run_bo_with_config(cached: bool, seed: int = 42) -> tuple[list, list, jnp.ndarray, jnp.ndarray]:
    """Run BO with or without caching, return key matrices / results for comparison"""

    # Get initial data
    X_observed, X, y_observed, y = get_data(n_init=100)

    # Initialize GP with same params
    amp = jnp.var(y_observed)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_observed)
    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    # Initialize GP w/ appropriate class
    fp_func = config_fp_func(sparse=True, radius=2)
    if cached:
        gp = tanimoto_gp.FixedTanimotoGP(gp_params, fp_func, X_observed, y_observed)
    else:
        gp = tanimoto_gp.ConstantMeanTanimotoGP(fp_func, X_observed, y_observed)
    
    # Store matrices at each iteration
    K_test_trains = []
    L_matrices = []

    # Run BO loop w/ caching
    if cached:
        for i in range(5):
            idx = acq_funcs.ei(X, gp, gp_params, epsilon=0.01)
            X_new, y_new = X[idx], y[idx]
            X, y = np.delete(X, idx, axis=0), np.delete(y, idx, axis=0)
            X_observed = np.append(X_observed, X_new)
            y_observed = np.append(y_observed, y_new)
            gp.add_observation(gp_params, idx, y_new)

            K_test_trains.append(gp._K_test_train)
            L_matrices.append(gp._cached_L)
    # Run BO loop w/o caching
    else:
        for i in range(5):
            idx = acq_funcs.ei(X, gp, gp_params, epsilon=0.01)
            X_new, y_new = X[idx], y[idx]
            X, y = np.delete(X, idx, axis=0), np.delete(y, idx, axis=0)
            X_observed = np.append(X_observed, X_new)
            y_observed = np.append(y_observed, y_new)
            gp.set_training_data(X_observed, y_observed)

            K_test_train = compute_K_test_train(gp, X)
            L = compute_cholesky(gp, gp_params)
            K_test_trains.append(K_test_train)
            L_matrices.append(L)
    
    return K_test_trains, L_matrices


def compute_K_test_train(gp, smiles_test: list[str]) -> jnp.ndarray:
    """
    Compute K_test_train
    
    Args:
        gp: The GP instance (either ConstantMeanTanimotoGP or FixedTanimotoGP)
        smiles_test: List of SMILES strings for test molecules
        
    Returns:
        K_test_train matrix as jnp.ndarray
    """
    # Convert test SMILES to fingerprints using GP's fingerprint function
    fp_test = [gp._fp_func(smiles) for smiles in smiles_test]
    
    # Compute similarities between test and training fingerprints
    K_test_train = jnp.asarray([
        DataStructs.BulkTanimotoSimilarity(fp, gp._fp_train) 
        for fp in fp_test
    ])
    
    return K_test_train


def compute_cholesky(gp, gp_params):
    """
    Computes Cholesky factor using kern_gp library function
    
    Args:
        gp: The GP instance
        gp_params: The GP parameters
    """
    s = tanimoto_gp.TRANSFORM(gp_params.raw_noise)
    a = tanimoto_gp.TRANSFORM(gp_params.raw_amplitude)
    return kgp._k_cholesky(gp._K_train_train, s/a)


def main():
    """Run BO with and without caching and compare results"""
    cached_matrices, cached_L = run_bo_with_config(cached=True)
    uncached_matrices, uncached_L = run_bo_with_config(cached=False)

    # Compare matrices at each iteration
    for i in range(len(cached_matrices)):
        k_test_train_match = np.allclose(cached_matrices[i], uncached_matrices[i], rtol=1e-5)
        L_match = np.allclose(cached_L[i], uncached_L[i], rtol=1e-5)

        print(f"Iteration {i}:")
        print(f"K_test_train matrices match: {k_test_train_match}")
        print(f"Cholesky factors match: {L_match}")

        if not k_test_train_match or not L_match:
            print("Detailed differences:")
            print("K_test_train max diff:", np.max(np.abs(cached_matrices[i] - uncached_matrices[i])))
            print("L max diff:", np.max(np.abs(cached_L[i] - uncached_L[i])))


if __name__ == "__main__":
    main()