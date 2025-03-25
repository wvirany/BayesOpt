import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cholesky, solve_triangular

import kern_gp as kgp


def test_update_cholesky():
    """Test that update_cholesky produces same result as recomputing full cholesky factor"""

    # Set up simple PD matrix
    K = jnp.array([
        [1.0, 0.7, 0.5],
        [0.7, 1.0, 0.3],
        [0.5, 0.3, 1.0]
    ])

    # Initial cholesky of 2x2 submatrix
    initial_K = K[:2, :2]
    L_initial = cholesky(initial_K, lower=True)

    # Get the new row/column and diagonal element
    k_new = K[2, :2]
    k_new_new = K[2, 2]

    # Method 1: Update the Cholesky factor using update_cholesky
    L_updated = kgp.update_cholesky(L_initial, k_new, k_new_new)

    # Method 2: Compute full Cholesky directly
    L_direct = cholesky(K, lower=True)

    # Check that they match
    assert(jnp.allclose(L_updated, L_direct, rtol=1e-5))


def test_update_cholesky_with_noise():
    """Test Cholesky update when we have a noise term (K + sI)"""
    
    # Set up simple PD matrix
    K = jnp.array([
        [1.0, 0.7, 0.5],
        [0.7, 1.0, 0.3],
        [0.5, 0.3, 1.0]
    ])
    
    # Add noise term
    s = 0.1
    K_noisy = K + s * jnp.eye(3)
    
    # Initial cholesky of 2x2 submatrix
    initial_K = K_noisy[:2, :2]
    L_initial = cholesky(initial_K, lower=True)
    
    # Get the new row/column and diagonal element
    k_new = K_noisy[2, :2]
    k_new_new = K_noisy[2, 2]  # This includes the noise term
    
    # Method 1: Update the Cholesky factor
    L_updated = kgp.update_cholesky(L_initial, k_new, k_new_new)
    
    # Method 2: Compute full Cholesky directly
    L_direct = cholesky(K_noisy, lower=True)
    
    # Check that they match
    assert(jnp.allclose(L_updated, L_direct, rtol=1e-5))


def main():

    test_update_cholesky()


if __name__ == "__main__":
    main()