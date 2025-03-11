import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
import optax

import tanimoto_gp

import datamol as dm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Crippen

from functools import lru_cache, partial


def optimize_params(gp, gp_params, tol=1e-3, max_iters=10000):
    """
    Optimize GP parameters until convergence or max steps reached
    
    Args:
        gp: Gaussian Process instance
        gp_params: Initial parameters
        tol: Tolerance for convergence (default 1e-3)
        max_steps: Maximum optimization steps (default 1000)
    """

    # Compute minimum noise value
    var_y = jnp.var(gp._y_train)
    min_noise = 1e-4 * var_y
    min_raw_noise = jnp.log(jnp.exp(min_noise) - 1)

    print(f"Start MLL: {gp.marginal_log_likelihood(params=gp_params)}")

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(gp_params)

    # Perform one step of gradient descent
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(lambda x: -gp.marginal_log_likelihood(x))(params)
        grad_norm = jnp.linalg.norm(jnp.array(grads))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        min_noise_reached = False

        # Gradient clipping for stability
        noise = tanimoto_gp.TRANSFORM(params.raw_noise)
        if noise < min_noise:
            params = tanimoto_gp.TanimotoGP_Params(
                raw_amplitude=params.raw_amplitude,
                raw_noise=min_raw_noise
            )
            min_noise_reached = True

        return params, opt_state, grad_norm, loss, min_noise_reached

    # Run optimization loop
    for i in range(max_iters):

        gp_params, opt_state, grad_norm, loss, min_noise_reached = step(gp_params, opt_state)

        if min_noise_reached:
            print("Minimum noise value reached, stopping early")
            break

        if grad_norm < tol:
            print(f"Converged after {i+1} steps, gradient norm = {grad_norm}")
            break

        if i % 1000 == 0:
            print(f"Iteration {i}:")
            print(f"  Loss: {loss}")
            print(f"  Gradient norm: {grad_norm}")
            print(f"  Params: {gp_params}")
            print(f"  Natural params: {natural_params(gp_params)}")

    print(f"End MLL (after optimization): {-loss}")
    print(f"End GP parameters (after optimization): {gp_params}")

    return gp_params



def init_gp(smiles_train, y_train, amp=1.0, noise=1e-2, sparse=True, radius=2, count=True, fpSize=1024):

    fp_func = config_fp_func(sparse=sparse, radius=radius, count=count, fpSize=fpSize)

    gp = tanimoto_gp.TanimotoGP(fp_func, smiles_train, y_train)

    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(amp), raw_noise=jnp.asarray(noise))
    gp_params = optimize_params(gp, gp_params, tol=1e-3, max_iters=10000)

    return gp, gp_params



def config_fp_func(fp_type='ecfp', sparse=True, radius=2, count=True, fpSize=1024):

    fp_func = partial(smiles_to_fp, fp_type=fp_type, sparse=sparse, radius=radius, count=count, fpSize=fpSize)

    return fp_func



@lru_cache(maxsize=100_000)
def smiles_to_fp(smiles: str, fp_type: str = 'ecfp', sparse=True, radius=2, count=True, fpSize=2048):
    """
    Convert smiles to sparse count fingerprint of given type

    Arguments:
        - smiles: SMILES string representing molecule
        - fp_type: Type of molecular fingerprint
        - sparse: True for sparse fingerprint, false otherwise
        - fpSize: Size of fingerprint vector, if not using sparse fingerprints
        - radius: Radius of fingerprint generator for ecfp and fcfp fingerprints

    Accepted fingerprint types:
        - 'ecfp': Extended connectivity fingerprint
        - 'fcfp': Functional connectivity fingerprint
        - 'topological': Topological Torsion Fingerprint
        - 'atompair': Atom pair fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)

    if fp_type == 'ecfp':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    elif fp_type == 'fcfp':
        feature_inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            atomInvariantsGenerator=feature_inv_gen,
            fpSize=fpSize
        )
    elif fp_type == 'topological':
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=1,  
            maxPath=5,
            useBondOrder=True,  # this replaces bondBits
            branchedPaths=True,  # this adds branched subgraphs, not just linear paths
            fpSize=fpSize
        )
    elif fp_type == 'atompair':
        fpgen = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=1,  # minimum number of bonds between atoms
            maxDistance=5,  # maximum number of bonds between atoms
            includeChirality=False,  # whether to include chirality in atom invariants
            use2D=True,  # use 2D (topological) distances rather than 3D
            fpSize=fpSize
        )

    if count:
        # Returns sparse fingerprint if sparse=True, otherwise returns fingerprint of specified size
        return fpgen.GetSparseCountFingerprint(mol) if sparse else fpgen.GetCountFingerprint(mol)
    elif ~count:
        # Returns sparse fingerprint if sparse=True, otherwise returns fingerprint of specified size
        return fpgen.GetSparseFingerprint(mol) if sparse else fpgen.GetFingerprint(mol)



def compute_mse(y_true, y_pred):
    """Calculate Mean Squared Error between true and predicted values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)



def compute_pearson(x, y):
    """Calculate Pearson correlation coefficient between two arrays."""
    x, y = np.array(x), np.array(y)
    x_mean, y_mean = np.mean(x), np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator if denominator != 0 else 0



def test_log_likelihood(mean, covar, y_test):
    """
    Evaluates test log-likelihood given target values, mean, and full covariance matrix
    """
    n = len(y_test)
    return multivariate_normal.logpdf(y_test, mean=mean, cov=covar) / n



def natural_params(gp_params):
    """Returns the natural parameters (after softplus transform, positive values)"""
    return [
        float(tanimoto_gp.TRANSFORM(gp_params.raw_amplitude)),
        float(tanimoto_gp.TRANSFORM(gp_params.raw_noise))
        ]



def evaluate_gp(smiles_train,
                smiles_test,
                y_train,
                y_test,
                fp_type='ecfp',
                sparse=True,
                fpSize=2048,
                radius=2,
                tol=1e-3,
                max_iters=1000):
    """
    Evaluate GP performance with given dataset and fingerprint configurations
    """

    fp_func = partial(smiles_to_fp, fp_type=fp_type, sparse=sparse, fpSize=fpSize, radius=radius)

    gp = tanimoto_gp.TanimotoGP(fp_func, smiles_train, y_train)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(-1.0), raw_noise=jnp.asarray(1e-2))
    gp_params = optimize_params(gp, gp_params, tol=tol, max_iters=max_iters)

    mean, var = gp.predict_y(gp_params, smiles_test, full_covar=True)

    mse = compute_mse(y_test, mean)
    pearson = compute_pearson(y_test, mean)
    tll = test_log_likelihood(mean, var, y_test)

    params = natural_params(gp_params)

    return mean, var, mse, pearson, tll, params