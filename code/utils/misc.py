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



def optimize_params(gp, gp_params, tol=1e-3, max_iters=1000):
    """
    Optimize GP parameters until convergence or max steps reached
    
    Args:
        gp: Gaussian Process instance
        gp_params: Initial parameters
        tol: Tolerance for convergence (default 1e-3)
        max_steps: Maximum optimization steps (default 1000)
    """

    print(f"Start MLL: {gp.marginal_log_likelihood(params=gp_params)}")

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(gp_params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(lambda x: -gp.marginal_log_likelihood(x))(params)
        grad_norm = jnp.linalg.norm(jnp.array(grads))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grad_norm, loss

    # Run optimization loop
    for i in range(max_iters):

        gp_params, opt_state, grad_norm, loss = step(gp_params, opt_state)

        if grad_norm < tol:
            print(f"Converged after {i+1} steps, gradient norm = {grad_norm}")
            break

    print(f"End MLL (after optimization): {-loss}")
    print(f"End GP parameters (after optimization): {gp_params}")

    return gp_params



@lru_cache(maxsize=100_000)
def smiles_to_fp(smiles: str, fp_type: str = 'ecfp', sparse=True, fpSize=2048, radius=2):
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
    
    # Returns sparse fingerprint if sparse=True, otherwise returns fingerprint of specified size
    return fpgen.GetSparseCountFingerprint(mol) if sparse else fpgen.GetCountFingerprint(mol)



def test_log_likelihood(smiles_test, mean, covar, y_test=None):
    """
    Evaluates test log-likelihood given target SMILES strings, mean, and full covariance matrix
    """

    if y_test is None:
        y_test = jnp.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles_test])
    
    return multivariate_normal.logpdf(y_test, mean=mean, cov=covar)



def natural_params(gp_params):
    """Returns the natural parameters (after softplus transform, positive values)"""
    return [
        float(tanimoto_gp.TRANSFORM(gp_params.raw_amplitude)),
        float(tanimoto_gp.TRANSFORM(gp_params.raw_noise))
        ]



def evaluate_gp(smiles_train,
                y_train,
                smiles_test,
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

    tll = test_log_likelihood(smiles_test, mean, var)

    params = natural_params(gp_params)

    return mean, var, tll, params