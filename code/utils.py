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



def optimize_params(gp, gp_params):
    """
    Optimize GP parameters
    """

    print(f"Start MLL: {gp.marginal_log_likelihood(params=gp_params)}")

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(lambda x: -gp.marginal_log_likelihood(x))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(gp_params)

    # Run optimization loop
    for _ in range(100):
        gp_params, opt_state, loss = step(gp_params, opt_state)

    print(f"End MLL (after optimization): {gp.marginal_log_likelihood(params=gp_params)}")
    print(f"End GP parameters (after optimization): {gp_params}")

    return gp_params



@lru_cache(maxsize=100_000)
def smiles_to_fp(smiles: str, fp_type: str = 'ecfp', sparse=True, fpSize=2048):
    """
    Convert smiles to sparse count fingerprint of given type

    Arguments:
        - smiles: SMILES string representing molecule
        - fp_type: Type of molecular fingerprint
        - sparse: True for sparse fingerprint, false otherwise

    Accepted fingerprint types:
        - 'ecfp': Extended connectivity fingerprint
        - 'fcfp': Functional connectivity fingerprint
        - 'topological': Topological Torsion Fingerprint
        - 'atompair': Atom pair fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)

    if fp_type == 'ecfp':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fpSize)
    elif fp_type == 'fcfp':
        feature_inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
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



def test_log_likelihood(smiles_test, mean, covar):
    """
    Evaluates test log-likelihood given target SMILES strings, mean, and full covariance matrix
    """
    
    y_test_logp = jnp.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles_test])
    
    return multivariate_normal.logpdf(y_test_logp, mean=mean, cov=covar)



def evaluate_gp(smiles_train, y_train, smiles_test, fp_type='ecfp', sparse=True, fpSize=2048):
    """
    Evaluate GP with given dataset and fingerprint of given type
    """

    fp_func = partial(smiles_to_fp, fp_type=fp_type, sparse=sparse, fpSize=fpSize)

    gp = tanimoto_gp.TanimotoGP(fp_func, smiles_train, y_train)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(1.0), raw_noise=jnp.asarray(1e-2))
    gp_params = optimize_params(gp, gp_params)

    mean, var = gp.predict_y(gp_params, smiles_test, full_covar=True)

    tll = test_log_likelihood(smiles_test, mean, var)

    return mean, var, tll