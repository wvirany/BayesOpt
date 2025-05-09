import numpy as np
import jax.numpy as jnp
from scipy.stats import norm


def ei(X, gp, gp_params, epsilon=0.01):
    """
    Computes the expected improvement (EI) at points X
    using a fitted Gaussian process surrogate model

    Args:
        X: Points at which UCB will be computed (m x d)
        gp: A GP model fitted to samples
        gp_params: Hyperparameters of GP
        epsilon: Exploitation-exploration tradeoff parameter

    Returns:
        Index of unobserved data which maximizes EI
    """

    # Get mean and standard deviation predictions
    mean, var = gp.predict_y(gp_params, X, full_covar=False)
    std = jnp.sqrt(var)

    # Find incumbent by taking posterior mean at best observed point
    best_idx = jnp.argmax(gp._y_train)
    best_x = [gp._smiles_train[best_idx]]
    incumbent_mean, _ = gp.predict_y(gp_params, best_x, full_covar=False, from_train=True)
    incumbent = incumbent_mean[0]

    # Compute improvement
    improvement = mean - incumbent - epsilon

    # Compute Z-score
    z = improvement / std

    # Compute EI
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)

    # Handle numerical issues
    ei = ei.at[std < 1e-10].set(0)

    # Maximizer for acquisition function
    return jnp.argmax(ei)




def ucb(X, gp, gp_params, beta=0.1):
    """
    Computes the upper confidence bound (UCB) at points X
    using a fitted Gaussian process surrogate model

    Args:
        X: Points at which UCB will be computed (m x d)
        gp: A GP model fitted to samples
        gp_params: Hyperparameters of GP
        beta: Exploration-epxloitation trade-off parameter.
               Default is 2.576 (99% confidence interval)

    Returns:
        Index of unobserved data which maximizes UCB
    """

    # Get mean and standard deviation predictions
    mean, var = gp.predict_y(gp_params, X, full_covar=False)

    # Calculate upper confidence bound
    ucb = mean + beta * np.sqrt(var)

    return np.argmax(ucb)


def uniform(X, gp, gp_params, epsilon=None):

    # Function arguments (gp, gp_params, epsilon) are placeholders
    return np.random.randint(len(X))