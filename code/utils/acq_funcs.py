import numpy as np
from scipy.stats import norm


def ei(X, gp, gp_params, beta=0.01):
    """
    Computes the expected improvement (EI) at points X
    using a fitted Gaussian process surrogate model

    Args:
        X: Points at which UCB will be computed (m x d)
        gp: A GP model fitted to samples
        gp_params: Hyperparameters of GP
        beta: Exploitation-exploration tradeoff parameter

    Returns:
        Index of unobserved data which maximizes EI
    """

    # Get mean and standard deviation predictions
    mean, var = gp.predict_y(gp_params, X, full_covar=False)
    std = np.sqrt(var)

    # Find incumbent value (current best observation)
    incumbent = np.max(gp._y_train)

    # Compute improvement
    improvement = mean - incumbent - beta

    # Compute Z-score
    z = improvement / std

    # Compute EI
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)

    # Handle numerical issues
    ei = ei.at[std < 1e-10].set(0)

    idx = np.argmax(ei)

    return idx




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

    idx = np.argmax(ucb)

    return idx


def uniform(X, gp, gp_params, beta=None):

    # beta is a placeholder

    idx = np.random.randint(len(X))

    return idx