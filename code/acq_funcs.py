import numpy as np

def upper_confidence_bound(X, gp, gp_params, beta=.15):
    """
    Computes the upper confidence bound (UCB) at points X
    using a fitted Gaussian process surrogate model

    Args:
        X: Points at which UCB will be computed (m x d)
        gp: A GP model fitted to samples
        beta: Exploration-epxloitation trade-off parameter.
               Default is 2.576 (99% confidence interval)

    Returns:
        UCB scores at points X
    """

    # Get mean and standard deviation predictions
    mean, var = gp.predict_y(gp_params, X, full_covar=False)

    # Calculate upper confidence bound
    ucb = mean + beta * np.sqrt(var)

    return ucb