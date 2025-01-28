import numpy as np
from scipy import stats

def optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq_func, num_iters=30):
    
    best = []
    
    for i in range(num_iters):

        best.append(np.max(y_observed))

        print(f"Iter: {i} | Current best: {np.max(best)} | Percentile: {stats.percentileofscore(y, np.max(best))}")

        idx = acq_func(X, gp, gp_params)

        X_new = X.pop(idx)
        y_new = y.pop(idx)

        percentile = stats.percentileofscore(y, y_new)

        print(f"Observed function value: {y_new} | Percentile: {percentile}")

        X_observed.append(X_new)
        y_observed.append(y_new)

        gp.set_training_data(X_observed, y_observed)

    return best, X_observed, y_observed, gp