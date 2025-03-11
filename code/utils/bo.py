import numpy as np
from scipy import stats

def optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq_func, epsilon, num_iters=30):
    
    best = []
    top10 = []
    y_init = y.copy()

    best.append(np.max(y_observed))
    top10.append(find_top10_avg(y_observed))
    
    for i in range(1, num_iters+1):

        print(f"Iter: {i} | Current best: {np.max(best):0.3f} | Top 10: {top10[-1]:0.3f}")

        idx = acq_func(X, gp, gp_params, epsilon)

        X_new = X[idx]
        y_new = y[idx]
        X, y = np.delete(X, idx, axis=0), np.delete(y, idx, axis=0)

        percentile = stats.percentileofscore(y_init, y_new)

        print(f"Observed function value: {y_new:0.3f} | Percentile: {percentile:0.3f}")

        X_observed = np.append(X_observed, X_new)
        y_observed = np.append(y_observed, y_new)

        best.append(np.max(y_observed))
        top10.append(find_top10_avg(y_observed))

        gp.add_observation(gp_params, X_new, y_new)

    print(f"Best observed molecule: {np.max(best):0.3f} | Top 10: {top10[-1]}")

    return best, top10,  X_observed, y_observed, gp



def find_top10_avg(x):

    if x.size < 10:
        raise ValueError("Size of array must be larger than 10")

    indices = np.argpartition(x, -10)[-10:]
    top10 = x[indices]

    return np.mean(top10)