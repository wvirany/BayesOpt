import numpy as np
from scipy import stats

def optimization_loop(X, y, X_observed, y_observed, gp, gp_params, acq_func, num_iters=30):
    
    best = []
    num_top10 = 0
    y_init = y.copy()

    # for s in y_observed:
    #     percentile = stats.percentileofscore(y_init, s)
    #     if percentile > 90:
    #         num_top10 += 1
    
    for i in range(num_iters):

        best.append(np.max(y_observed))

        print(f"Iter: {i} | Current best: {np.max(best):0.3f} | Percentile: {stats.percentileofscore(y_init, np.max(best)):0.3f}")

        idx = acq_func(X, gp, gp_params)

        X_new = X.pop(idx)
        y_new = y.pop(idx)

        percentile = stats.percentileofscore(y_init, y_new)

        if percentile > 90:
            num_top10 += 1

        print(f"Observed function value: {y_new:0.3f} | Percentile: {percentile:0.3f}")

        X_observed.append(X_new)
        y_observed.append(y_new)

        gp.set_training_data(X_observed, y_observed)

    return best, X_observed, y_observed, gp, num_top10