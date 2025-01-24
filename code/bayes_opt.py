import numpy as np
import jax.numpy as jnp
import pandas as pd

import polaris as po
from polaris.hub.client import PolarisHubClient

import tanimoto_gp
from utils import smiles_to_fp, optimize_params
import acq_funcs

import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial
import warnings
warnings.filterwarnings("ignore")

# Set up Seaborn plotting style
sns.set_style("darkgrid",
              {"axes.facecolor": ".95",
               "axes.edgecolor": "#000000",
               "grid.color": "#EBEBE7",
               "font.family": "serif",
               "axes.labelcolor": "#000000",
               "xtick.color": "#000000",
               "ytick.color": "#000000",
               "grid.alpha": 0.4 })
sns.set_palette('muted')


client = PolarisHubClient()
client.login()

dataset = po.load_dataset("biogen/adme-fang-v1")

print(f"Dataset size: {dataset.size()}")

# Example of how to get first element
dataset.get_data(
    row=dataset.rows[0],
    col=dataset.columns[1]
)

# Get all SMILES strings and logP values from dataset
X = [dataset.get_data(
    row=dataset.rows[i],
    col='MOL_smiles'
    ) for i in range(dataset.size()[0])]

y = [dataset.get_data(
    row=dataset.rows[i],
    col='LOG_SOLUBILITY'
    ) for i in range(dataset.size()[0])]


# Filter molecules with NaN logP values
filter = ~np.isnan(y)

X = [i for idx, i in enumerate(X) if filter[idx]]
y = [i for idx, i in enumerate(y) if filter[idx]]

print(f"Number of molecules after filtering NaNs: {len(X)}")


split = int(.2 * len(X))

X_observed, X_unlabeled = X[:split], X[split:]
y_observed, y_unlabeled = y[:split], y[split:]

gp = tanimoto_gp.TanimotoGP(smiles_to_fp, X_observed, y_observed)
gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(1.0), raw_noise=jnp.asarray(1e-2))
gp_params = optimize_params(gp, gp_params)

best = []

for i in range(30):

    best.append(np.max(y_observed))

    ucb = upper_confidence_bound(X_unlabeled, gp, gp_params)
    idx = np.argmax(ucb)

    X_new = X_unlabeled.pop(idx)
    y_new = y_unlabeled.pop(idx)

    print(idx, X_new, y_new)

    X_observed.append(X_new)
    y_observed.append(y_new)

    gp = tanimoto_gp.TanimotoGP(smiles_to_fp, X_observed, y_observed)


split = int(.2 * len(X))

X_observed, X_unlabeled = X[:split], X[split:]
y_observed, y_unlabeled = y[:split], y[split:]

gp = tanimoto_gp.TanimotoGP(smiles_to_fp, X_observed, y_observed)
gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(1.0), raw_noise=jnp.asarray(1e-2))
gp_params = optimize_params(gp, gp_params)

best_uniform = []

for i in range(30):

    best_uniform.append(np.max(y_observed))

    idx = np.random.randint(len(X_unlabeled))

    X_new = X_unlabeled.pop(idx)
    y_new = y_unlabeled.pop(idx)

    print(idx, X_new, y_new)

    X_observed.append(X_new)
    y_observed.append(y_new)

    gp = tanimoto_gp.TanimotoGP(smiles_to_fp, X_observed, y_observed)


plt.plot(np.arange(len(best)), best, lw=1, label='UCB')
plt.scatter(np.arange(len(best)), best, s=5)

plt.plot(np.arange(len(best)), best_uniform, lw=1, label='Uniform')
plt.scatter(np.arange(len(best)), best_uniform, s=5)

plt.xlabel('Iteration')
plt.ylabel('LogP')
plt.title('Max logP value in set at each iteration')
plt.legend()

plt.show()


print(np.quantile(y, .995))