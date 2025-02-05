import numpy as np
from sklearn.model_selection import train_test_split

import polaris as po
from polaris.hub.client import PolarisHubClient

from rdkit import Chem
from rdkit.Chem import Crippen

import warnings
warnings.filterwarnings("ignore")


# Login to Polaris
client = PolarisHubClient()
client.login()


DATASET = "biogen/adme-fang-v1"


def split_data(X, y, split_method, frac, as_list=False, random_seed=1):

    if split_method == 'random':
        X, X_observed, y, y_observed = train_test_split(X, y, test_size=frac, random_state=random_seed)

    elif split_method == 'n_worst':
        
        n = int(frac * len(X))
        sorted_indices = np.argsort(y)

        lowest_indices = sorted_indices[:n] # Lowest n values
        rest_indices = sorted_indices[n:]   # All other indices

        X, X_observed, y, y_observed = X[rest_indices], X[lowest_indices], y[rest_indices], y[lowest_indices]

    if as_list:
        return list(X), list(X_observed), list(y), list(y_observed)
    else:
        return X, X_observed, y, y_observed


def get_data(dataset=DATASET, endpoint='LOG_SOLUBILITY', split=False, split_method='random', frac=0.1, as_list=False):

    dataset = po.load_dataset(dataset)

    # Get all SMILES strings and logS values from dataset
    X = [dataset.get_data(
        row=dataset.rows[i],
        col='MOL_smiles'
        ) for i in range(dataset.size()[0])]

    y = [dataset.get_data(
        row=dataset.rows[i],
        col=endpoint
        ) for i in range(dataset.size()[0])]
    
    # Filter molecules with NaN logP values
    filter = ~np.isnan(y)

    X = np.array([i for idx, i in enumerate(X) if filter[idx]])
    y = np.array([i for idx, i in enumerate(y) if filter[idx]])

    if split:
        return split_data(X, y, split_method=split_method, frac=frac, as_list=as_list)
    else:
        if as_list:
            return list(X), list(y)
        else:
            return X, y