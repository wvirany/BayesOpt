import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import polaris as po
# from polaris.hub.client import PolarisHubClient

from rdkit import Chem
from rdkit.Chem import Crippen

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")



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


def get_data(dataset='biogen/adme-fang-v1', endpoint='LOG_SOLUBILITY', split=False, split_method='random', frac=0.1, as_list=False):

    # Login to Polaris
    client = PolarisHubClient()
    client.login()

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
        


def get_dockstring_dataset(n_train=10000, target='PARP1', seed=42):
    dataset_path = Path('/projects/wavi0116/code/BayesOpt/data/dockstring-dataset.tsv')
    assert dataset_path.exists()

    dataset_split_path = Path('/projects/wavi0116/code/BayesOpt/data/cluster_split.tsv')
    assert dataset_split_path.exists()

    df = pd.read_csv(dataset_path, sep="\t")

    splits = (
        pd.read_csv(dataset_split_path, sep="\t")
        .loc[df.index]
    )

    # Create train and test datasets
    df_train = df[splits["split"] == "train"]
    df_test = df[splits["split"] == "test"]

    if n_train < len(df_train):
        df_train = df_train.sample(n=n_train, random_state=seed)

    smiles_train = df_train["smiles"].values
    smiles_test = df_test["smiles"].values

    y_train = np.minimum(df_train[target].values, 5.0)
    y_test = np.minimum(df_test[target].values, 5.0)

    smiles_train = smiles_train[~np.isnan(y_train)]
    y_train_nonan = y_train[~np.isnan(y_train)]

    smiles_test = smiles_test[~np.isnan(y_test)]
    y_test_nonan = y_test[~np.isnan(y_test)]

    return smiles_train, smiles_test, y_train_nonan, y_test_nonan