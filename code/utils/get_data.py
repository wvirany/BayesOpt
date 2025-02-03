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


def get_data(dataset=DATASET, endpoint='LOG_SOLUBILITY'):

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

    smiles_train, smiles_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return smiles_train, smiles_test, y_train, y_test