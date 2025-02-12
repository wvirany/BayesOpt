import numpy as np
import jax.numpy as jnp
import pandas as pd
import sklearn.metrics

import tanimoto_gp
from utils.misc import smiles_to_fp, optimize_params

from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")



def init_gp(smiles_train, y_train):
    
    gp = tanimoto_gp.TanimotoGP(smiles_to_fp, smiles_train, y_train)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(1.0), raw_noise=jnp.asarray(1e-2))
    gp_params = optimize_params(gp, gp_params, tol=1e-3, max_iters=10000)

    return gp, gp_params



def get_dataset(n_train=10000):
    dataset_path = Path('data/dockstring-dataset.tsv')
    assert dataset_path.exists()

    dataset_split_path = Path('data/cluster_split.tsv')
    assert dataset_split_path.exists()

    df = pd.read_csv(dataset_path, sep="\t")

    splits = (
        pd.read_csv(dataset_split_path, sep="\t")
        .loc[df.index]
    )

    # Create train and test datasets
    df_train = df[splits["split"] == "train"]
    df_test = df[splits["split"] == "test"]

    df_train_10k = df_train.sample(n=n_train, random_state=42)

    smiles_train = df_train_10k["smiles"].values
    smiles_test = df_test["smiles"].values

    y_train = np.minimum(df_train_10k['PARP1'].values, 5.0)
    y_test = np.minimum(df_test['PARP1'].values, 5.0)

    smiles_train = smiles_train[~np.isnan(y_train)]
    y_train_nonan = y_train[~np.isnan(y_train)]

    smiles_test = smiles_test[~np.isnan(y_test)]
    y_test_nonan = y_test[~np.isnan(y_test)]

    print(len(smiles_train), len(smiles_test))


    return smiles_train, smiles_test, y_train_nonan, y_test_nonan



def main(n_train=10000):

    smiles_train, smiles_test, y_train, y_test = get_dataset(n_train=n_train)

    gp, gp_params = init_gp(smiles_train, y_train)

    mean, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)

    r2 = sklearn.metrics.r2_score(y_test, mean)

    print(f"R2 Score: {r2}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=10000)

    args = parser.parse_args()

    main(n_train=args.n_train)