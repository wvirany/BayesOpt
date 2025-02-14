import numpy as np
import jax.numpy as jnp
import pandas as pd
import sklearn.metrics

from rdkit.Chem import DataStructs

import tanimoto_gp
from utils.misc import smiles_to_fp, optimize_params
from utils import GPCheckpoint

from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")



def init_gp(smiles_train, y_train):
    
    gp = tanimoto_gp.TanimotoGP(smiles_to_fp, smiles_train, y_train)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(1.0), raw_noise=jnp.asarray(1e-2))
    gp_params = optimize_params(gp, gp_params, tol=1e-3, max_iters=10000)

    return gp, gp_params



def get_dataset(n_train=10000, target='PARP1'):
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

    y_train = np.minimum(df_train_10k[target].values, 5.0)
    y_test = np.minimum(df_test[target].values, 5.0)

    smiles_train = smiles_train[~np.isnan(y_train)]
    y_train_nonan = y_train[~np.isnan(y_train)]

    smiles_test = smiles_test[~np.isnan(y_test)]
    y_test_nonan = y_test[~np.isnan(y_test)]

    return smiles_train, smiles_test, y_train_nonan, y_test_nonan



def main(from_checkpoint=False, path=None, n_train=10000, target='PARP1'):

    smiles_train, smiles_test, y_train, y_test = get_dataset(n_train=n_train, target=target)

    print(f"Train size: {len(smiles_train)}\nTest size: {len(smiles_test)}")


    if from_checkpoint:
        gp, gp_params = GPCheckpoint.load_gp_checkpoint(path)
    else:
        gp, gp_params = init_gp(smiles_train, y_train)
        GPCheckpoint.save_gp_checkpoint(gp, gp_params, path)
    
    print(gp_params)

    smiles_test = smiles_test
    y_test = y_test
    
    mean, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)

    r2 = sklearn.metrics.r2_score(y_test, mean)

    print(f"R2 Score: {r2}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--target", type=str, default='PARP1')

    args = parser.parse_args()

    # If loading checkpoint, model path must be included
    if args.path is None:
        parser.error("--path must be specified")

    main(from_checkpoint=args.from_checkpoint, path=args.path, n_train=args.n_train, target=args.target)