import jax.numpy as jnp

import tanimoto_gp
from tanimoto_gp import ZeroMeanTanimotoGP, TanimotoGP_Params
from utils.misc import smiles_to_fp

import pickle
from dataclasses import dataclass

@dataclass
class GPCheckpoint:
    smiles_train: list[str]
    y_train: list[float]
    y_mean: float
    K_train_train: jnp.ndarray
    gp_params: TanimotoGP_Params

def save_gp_checkpoint(gp: ZeroMeanTanimotoGP, gp_params: TanimotoGP_Params, path: str):
    checkpoint = GPCheckpoint(
        smiles_train=gp._smiles_train,
        y_train=gp._y_train.tolist(),
        y_mean=float(gp._y_mean),
        K_train_train=gp._K_train_train,
        gp_params=gp_params
    )

    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_gp_checkpoint(path: str) -> tuple[ZeroMeanTanimotoGP, TanimotoGP_Params]:
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    gp = ZeroMeanTanimotoGP(smiles_to_fp, [], [])

    gp._smiles_train = checkpoint.smiles_train
    gp._y_train = jnp.asarray(checkpoint.y_train)
    gp._y_mean = checkpoint.y_mean
    gp._y_centered = gp._y_train - gp._y_mean
    gp._K_train_train = checkpoint.K_train_train
    gp._fp_train = [smiles_to_fp(s) for s in checkpoint.smiles_train]

    return gp, checkpoint.gp_params