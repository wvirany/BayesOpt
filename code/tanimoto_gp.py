from typing import Any, Callable, NamedTuple

import kern_gp as kgp
from jax import numpy as jnp
from jax.nn import softplus
from jax.scipy.stats import multivariate_normal
from rdkit import DataStructs

TRANSFORM = softplus  # fixed transform function


class TanimotoGP_Params(NamedTuple):
    # Inverse softplus of GP parameters
    raw_amplitude: jnp.ndarray
    raw_noise: jnp.ndarray


class BaseTanimotoGP:
    def __init__(self, fp_func: Callable[[str], Any], smiles_train: list[str], y_train):
        self._fp_func = fp_func
        self.set_training_data(smiles_train, y_train)

    def _setup_kernel(self, smiles_train: list[str]):
        self._fp_train = [self._fp_func(smiles) for smiles in smiles_train]
        self._K_train_train = jnp.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in self._fp_train]
        )

    def marginal_log_likelihood(self, params: TanimotoGP_Params) -> jnp.ndarray:
        return kgp.mll_train(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            y_train=self._get_training_targets(),
        )

    def predict_f(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True):
        fp_test = [self._fp_func(smiles) for smiles in smiles_test]
        K_test_train = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in fp_test])
        K_test_test = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, fp_test) for fp in fp_test]) if full_covar else jnp.ones((len(smiles_test)), dtype=float)

        return kgp.noiseless_predict(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            k_test_train=K_test_train,
            k_test_test=K_test_test,
            y_train=self._get_training_targets(),
            full_covar=full_covar,
        )

    def test_log_likelihood(self, params, smiles_test, y_test):
        mean, cov = self.predict_y(params, smiles_test, full_covar=True)
        return multivariate_normal.logpdf(y_test, mean=mean, cov=cov)

    def _get_training_targets(self):
        """
        Returns uncentered training data for zero-mean GP, centered training data for adjusted GP
        """
        raise NotImplementedError

    def predict_y(self, params, smiles_test, full_covar=True):
        raise NotImplementedError


class ZeroMeanTanimotoGP(BaseTanimotoGP):
    def set_training_data(self, smiles_train, y_train):
        self._y_train = jnp.asarray(y_train)
        self._setup_kernel(smiles_train)

    def _get_training_targets(self):
        return self._y_train

    def predict_y(self, params, smiles_test, full_covar=True):
        mean, covar = self.predict_f(params, smiles_test, full_covar)
        if full_covar:
            covar = covar + jnp.eye(len(smiles_test)) * TRANSFORM(params.raw_noise)
        else:
            covar += TRANSFORM(params.raw_noise)
        return mean, covar


class TanimotoGP(BaseTanimotoGP):
    def set_training_data(self, smiles_train, y_train):
        self._y_train = jnp.asarray(y_train)
        self._y_mean = jnp.mean(self._y_train)
        self._y_centered = self._y_train - self._y_mean
        self._setup_kernel(smiles_train)

    def _get_training_targets(self):
        return self._y_centered

    def predict_y(self, params, smiles_test, full_covar=True):
        mean, covar = self.predict_f(params, smiles_test, full_covar)
        if full_covar:
            covar = covar + jnp.eye(len(smiles_test)) * TRANSFORM(params.raw_noise)
        else:
            covar += TRANSFORM(params.raw_noise)
        return mean + self._y_mean, covar