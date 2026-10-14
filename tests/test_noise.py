"""Bases of the white plus slow noise model and recovery of its cross-fitted covariances."""

import numpy as np

from fmri_fus_bits.noise import (
    alternating_folds,
    contrast_basis,
    cross_fitted_covariances,
    spectral_basis,
)


def test_contrast_basis_is_orthonormal_complement_of_constant() -> None:
    kb = contrast_basis(7)
    assert np.allclose(kb.T @ kb, np.eye(6))
    assert np.allclose(np.ones(7) @ kb, 0.0)


def test_spectral_basis_is_positive_semidefinite() -> None:
    _, cos = spectral_basis(9, 40)
    weights = np.random.default_rng(0).uniform(0, 1, 40)
    assert np.linalg.eigvalsh(np.einsum("m,mts->ts", weights, cos)).min() > -1e-10


def test_alternating_folds_balance_classes() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    folds = alternating_folds(y, 2)
    assert folds.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_cross_fitted_level_is_unbiased_in_quiet_channels() -> None:
    """Eigenvalues of an in-sample covariance underestimate the quiet directions; the
    cross-fitted channel levels do not (white noise, known spectrum of levels)."""
    rng = np.random.default_rng(3)
    n, d, p, k = 120, 5, 30, 3
    levels = np.geomspace(10.0, 1.0, p)
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    z = (rng.standard_normal((n, d, p)) * np.sqrt(levels)) @ q.T
    y = np.arange(n) % k
    v, cov = cross_fitted_covariances(z, y, k)
    fitted = np.trace(cov, axis1=1, axis2=2) / d
    true_along = np.einsum("pj,p,pj->j", q.T @ v, levels, q.T @ v)
    assert abs(np.mean(fitted[-10:] / true_along[-10:]) - 1.0) < 0.1
    w_in, u_in = np.linalg.eigh(np.einsum("ntp,ntq->pq", z, z) / (n * d))
    oracle = np.einsum("pj,p,pj->j", q.T @ u_in, levels, q.T @ u_in)
    assert np.mean(w_in[:10] / oracle[:10]) < 0.85
