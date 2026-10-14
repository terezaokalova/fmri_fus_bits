"""Information in a Gaussian class constellation.

`mixture_mi_bits` is I(Y; X) for Y uniform over K classes and X | Y = y ~ N(mu_y, I), with
the means given only through their squared distances; `crossnobis` estimates those squared
distances without bias from two independent estimates of the class means.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]


def embed_squared_distances(d2: FloatArray) -> FloatArray:
    """Classical MDS of a K x K squared-distance matrix.

    Negative entries are set to zero and negative eigenvalues of the double-centred Gram
    matrix are clipped, which gives the nearest Euclidean configuration. Returns K x K
    coordinates whose pairwise squared distances reproduce a Euclidean `d2` exactly.
    """
    k = d2.shape[0]
    if d2.shape != (k, k):
        raise ValueError("d2 must be a square matrix")
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 0.0)
    j = np.eye(k) - 1.0 / k
    w, v = np.linalg.eigh(-0.5 * j @ d2 @ j)
    return np.asarray(v * np.sqrt(np.clip(w, 0.0, None)), dtype=np.float64)


def mixture_mi_bits(d2: FloatArray, rng: np.random.Generator, n_samples: int = 20000) -> float:
    """Monte Carlo I(Y; X) in bits for equiprobable classes with unit-covariance Gaussian
    likelihoods whose means have squared distances `d2` (embedded by
    `embed_squared_distances`); `n_samples` draws of X per class."""
    mu = embed_squared_distances(d2)
    k = mu.shape[0]
    cond_entropy = 0.0
    for y in range(k):
        x = mu[y] + rng.standard_normal((n_samples, k))
        logits = -0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1)
        cond_entropy += float(-(logits[:, y] - logsumexp(logits, axis=1)).mean())
    return float(np.log2(k) - cond_entropy / k / np.log(2))


def crossnobis(s_a: FloatArray, s_b: FloatArray, metric: FloatArray) -> FloatArray:
    """Cross-validated squared Mahalanobis distances between class means.

    `s_a` and `s_b` (K x p) are independent estimates of the class means and `metric` (p x p)
    is fixed (not estimated from either); entry (i, j) is the symmetrised
    (a_i - a_j)^T M (b_i - b_j), unbiased for (mu_i - mu_j)^T M (mu_i - mu_j).
    """
    if s_a.shape != s_b.shape or metric.shape != (s_a.shape[1], s_a.shape[1]):
        raise ValueError("s_a, s_b must be K x p and metric p x p")
    da = s_a[:, None, :] - s_a[None, :, :]
    db = s_b[:, None, :] - s_b[None, :, :]
    m_sym = 0.5 * (metric + metric.T)
    return np.asarray(np.einsum("ijp,pq,ijq->ij", da, m_sym, db), dtype=np.float64)
