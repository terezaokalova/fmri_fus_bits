"""Identities and limits of the constellation information measures."""

import numpy as np

from fmri_fus_bits.information import crossnobis, embed_squared_distances, mixture_mi_bits


def binary_mi_bits(d: float) -> float:
    """Two classes at distance d: I = 1 - E[log2(1 + exp(-(d z + d^2 / 2)))], z ~ N(0, 1),
    by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()
    return float(1.0 - w @ np.log2(1.0 + np.exp(-(d * x + d**2 / 2))))


def test_embedding_reproduces_euclidean_distances() -> None:
    pts = np.random.default_rng(0).standard_normal((6, 4))
    d2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    e = embed_squared_distances(d2)
    assert np.allclose(((e[:, None] - e[None]) ** 2).sum(-1), d2)


def test_mixture_mi_limits_and_binary_case() -> None:
    rng = np.random.default_rng(1)
    assert abs(mixture_mi_bits(np.zeros((4, 4)), rng)) < 1e-12
    assert abs(mixture_mi_bits(1e4 * (1 - np.eye(4)), rng) - 2.0) < 1e-6
    for d in (0.5, 1.5, 3.0):
        d2 = np.array([[0.0, d**2], [d**2, 0.0]])
        assert abs(mixture_mi_bits(d2, rng, n_samples=200000) - binary_mi_bits(d)) < 0.005


def test_crossnobis_is_unbiased() -> None:
    rng = np.random.default_rng(2)
    k, p, n = 4, 6, 10
    mu = rng.standard_normal((k, p))
    metric = np.diag(rng.uniform(0.5, 2.0, p))
    true = np.einsum("ijp,pq,ijq->ij", mu[:, None] - mu[None], metric, mu[:, None] - mu[None])
    est = np.mean(
        [
            crossnobis(
                mu + rng.standard_normal((k, p)) / np.sqrt(n),
                mu + rng.standard_normal((k, p)) / np.sqrt(n),
                metric,
            )
            for _ in range(4000)
        ],
        axis=0,
    )
    assert np.allclose(est, true, atol=0.05)
