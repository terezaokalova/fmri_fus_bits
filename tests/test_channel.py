"""Recovery of the trial-locked channel model from synthetic trials."""

import numpy as np

from fmri_fus_bits.channel import baseline_windows, classify, fit_trial_model
from fmri_fus_bits.noise import contrast_basis


def synthetic(rng: np.random.Generator, eps_scale: float, n: int = 240, signal: float = 0.6):
    n_tau, p, k, n_pre = 10, 20, 4, 2
    tau = np.arange(n_tau) - n_pre + 0.5
    h = np.where(tau > 0, (tau / 4) ** 2 * np.exp(-(tau - 4) / 1.5), 0.0)
    h /= np.linalg.norm(h)
    kb = contrast_basis(n_tau)
    lag = np.abs(np.subtract.outer(np.arange(n_tau), np.arange(n_tau)))
    t_s = 0.8**lag
    t_s = t_s * (n_tau - 1) / np.trace(kb.T @ t_s @ kb)
    v, _ = np.linalg.qr(rng.standard_normal((p, p)))
    ell, f = np.geomspace(20, 1, p), np.linspace(0.9, 0.05, p)
    s = rng.standard_normal((k, p)) * signal
    y = np.arange(n) % k
    c = ell[:, None, None] * ((1 - f)[:, None, None] * np.eye(n_tau) + f[:, None, None] * t_s)
    noise = np.einsum("jts,nsj->ntj", np.linalg.cholesky(c), rng.standard_normal((n, n_tau, p)))
    eps = rng.standard_normal((n, p)) * np.sqrt(eps_scale * ell)
    x_ch = (
        h[None, :, None] * (s[y] @ v + eps)[:, None, :] + noise + 3 * rng.standard_normal((n, 1, p))
    )
    return x_ch @ v.T, y, k, kb, n_pre, h, t_s, ell, f, v


def test_fit_recovers_shape_noise_and_trial_variability() -> None:
    for eps_scale, eps_lo, eps_hi in ((0.0, 0.0, 0.25), (1.0, 0.5, 1.3)):
        x, y, k, kb, n_pre, h, t_s, ell, f, v = synthetic(np.random.default_rng(1), eps_scale)
        z = np.einsum("ts,ntp->nsp", kb, x)
        m = fit_trial_model(z, y, k, kb, n_pre)
        assert float(m.h @ h) > 0.95
        gamma_fit = m.noise.t_s[0, 0] - m.noise.t_s[0, 1:]
        assert np.max(np.abs(gamma_fit - (t_s[0, 0] - t_s[0, 1:]))) < 0.15
        match = np.abs(m.v.T @ v) ** 2
        assert np.all(np.abs(m.noise.ell / (match @ ell) - 1) < 0.2)
        assert np.max(np.abs(m.noise.f - match @ f)) < 0.2
        assert eps_lo <= m.noise.eps.sum() / m.noise.ell.sum() <= eps_hi


def test_heldout_classifier_beats_chance() -> None:
    x, y, k, kb, n_pre, *_ = synthetic(np.random.default_rng(2), 0.0, n=320, signal=1.5)
    train = (np.arange(y.size) // k) % 2 == 0
    z = np.einsum("ts,ntp->nsp", kb, x[train])
    m = fit_trial_model(z, y[train], k, kb, n_pre)
    windows = baseline_windows(kb.shape[0], n_pre)
    pred = classify(m, x[~train], windows)
    acc = (pred == y[~train][:, None]).mean(0)
    assert acc[-1] > 0.6
    assert acc[0] < 0.4
