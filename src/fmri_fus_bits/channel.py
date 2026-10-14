"""Trial-locked hemodynamic channel: one response shape, white plus slow noise.

Trial b with class y_b, in the d contrasts of a window of n_tau bins (offset projected out)
and P spatial components, is

    z_b = mu + h' (s_y + eps_b)^T + noise_b,

with h zero before the cue (n_pre bins; targets are random across trials, so no class
information precedes the cue), unit norm and positive peak, and noise and eps as in
`fmri_fus_bits.noise`. The fit alternates the noise model (cross-fitted within-class
covariances) with a per-channel GLS rank-1 fit of the class means for h. Per channel the
amplitude estimate a_bj = h'^T C_j^-1 (z_bj - mu_j) / g_j has variance eps_j + l_j / g_j.
"""

from dataclasses import dataclass

import numpy as np

from fmri_fus_bits.noise import (
    FloatArray,
    IntArray,
    WhiteSlowNoise,
    channel_covariances,
    contrast_basis,
    cross_fitted_covariances,
    initial_slow,
    solve_channels,
    update_noise,
)


@dataclass(frozen=True)
class TrialModel:
    """Fitted channel. `v` (p x p) maps component coordinates to channels (x_ch = x @ v);
    `s_y` (k x p) and `w` (p x p, within-class covariance of the amplitudes) are in component
    coordinates; `mu` (d x p) is the contrast-space mean; `kb` the window's contrast basis."""

    kb: FloatArray
    n_pre: int
    mu: FloatArray
    h: FloatArray
    noise: WhiteSlowNoise
    v: FloatArray
    s_y: FloatArray
    w: FloatArray
    last_delta_h: float
    last_delta_t: float

    @property
    def hp(self) -> FloatArray:
        return np.asarray(self.kb.T @ self.h, dtype=np.float64)


def causal_unit(h: FloatArray) -> FloatArray:
    h = h / np.linalg.norm(h)
    return np.asarray(h * np.sign(h[np.argmax(np.abs(h))]), dtype=np.float64)


def amplitudes(z_ch: FloatArray, hp: FloatArray, c: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-trial, per-channel GLS amplitudes (n, p) and gains g_j for channel-coordinate
    contrast data z_ch (n, d, p) and unit-level channel covariances c (p, d, d)."""
    cinv_h = solve_channels(c, hp)
    g = cinv_h @ hp
    return np.asarray(np.einsum("pt,ntp->np", cinv_h, z_ch) / g, dtype=np.float64), g


def fit_trial_model(
    z: FloatArray,
    y: IntArray,
    k: int,
    kb: FloatArray,
    n_pre: int,
    n_iter: int = 30,
    h_inner: int = 5,
    n_freq: int = 80,
) -> TrialModel:
    """Fit on contrast-space trials z (n, d, p) with labels y in 0..k-1."""
    n, d, p = z.shape
    n_tau = kb.shape[0]
    post = np.eye(n_tau)[:, n_pre:]
    amat = kb.T @ post
    counts = np.bincount(y, minlength=k).astype(np.float64)
    if counts.size != k or np.any(counts < 2):
        raise ValueError("labels must be 0..k-1 with at least 2 trials per class")
    mu = z.mean(0)
    zc = z - mu
    means = np.stack([zc[y == c].mean(0) for c in range(k)])
    v, cov = cross_fitted_covariances(z, y, k)
    qa, ra = np.linalg.qr(amat)
    u, _, _ = np.linalg.svd(
        qa.T @ np.transpose(means, (1, 0, 2)).reshape(d, -1), full_matrices=False
    )
    h = causal_unit(post @ np.linalg.solve(ra, u[:, 0]))
    t_s, _ = initial_slow(cov, kb, n_freq)
    m_ch = np.einsum("ktp,pj->ktj", means, v)
    deltas = (np.inf, np.inf)
    noise = None
    for _ in range(n_iter):
        h_prev, t_prev = h.copy(), t_s.copy()
        noise = update_noise(cov, kb.T @ h, t_s, kb, n_freq)
        t_s = noise.t_s
        c = noise.contrast_covariances(kb)
        if np.linalg.eigvalsh(c).min() <= 0:
            raise ValueError("a channel covariance is not positive definite")
        cinv_a = np.linalg.solve(c, np.broadcast_to(amat, (p,) + amat.shape))
        for _ in range(h_inner):
            hp = kb.T @ h
            cinv_h = solve_channels(c, hp)
            sc = np.einsum("jt,ktj->kj", cinv_h, m_ch) / (cinv_h @ hp)
            lhs = np.einsum(
                "j,tr,jts->rs", (counts[:, None] * sc**2).sum(0) / noise.ell, amat, cinv_a
            )
            rhs = np.einsum("kj,jtr,ktj->r", counts[:, None] * sc / noise.ell, cinv_a, m_ch)
            h = causal_unit(post @ np.linalg.solve(lhs, rhs))
        deltas = (float(np.linalg.norm(h - h_prev)), float(np.abs(t_s - t_prev).max()))
    if noise is None:
        raise ValueError("n_iter must be positive")
    a, _ = amplitudes(np.einsum("ntp,pj->ntj", zc, v), kb.T @ h, noise.contrast_covariances(kb))
    s_ch = np.stack([a[y == c].mean(0) for c in range(k)])
    r = a - s_ch[y]
    w_ch = r.T @ r / (n - k)
    return TrialModel(
        kb=kb,
        n_pre=n_pre,
        mu=mu,
        h=h,
        noise=noise,
        v=v,
        s_y=s_ch @ v.T,
        w=v @ w_ch @ v.T,
        last_delta_h=deltas[0],
        last_delta_t=deltas[1],
    )


def window_gains(model: TrialModel, idx: list[int]) -> FloatArray:
    """Per-channel gains g_j(S) for the bins idx, offset projected out within the window."""
    ks = contrast_basis(len(idx))
    hs = ks.T @ model.h[idx]
    ts = ks.T @ model.noise.t_s[np.ix_(idx, idx)] @ ks
    return np.asarray(
        solve_channels(channel_covariances(model.noise.f, ts), hs) @ hs, dtype=np.float64
    )


def amplitude_variances(model: TrialModel, idx: list[int]) -> FloatArray:
    """Per-channel variance of the amplitude estimate from bins idx: eps_j + l_j / g_j(S)."""
    return np.asarray(
        model.noise.eps + model.noise.ell / window_gains(model, idx), dtype=np.float64
    )


def baseline_windows(n_tau: int, n_pre: int) -> list[list[int]]:
    """The pre-cue bins plus one post-cue bin, for each post-cue bin; then the full window."""
    return [list(range(n_pre)) + [j] for j in range(n_pre, n_tau)] + [list(range(n_tau))]


def classify(model: TrialModel, x: FloatArray, windows: list[list[int]]) -> IntArray:
    """Bayes classifier of the fitted model on original-bin data x (n, n_tau, p) (any
    per-trial offset); predictions (n, len(windows))."""
    s_ch = model.s_y @ model.v
    mu_full = model.kb @ model.mu
    preds = []
    for idx in windows:
        ks = contrast_basis(len(idx))
        hs = ks.T @ model.h[idx]
        ts = ks.T @ model.noise.t_s[np.ix_(idx, idx)] @ ks
        cinv_h = solve_channels(channel_covariances(model.noise.f, ts), hs)
        g = cinv_h @ hs
        ys = np.einsum("ts,ntp,pj->nsj", ks, x[:, idx] - mu_full[idx], model.v)
        a = np.einsum("js,nsj->nj", cinv_h, ys) / g
        var = model.noise.eps + model.noise.ell / g
        preds.append(np.argmax(-0.5 * (((a[:, None, :] - s_ch[None]) ** 2) / var).sum(-1), axis=1))
    return np.asarray(np.stack(preds, axis=1), dtype=np.int64)


def discriminant_variance_ratio(model: TrialModel, other_s: FloatArray) -> float:
    """Measured over modelled variance of the full-window amplitude estimate along
    u = (N_a + Sigma_eps)^-1 (s_i - s_j), u from independent class means `other_s`; mean over
    class pairs. 1 when the model is right."""
    var = amplitude_variances(model, list(range(model.kb.shape[0])))
    w_ch = model.v.T @ model.w @ model.v
    vals = []
    k = other_s.shape[0]
    for i in range(k):
        for j in range(i + 1, k):
            u = ((other_s[i] - other_s[j]) @ model.v) / var
            vals.append(float(u @ w_ch @ u / (u @ (var * u))))
    return float(np.mean(vals))
