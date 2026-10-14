"""Per-trial information for a trial design: neighbouring trials' responses in the decoding
window handled by zero-forcing, as noise, or by decision feedback.

Trial 0 is decoded from a window of bins (or frames) with a per-trial offset projected out.
Neighbour m contributes r_m (s_{y_m} + eps_m) with an independent uniform target:
- zero-forcing: neighbour amplitudes unknown in every channel and projected out;
- interference as noise: each neighbour adds r_m r_m^T (x) (Sigma_s + Sigma_eps), Sigma_s the
  covariance of the class patterns (the class-common response is known);
- decision feedback: past neighbours known and subtracted, future ones as noise.
These are the standard equalisers of an intersymbol-interference channel.
"""

import numpy as np

from fmri_fus_bits.noise import FloatArray, WhiteSlowNoise, channel_covariances, contrast_basis


def neighbour_onsets(
    period: float, window_end: float, horizon_s: float
) -> tuple[list[float], list[float]]:
    """Onsets m P of past neighbours back to -horizon_s and future ones up to window_end."""
    if not np.isfinite(period):
        return [], []
    m_lo, m_hi = int(np.floor(-horizon_s / period)), int(np.ceil(window_end / period))
    return [m * period for m in range(m_lo, 0)], [m * period for m in range(1, m_hi + 1)]


def zero_forcing_variances(
    noise: WhiteSlowNoise, r: FloatArray, neighbours: list[FloatArray]
) -> FloatArray:
    """Per-channel variance of trial 0's amplitude, eps_j + l_j / g_j, with
    g_j = r^T [C_j^-1 - C_j^-1 X (X^T C_j^-1 X)^+ X^T C_j^-1] r, X = [1, neighbours], in the
    original bins of the window (C_j unit-level, slow part noise.t_s on these bins)."""
    x = np.column_stack([np.ones(r.size)] + neighbours)
    c = channel_covariances(noise.f, noise.t_s)
    cinv_r = np.linalg.solve(c, np.broadcast_to(r[:, None], (c.shape[0], r.size, 1)))[..., 0]
    cinv_x = np.linalg.solve(c, np.broadcast_to(x, (c.shape[0],) + x.shape))
    xtcx = np.einsum("tk,jtl->jkl", x, cinv_x)
    xtcr = np.einsum("jtk,t->jk", cinv_x, r)
    g = cinv_r @ r - np.einsum("jk,jkl,jl->j", xtcr, np.linalg.pinv(xtcx), xtcr)
    g = np.clip(g, 0.0, None)
    with np.errstate(divide="ignore"):
        return np.asarray(noise.eps + np.where(g > 0, noise.ell / g, np.inf), dtype=np.float64)


def pattern_covariance(s_a: FloatArray, s_b: FloatArray) -> FloatArray:
    """Covariance of the class patterns under uniform targets from two independent estimates
    of the class means (symmetrised cross product), projected to the nearest PSD matrix."""
    a, b = s_a - s_a.mean(0), s_b - s_b.mean(0)
    cov = 0.5 * (a.T @ b + b.T @ a) / a.shape[0]
    w, v = np.linalg.eigh(cov)
    return np.asarray((v * np.clip(w, 0.0, None)) @ v.T, dtype=np.float64)


def interference_metric(
    noise: WhiteSlowNoise, r: FloatArray, interferers: list[FloatArray], pattern_cov_ch: FloatArray
) -> FloatArray:
    """Channel-coordinate metric G^T Omega^-1 G for trial 0's class means, with the listed
    neighbours as Gaussian interference (pattern_cov_ch: class-pattern covariance in channel
    coordinates). Omega is the (channel x contrast) covariance after offset projection,
    including trial 0's own eps along r; G = I (x) K^T r."""
    n_t, p = r.size, noise.ell.size
    kb = contrast_basis(n_t)
    c = noise.ell[:, None, None] * channel_covariances(noise.f, noise.t_s)
    d = n_t - 1
    omega = np.zeros((p * d, p * d))
    for j in range(p):
        omega[j * d : (j + 1) * d, j * d : (j + 1) * d] = kb.T @ c[j] @ kb
    rc = kb.T @ r
    eps = np.diag(noise.eps)
    omega += np.kron(eps, np.outer(rc, rc))
    if interferers:
        ri = kb.T @ np.column_stack(interferers)
        omega += np.kron(pattern_cov_ch + eps, ri @ ri.T)
    g = np.kron(np.eye(p), rc[:, None])
    return np.asarray(g.T @ np.linalg.solve(omega, g), dtype=np.float64)


def separable_gains(
    t_corr: FloatArray,
    r: FloatArray,
    past: list[FloatArray],
    future: list[FloatArray],
    lam: float,
    sigmas: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    """Separable noise T (x) Sigma, Sigma-whitened coordinates, class configuration with
    per-axis pattern variances `sigmas` and trial variability lam Sigma: zero-forcing gain
    (D2 = delta2 g / (1 + lam g)) and per-axis interference-as-noise and decision-feedback
    gains (D2_ab = sum_k (E_ak - E_bk)^2 g_k), window offset projected out."""
    n = r.size
    kb = contrast_basis(n)
    tm = kb.T @ t_corr @ kb
    rc = kb.T @ r
    tinv_r = np.linalg.solve(tm, rc)
    g0 = float(rc @ tinv_r)
    nb = [kb.T @ c for c in past + future]
    if nb:
        x = np.column_stack(nb)
        xr = x.T @ tinv_r
        g_zf = g0 - float(xr @ np.linalg.pinv(x.T @ np.linalg.solve(tm, x)) @ xr)
    else:
        g_zf = g0
    rp = kb.T @ np.column_stack(past) if past else np.zeros((n - 1, 0))
    rf = kb.T @ np.column_stack(future) if future else np.zeros((n - 1, 0))
    tin, dfe = [], []
    for s in sigmas:
        om_f = tm + lam * np.outer(rc, rc) + (s + lam) * rf @ rf.T
        dfe.append(float(rc @ np.linalg.solve(om_f, rc)))
        tin.append(float(rc @ np.linalg.solve(om_f + (s + lam) * rp @ rp.T, rc)))
    return max(g_zf, 0.0), np.array(tin), np.array(dfe)
