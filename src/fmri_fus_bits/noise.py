"""White plus slow noise for trial-locked windows of binned hemodynamic data.

A trial window has n bins; a per-trial offset is unknown, so data are used through the d = n - 1
contrasts of an orthonormal basis K of the complement of the constant. In a spatial channel
basis V (orthonormal columns), channel j has contrast-space temporal covariance

    C_j = l_j [(1 - f_j) I + f_j T_s'],   T_s' = K^T T_s K,   tr T_s' = d,

with T_s a stationary covariance in the original bins with a non-negative spectrum on a
frequency grid (positive semidefinite by construction; zero frequency is the offset and is
not identified). Trial-to-trial amplitude variability along the response shape h adds
eps_j h' h'^T, h' = K^T h.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def contrast_basis(n: int) -> FloatArray:
    """Orthonormal basis (n x n-1) of the complement of the constant vector."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, : n - 1]]))
    return np.asarray(q[:, 1:], dtype=np.float64)


def spectral_basis(n: int, n_freq: int) -> tuple[FloatArray, FloatArray]:
    """Angular frequencies omega_m = pi m / n_freq (m = 1..n_freq, radians per bin) and the
    n x n matrices cos(omega_m (t - t')). Non-negative combinations of them are the stationary
    covariances with a non-negative spectrum on the grid (Bochner, Caratheodory-Toeplitz)."""
    omegas = np.pi * np.arange(1, n_freq + 1) / n_freq
    lag = np.subtract.outer(np.arange(n), np.arange(n))
    return omegas, np.cos(omegas[:, None, None] * lag[None])


def channel_covariances(f: FloatArray, ts_c: FloatArray) -> FloatArray:
    """Unit-level channel covariances (p, m, m): (1 - f_j) I + f_j T_s'."""
    eye = np.eye(ts_c.shape[0])
    return np.asarray((1.0 - f)[:, None, None] * eye + f[:, None, None] * ts_c, dtype=np.float64)


def solve_channels(c: FloatArray, b: FloatArray) -> FloatArray:
    """Rows c_j^-1 b for c (p, m, m) and b (m,)."""
    rhs = np.broadcast_to(b[:, None], (c.shape[0], b.size, 1))
    return np.asarray(np.linalg.solve(c, rhs)[..., 0], dtype=np.float64)


def alternating_folds(y: IntArray, k: int) -> IntArray:
    """Fold 0/1 alternating in trial order within each class."""
    folds = np.empty(y.size, dtype=np.int64)
    for c in range(k):
        folds[y == c] = np.arange(int(np.sum(y == c))) % 2
    return folds


def cross_fitted_covariances(z: FloatArray, y: IntArray, k: int) -> tuple[FloatArray, FloatArray]:
    """Spatial channel basis V (p x p, columns by decreasing level) and channel temporal
    covariances (p, d, d) of the within-class deviations of z (n, d, p).

    Class means are removed within each of two folds; eigenvectors of one fold's pooled
    spatial covariance are paired with temporal covariances along them from the other fold
    (variances along eigenvectors chosen on the same data are biased low in the quiet
    channels). V diagonalises the fold-averaged spatial covariance and the channel
    covariances are obtained by projection. Expected value per channel: C_j + eps_j h' h'^T.
    """
    d = z.shape[1]
    folds = alternating_folds(y, k)
    dev: list[tuple[FloatArray, int]] = []
    for fold in (0, 1):
        zf, yf = z[folds == fold], y[folds == fold]
        if any(np.sum(yf == c) < 1 for c in range(k)):
            raise ValueError("every class needs at least one trial in each fold")
        means = np.stack([zf[yf == c].mean(0) for c in range(k)])
        dev.append((zf - means[yf], zf.shape[0] - k))
    fold_v, fold_c = [], []
    for fold in (0, 1):
        (din, _), (dout, dof_out) = dev[fold], dev[1 - fold]
        if dof_out <= 0:
            raise ValueError("each fold needs more trials than classes")
        _, vf = np.linalg.eigh(np.einsum("ntp,ntq->pq", din, din))
        proj = np.einsum("ntp,pj->ntj", dout, vf)
        fold_v.append(vf)
        fold_c.append(np.einsum("ntj,nsj->jts", proj, proj) / dof_out)
    level = 0.5 * sum(
        vf @ np.diag(np.trace(cf, axis1=1, axis2=2) / d) @ vf.T for vf, cf in zip(fold_v, fold_c)
    )
    lvals, v = np.linalg.eigh(level)
    v = v[:, np.argsort(lvals)[::-1]]
    cov = 0.5 * sum(np.einsum("ji,its->jts", (v.T @ vf) ** 2, cf) for vf, cf in zip(fold_v, fold_c))
    return np.asarray(v, dtype=np.float64), np.asarray(cov, dtype=np.float64)


@dataclass(frozen=True)
class WhiteSlowNoise:
    """Per-channel level l, slow fraction f and trial variability eps (p,), slow covariance
    t_s (n x n, original bins, tr K^T t_s K = d) and its spectrum on `omegas`."""

    ell: FloatArray
    f: FloatArray
    eps: FloatArray
    t_s: FloatArray
    spectrum: FloatArray
    omegas: FloatArray

    def contrast_covariances(self, kb: FloatArray) -> FloatArray:
        """Unit-level channel covariances in the contrast space of kb."""
        return channel_covariances(self.f, kb.T @ self.t_s @ kb)


def slow_from_spectrum(
    spec: FloatArray, cosines: FloatArray, kb: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    """Slow covariance from spectral weights scaled to tr K^T T_s K = d: (t_s, spec, kappa),
    kappa the applied scale."""
    t_s = np.einsum("m,mts->ts", spec, cosines)
    trace = float(np.trace(kb.T @ t_s @ kb))
    if trace <= 0:
        raise ValueError("slow spectrum has no power outside zero frequency")
    kappa = kb.shape[1] / trace
    return (
        np.asarray(t_s * kappa, dtype=np.float64),
        np.asarray(spec * kappa, dtype=np.float64),
        kappa,
    )


def initial_slow(cov: FloatArray, kb: FloatArray, n_freq: int) -> tuple[FloatArray, FloatArray]:
    """Separable start: non-negative spectrum fitted to the channel-averaged normalised
    covariance (T_s = I would coincide with the white term)."""
    scale = np.trace(cov, axis1=1, axis2=2) / kb.shape[1]
    _, cosines = spectral_basis(kb.shape[0], n_freq)
    design = np.stack([(kb.T @ b @ kb).ravel() for b in cosines], axis=1)
    spec, _ = nnls(design, (cov / scale[:, None, None]).mean(0).ravel())
    t_s, spec_scaled, _ = slow_from_spectrum(np.asarray(spec, dtype=np.float64), cosines, kb)
    return t_s, spec_scaled


def update_noise(
    cov: FloatArray, hp: FloatArray, t_s: FloatArray, kb: FloatArray, n_freq: int
) -> WhiteSlowNoise:
    """One alternation step given h' and the current slow covariance: per-channel
    non-negative least squares of the normalised channel covariance on I, T_s' and h'h'^T,
    then the slow spectrum by non-negative least squares across channels."""
    d = kb.shape[1]
    scale = np.trace(cov, axis1=1, axis2=2) / d
    cov_n = cov / scale[:, None, None]
    ts_c = kb.T @ t_s @ kb
    cols = np.column_stack([np.eye(d).ravel(), ts_c.ravel(), np.outer(hp, hp).ravel()])
    wse = np.array([nnls(cols, cov_n[j].ravel())[0] for j in range(cov.shape[0])])
    if not np.any(wse[:, 1] > 0):
        raise ValueError("no channel has a slow component")
    if np.any(wse[:, :2].sum(1) <= 0):
        raise ValueError("a channel has zero noise weight")
    omegas, cosines = spectral_basis(kb.shape[0], n_freq)
    design = np.stack([(wse[:, 1, None, None] * (kb.T @ b @ kb)).ravel() for b in cosines], axis=1)
    target = cov_n - wse[:, 0, None, None] * np.eye(d) - wse[:, 2, None, None] * np.outer(hp, hp)
    spec_raw, _ = nnls(design, target.ravel())
    t_new, spec, kappa = slow_from_spectrum(np.asarray(spec_raw, dtype=np.float64), cosines, kb)
    white, slow = wse[:, 0] * scale, wse[:, 1] / kappa * scale
    return WhiteSlowNoise(
        ell=white + slow,
        f=slow / (white + slow),
        eps=wse[:, 2] * scale,
        t_s=t_new,
        spectrum=spec,
        omegas=omegas,
    )


def ar1_bin_covariance(
    rho_per_frame: float, frame_s: float, edges: FloatArray, sub: int = 20
) -> FloatArray:
    """Covariance of bin averages (bins given by edges, in s) of a unit-variance process with
    continuous-time correlation rho^(|tau| / frame_s), scaled to tr K^T T K = n - 1 (the slow
    covariance convention, so it can replace t_s with f = 1)."""
    width = np.diff(edges)
    tt = edges[:-1, None] + width[:, None] * ((np.arange(sub) + 0.5) / sub)[None, :]
    lag = np.abs(tt[:, None, :, None] - tt[None, :, None, :])
    cov = (rho_per_frame ** (lag / frame_s)).mean(axis=(2, 3))
    kb = contrast_basis(edges.size - 1)
    return np.asarray(cov * kb.shape[1] / np.trace(kb.T @ cov @ kb), dtype=np.float64)
