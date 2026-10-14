"""Hemodynamic kernels and the responses to boxcar drives.

A kernel is represented by its step response S(t) (the integral of the impulse response), so
that the response to a drive on [onset, onset + d) is S(t - onset) - S(t - onset - d) and
kernels with a singular onset (gamma shape below 1) are handled exactly.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import gamma

from fmri_fus_bits.noise import FloatArray

StepResponse = Callable[[FloatArray], FloatArray]


def gamma_step(alpha: float, beta: float, delta: float) -> StepResponse:
    """Step response of a gamma density with shape alpha, scale beta (s) and onset delay
    delta (s); unit area."""

    def step(t: FloatArray) -> FloatArray:
        return np.asarray(gamma.cdf(t - delta, alpha, scale=beta), dtype=np.float64)

    return step


def canonical_bold_step(t: FloatArray) -> FloatArray:
    """Step response of the SPM canonical double gamma (peak 5 s, undershoot ratio 1/6),
    scaled to unit net area so that a sustained drive gives a unit plateau."""
    return np.asarray((gamma.cdf(t, 6) - gamma.cdf(t, 16) / 6) / (5.0 / 6.0), dtype=np.float64)


def boxcar_samples(step: StepResponse, d: float, onset: float, t: FloatArray) -> FloatArray:
    """Response at times t to a unit drive on [onset, onset + d)."""
    return np.asarray(step(t - onset) - step(t - onset - d), dtype=np.float64)


def boxcar_bins(
    step: StepResponse, d: float, onset: float, edges: FloatArray, sub: int = 50
) -> FloatArray:
    """Bin averages over edges of the response to a unit drive on [onset, onset + d)."""
    width = np.diff(edges)
    tt = edges[:-1, None] + width[:, None] * (np.arange(sub) / sub)[None, :]
    return np.asarray(boxcar_samples(step, d, onset, tt).mean(1), dtype=np.float64)


def mean_drive_bins(
    step: StepResponse, drives: FloatArray, edges: FloatArray, resolution_s: float = 0.05
) -> FloatArray:
    """Mean over trials of the binned responses to drives [0, d_b), d_b rounded to the
    resolution."""
    vals, counts = np.unique(np.round(drives / resolution_s) * resolution_s, return_counts=True)
    total = sum(c * boxcar_bins(step, float(d), 0.0, edges) for d, c in zip(vals, counts))
    return np.asarray(total / counts.sum(), dtype=np.float64)


@dataclass(frozen=True)
class KernelFit:
    """Gamma kernel with delay, signal per unit drive `amp`, residual sum of squares over the
    post-onset bins and the fitted response."""

    alpha: float
    beta: float
    delta_s: float
    amp: float
    ssr: float
    fitted: FloatArray

    @property
    def step(self) -> StepResponse:
        return gamma_step(self.alpha, self.beta, self.delta_s)


def fit_gamma_kernel(
    h: FloatArray, drives: FloatArray, edges: FloatArray, alpha: float | None = None
) -> KernelFit:
    """Least-squares gamma kernel (shape, scale, delay, amplitude) such that h on the bins
    with edges >= 0 equals amp x mean_drive_bins; shape fixed if `alpha` is given. Multiple
    starts; bounds shape [0.2, 20], scale [0.05, 30] s, delay [0, 4] s."""
    post = edges[:-1] >= 0
    hp = h[post]

    def unpack(q: FloatArray) -> tuple[float, float, float]:
        a = float(np.exp(q[0])) if alpha is None else alpha
        return a, float(np.exp(q[-2])), float(q[-1])

    def resid(q: FloatArray) -> FloatArray:
        r = mean_drive_bins(gamma_step(*unpack(q)), drives, edges)[post]
        return np.asarray(hp - (r @ hp) / (r @ r) * r, dtype=np.float64)

    best = None
    for b0 in (0.5, 1.5, 4.0):
        for d0 in (0.0, 1.0):
            head = [] if alpha is not None else [np.log(2.0)]
            lo = ([] if alpha is not None else [np.log(0.2)]) + [np.log(0.05), 0.0]
            hi = ([] if alpha is not None else [np.log(20.0)]) + [np.log(30.0), 4.0]
            sol = least_squares(resid, x0=np.array(head + [np.log(b0), d0]), bounds=(lo, hi))
            if best is None or sol.cost < best.cost:
                best = sol
    assert best is not None
    a_, b_, d_ = unpack(best.x)
    r = mean_drive_bins(gamma_step(a_, b_, d_), drives, edges)
    amp = float(r[post] @ hp / (r[post] @ r[post]))
    return KernelFit(
        alpha=a_,
        beta=b_,
        delta_s=d_,
        amp=amp,
        ssr=float(2 * best.cost),
        fitted=np.asarray(amp * r, dtype=np.float64),
    )


def step_quantile_times(
    step: StepResponse,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    horizon_s: float = 200.0,
    dt: float = 0.01,
) -> dict[float, float]:
    """Times at which the step response first reaches each fraction of its final value."""
    t = np.arange(0.0, horizon_s, dt)
    sr = step(t) / step(np.array([10 * horizon_s]))[0]
    return {q: float(t[np.argmax(sr >= q)]) for q in quantiles}
