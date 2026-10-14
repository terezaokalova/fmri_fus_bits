"""Limits and orderings of the neighbour treatments."""

import numpy as np

from fmri_fus_bits.equalizers import (
    interference_metric,
    neighbour_onsets,
    pattern_covariance,
    separable_gains,
    zero_forcing_variances,
)
from fmri_fus_bits.kernels import boxcar_bins, gamma_step
from fmri_fus_bits.noise import WhiteSlowNoise, ar1_bin_covariance

EDGES = np.arange(-2.0, 16.0 + 1.0, 1.0)


def noise_model(p: int = 4) -> WhiteSlowNoise:
    t_s = ar1_bin_covariance(0.9, 1.0, EDGES)
    return WhiteSlowNoise(
        ell=np.linspace(2.0, 1.0, p),
        f=np.linspace(0.8, 0.2, p),
        eps=np.full(p, 0.1),
        t_s=t_s,
        spectrum=np.zeros(1),
        omegas=np.zeros(1),
    )


def responses(period: float) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    step = gamma_step(1.5, 3.0, 1.0)
    r = boxcar_bins(step, 2.0, 0.0, EDGES)
    past, future = neighbour_onsets(period, EDGES[-1], 60.0)
    return (
        r,
        [boxcar_bins(step, 2.0, o, EDGES) for o in past],
        [boxcar_bins(step, 2.0, o, EDGES) for o in future],
    )


def test_isolated_trial_treatments_coincide() -> None:
    nm = noise_model()
    r, _, _ = responses(np.inf)
    var = zero_forcing_variances(nm, r, [])
    m = interference_metric(nm, r, [], np.eye(4))
    assert np.allclose(m, np.diag(1.0 / var))


def test_feedback_dominates_noise_treatment() -> None:
    nm = noise_model()
    r, past, future = responses(3.0)
    sig = pattern_covariance(*np.random.default_rng(0).standard_normal((2, 6, 4)))
    tin = interference_metric(nm, r, past + future, sig)
    dfe = interference_metric(nm, r, future, sig)
    assert np.linalg.eigvalsh(dfe - tin).min() > -1e-10


def test_zero_forcing_fails_when_target_is_in_neighbour_span() -> None:
    nm = noise_model()
    r, past, future = responses(3.0)
    iso = zero_forcing_variances(nm, r, [])
    assert np.all(iso / zero_forcing_variances(nm, r, past + future + [r]) < 1e-6)


def test_separable_gains_isolated_and_ordering() -> None:
    t_corr = ar1_bin_covariance(0.3, 0.72, EDGES)
    r, past, future = responses(np.inf)
    g_zf, tin, dfe = separable_gains(t_corr, r, past, future, 0.5, np.array([1.0, 2.0]))
    assert np.allclose(tin, g_zf / (1 + 0.5 * g_zf)) and np.allclose(dfe, tin)
    r, past, future = responses(4.0)
    _, tin, dfe = separable_gains(t_corr, r, past, future, 0.5, np.array([1.0, 2.0]))
    assert np.all(dfe >= tin) and tin[0] >= tin[1]
