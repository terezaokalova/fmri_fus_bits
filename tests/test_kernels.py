"""Step-response kernels and their fit."""

import numpy as np

from fmri_fus_bits.kernels import (
    boxcar_bins,
    canonical_bold_step,
    fit_gamma_kernel,
    gamma_step,
    mean_drive_bins,
    step_quantile_times,
)


def test_steps_have_unit_area_and_causality() -> None:
    t = np.array([-1.0, 0.0, 0.5, 200.0])
    s = gamma_step(0.8, 3.0, 0.5)(t)
    assert s[0] == 0.0 and s[1] == 0.0 and abs(s[-1] - 1.0) < 1e-12
    assert abs(canonical_bold_step(np.array([400.0]))[0] - 1.0) < 1e-12


def test_long_drive_reaches_unit_plateau() -> None:
    edges = np.arange(0.0, 61.0, 1.0)
    r = boxcar_bins(canonical_bold_step, 100.0, 0.0, edges)
    assert abs(r[-1] - 1.0) < 1e-3


def test_canonical_bold_peaks_near_5_s() -> None:
    q = step_quantile_times(canonical_bold_step, (0.5,))
    assert 5.0 < q[0.5] < 7.0


def test_kernel_fit_recovers_generating_kernel() -> None:
    edges = np.arange(-2.0, 17.0, 1.0)
    drives = np.random.default_rng(0).uniform(4.5, 5.7, 120)
    h = 0.7 * mean_drive_bins(gamma_step(2.5, 2.0, 0.6), drives, edges)
    fit = fit_gamma_kernel(h, drives, edges)
    assert np.max(np.abs(fit.fitted - h)) < 1e-4
    assert abs(fit.alpha - 2.5) < 0.1 and abs(fit.beta - 2.0) < 0.1 and abs(fit.delta_s - 0.6) < 0.1
    assert abs(fit.amp - 0.7) < 0.01
