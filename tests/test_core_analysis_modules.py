from __future__ import annotations

import numpy as np

from gpoloidal.analysis.hparam_sweep import SweepGrid, pivot_metric, run_2d_hparam_sweep
from gpoloidal.analysis.profiles import extract_band_profile, lognormal_bands_from_latent, sort_profile_by_coordinate
from gpoloidal.core.metrics import field_metrics, mean_chi2, weighted_mae, weighted_rmse


def test_core_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_est = np.array([1.0, 1.0, 4.0])
    m = field_metrics(y_est, y_true, prefix="t")
    assert set(["t_rmse", "t_mae", "t_rel_rmse", "t_corr", "t_neg_frac"]) <= set(m)
    assert m["t_neg_frac"] == 0.0

    sigma = np.ones_like(y_true) * 0.5
    chi2 = mean_chi2(y_est, y_true, sigma)
    assert chi2 > 0

    w = np.array([1.0, 2.0, 1.0])
    assert weighted_mae(y_est, y_true, w) > 0
    assert weighted_rmse(y_est, y_true, w) > 0


def test_analysis_profiles_helpers():
    x = np.array([0.4, 0.2, 0.3])
    y = np.array([4.0, 2.0, 3.0])
    xs, ys = sort_profile_by_coordinate(x, y)
    assert np.allclose(xs, [0.2, 0.3, 0.4])
    assert np.allclose(ys, [2.0, 3.0, 4.0])

    r = np.array([0.2, 0.3, 0.4, 0.5])
    z = np.array([0.0, 0.01, 0.03, 0.2])
    v = np.array([2.0, 3.0, 4.0, 5.0])
    xr, vr, mask = extract_band_profile(r, z, v, target=0.02, half_width=0.02)
    assert mask.sum() == 3
    assert np.allclose(xr, [0.2, 0.3, 0.4])
    assert np.allclose(vr, [2.0, 3.0, 4.0])

    bands = lognormal_bands_from_latent(np.array([0.0]), np.array([0.1]), nsigma=2.0)
    assert set(bands.keys()) == {"mean", "median", "lower", "upper"}
    assert float(bands["upper"][0]) > float(bands["lower"][0])


def test_analysis_hparam_sweep_helpers():
    grid = SweepGrid("length_scale", (0.04, 0.05), "obs_noise_level", (0.1, 0.2))

    def evaluator(length_scale, obs_noise_level):
        return {"mll": -(length_scale - 0.05) ** 2 - (obs_noise_level - 0.1) ** 2}

    df = run_2d_hparam_sweep(grid=grid, evaluator=evaluator)
    assert df.shape[0] == 4
    piv = pivot_metric(df, grid=grid, metric="mll")
    assert list(piv.index) == [0.04, 0.05]
    assert list(piv.columns) == [0.1, 0.2]
