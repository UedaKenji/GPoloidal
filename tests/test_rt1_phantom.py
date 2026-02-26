from __future__ import annotations

import numpy as np

from gpoloidal.rt1 import phantom


def test_double_phantom_applies_upper_z_mask_and_returns_finite_values():
    f = phantom.get_phantom_funtion("double")
    r = np.array([0.7, 0.7, 0.7], dtype=float)
    z = np.array([0.0, 0.49, 0.6], dtype=float)

    y = np.asarray(f(r, z), dtype=float)

    assert y.shape == z.shape
    assert np.all(np.isfinite(y))
    assert y[1] == 0.0
    assert y[2] == 0.0


def test_get_phantom_funtion_accepts_parameter_overrides():
    r = np.array([0.8, 0.9], dtype=float)
    z = np.array([0.1, 0.2], dtype=float)

    f_default = phantom.get_phantom_funtion("single", apply_sep_factor=False, z_clip_max=None)
    f_custom = phantom.get_phantom_funtion(
        "single",
        apply_sep_factor=False,
        z_clip_max=None,
        a=3.0,
        rmax=0.60,
    )

    y_default = np.asarray(f_default(r, z), dtype=float)
    y_custom = np.asarray(f_custom(r, z), dtype=float)

    assert y_default.shape == y_custom.shape
    assert np.all(np.isfinite(y_default))
    assert np.all(np.isfinite(y_custom))
    assert not np.allclose(y_default, y_custom)


def test_get_phantom_function_alias_matches_legacy_name():
    r = np.array([0.7], dtype=float)
    z = np.array([0.0], dtype=float)
    f1 = phantom.get_phantom_funtion("single")
    f2 = phantom.get_phantom_function("single")
    y1 = np.asarray(f1(r, z), dtype=float)
    y2 = np.asarray(f2(r, z), dtype=float)
    assert np.allclose(y1, y2)
